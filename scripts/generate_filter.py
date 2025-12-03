#!/usr/bin/env python3
"""
GPU Audio Upsampler - Multi-Rate Filter Coefficient Generation

FIRフィルタを生成し、検証する。位相タイプ（最小位相/ハイブリッド位相）を選択可能。

サポートするアップサンプリング比率:
- 16x: 44.1kHz → 705.6kHz, 48kHz → 768kHz
- 8x:  88.2kHz → 705.6kHz, 96kHz → 768kHz
- 4x:  176.4kHz → 705.6kHz, 192kHz → 768kHz
- 2x:  352.8kHz → 705.6kHz, 384kHz → 768kHz

位相タイプ:
- minimum: 最小位相（プリリンギング排除、周波数依存遅延）【従来】
- hybrid: 低域（≤150Hz）最小位相 + 高域線形位相（群遅延10ms整列）【新規】

仕様:
- タップ数: 640,000 (640k) デフォルト
- 通過帯域: 0-20,000 Hz
- 阻止帯域: 入力Nyquist周波数以降
- 阻止帯域減衰: -160 dB以下 (24bit品質に十分、最小位相変換後の現実的値)
- 窓関数: Kaiser (β ≈ 28 / 32bit Float実装の量子ノイズ限界に合わせた最適値)

注意:
- 最小位相/ハイブリッド: タップ数はアップサンプリング比率の倍数であること
- クリッピング防止のため係数は正規化される
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# GPU高速化（CuPy）のオプショナルサポート
try:
    import cupy as cp
    from cupyx.scipy import fft as cp_fft

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cp_fft = None


class PhaseType(Enum):
    """フィルタの位相タイプ"""

    MINIMUM = "minimum"  # 最小位相: プリリンギングなし、周波数依存遅延
    HYBRID = "hybrid"  # 低域最小位相 + 高域線形位相（10ms整列）


class MinimumPhaseMethod(Enum):
    """最小位相変換の手法"""

    HOMOMORPHIC = "homomorphic"  # ホモモルフィック法（デフォルト、高精度）
    HILBERT = "hilbert"  # ヒルベルト変換法（高速だが精度やや劣る）


# マルチレート設定
# 44.1kHz系と48kHz系、それぞれ16x/8x/4x/2xの組み合わせ
MULTI_RATE_CONFIGS = {
    # 44.1kHz family -> 705.6kHz output
    "44k_16x": {"input_rate": 44100, "ratio": 16, "stopband": 22050},
    "44k_8x": {"input_rate": 88200, "ratio": 8, "stopband": 44100},
    "44k_4x": {"input_rate": 176400, "ratio": 4, "stopband": 88200},
    "44k_2x": {"input_rate": 352800, "ratio": 2, "stopband": 176400},
    # 48kHz family -> 768kHz output
    "48k_16x": {"input_rate": 48000, "ratio": 16, "stopband": 24000},
    "48k_8x": {"input_rate": 96000, "ratio": 8, "stopband": 48000},
    "48k_4x": {"input_rate": 192000, "ratio": 4, "stopband": 96000},
    "48k_2x": {"input_rate": 384000, "ratio": 2, "stopband": 192000},
}

HYBRID_DEFAULT_CROSSOVER_HZ = 150.0
HYBRID_DEFAULT_TRANSITION_HZ = 40.0
HYBRID_DEFAULT_DELAY_MS = 10.0
HYBRID_DEFAULT_FAST_WINDOW = 32_768
HYBRID_FAST_WINDOW_TARGET = 0.99


@dataclass
class FilterConfig:
    """フィルタ生成の設定"""

    n_taps: int = 640_000
    input_rate: int = 44100
    upsample_ratio: int = 16
    passband_end: int = 20000
    stopband_start: int | None = None  # Noneの場合は入力Nyquist周波数
    stopband_attenuation_db: int = 160  # 24bit品質に十分、最小位相変換後の現実的値
    kaiser_beta: float = 28.0
    phase_type: PhaseType = PhaseType.MINIMUM
    minimum_phase_method: MinimumPhaseMethod = MinimumPhaseMethod.HOMOMORPHIC
    hybrid_crossover_hz: float = HYBRID_DEFAULT_CROSSOVER_HZ
    hybrid_transition_hz: float = HYBRID_DEFAULT_TRANSITION_HZ
    hybrid_delay_ms: float = HYBRID_DEFAULT_DELAY_MS
    hybrid_fast_window_samples: int = HYBRID_DEFAULT_FAST_WINDOW
    # DCゲインはゼロ詰めアップサンプル後の振幅を維持するためにアップサンプル比に合わせる
    # 全レートで音量統一のため target_dc_gain × dc_gain_factor に設定
    target_dc_gain: float | None = None
    dc_gain_factor: float = 0.99  # 音量統一用係数（-0.09dB）
    output_prefix: str | None = None

    def __post_init__(self) -> None:
        # バリデーション
        if self.n_taps <= 0:
            raise ValueError(f"タップ数は正の整数である必要があります: {self.n_taps}")
        if self.input_rate <= 0:
            raise ValueError(
                f"入力レートは正の整数である必要があります: {self.input_rate}"
            )
        if self.upsample_ratio <= 0:
            raise ValueError(
                f"アップサンプリング比率は正の整数である必要があります: {self.upsample_ratio}"
            )
        if self.kaiser_beta < 0:
            raise ValueError(
                f"カイザーベータは非負である必要があります: {self.kaiser_beta}"
            )

        # Nyquist制約チェック
        nyquist = self.input_rate // 2
        if self.passband_end > nyquist:
            raise ValueError(
                f"パスバンド終端 ({self.passband_end} Hz) は入力ナイキスト周波数 ({nyquist} Hz) 以下である必要があります"
            )

        if self.stopband_start is None:
            self.stopband_start = nyquist
        elif self.stopband_start <= self.passband_end:
            raise ValueError(
                f"ストップバンド開始 ({self.stopband_start} Hz) はパスバンド終端 ({self.passband_end} Hz) より大きい必要があります"
            )

        # ストップバンドが出力ナイキスト以上の場合はエラー
        output_nyquist = self.input_rate * self.upsample_ratio // 2
        if self.stopband_start >= output_nyquist:
            raise ValueError(
                f"ストップバンド開始 ({self.stopband_start} Hz) は出力ナイキスト周波数 ({output_nyquist} Hz) 未満である必要があります"
            )

        # DCゲインターゲットの設定（指定がなければアップサンプル比）
        if self.target_dc_gain is None:
            self.target_dc_gain = float(self.upsample_ratio)
        if self.target_dc_gain <= 0:
            raise ValueError(
                f"DCゲインのターゲットは正の値である必要があります: {self.target_dc_gain}"
            )
        # dc_gain_factor のバリデーション
        if not 0 < self.dc_gain_factor <= 1.0:
            raise ValueError(
                f"dc_gain_factorは0より大きく1.0以下である必要があります: {self.dc_gain_factor}"
            )

        if self.phase_type == PhaseType.HYBRID:
            if not (0 < self.hybrid_crossover_hz < self.passband_end):
                raise ValueError(
                    "hybrid_crossover_hz は0より大きく通過帯域終端未満である必要があります"
                )
            if self.hybrid_transition_hz <= 0:
                raise ValueError("hybrid_transition_hz は正の値である必要があります")
            if self.hybrid_delay_ms <= 0:
                raise ValueError("hybrid_delay_ms は正の値である必要があります")
            if self.hybrid_fast_window_samples <= 0:
                raise ValueError(
                    "hybrid_fast_window_samples は正の整数である必要があります"
                )
            if self.hybrid_delay_samples >= self.n_taps:
                raise ValueError(
                    f"hybrid_delay_ms に対応するサンプル数 ({self.hybrid_delay_samples}) がタップ数 ({self.n_taps}) 以上です"
                )

    @property
    def output_rate(self) -> int:
        return self.input_rate * self.upsample_ratio

    @property
    def hybrid_delay_seconds(self) -> float:
        return self.hybrid_delay_ms / 1000.0

    @property
    def hybrid_delay_samples(self) -> int:
        return int(round(self.hybrid_delay_seconds * self.output_rate))

    @property
    def family(self) -> str:
        return "44k" if self.input_rate % 44100 == 0 else "48k"

    @property
    def final_taps(self) -> int:
        """最終的なタップ数（ハイブリッド/最小位相は指定値を維持）"""
        return self.n_taps

    @property
    def taps_label(self) -> str:
        """ファイル名用のタップ数ラベル（パディング後の実タップ数を使用）

        2,000,000 taps -> "2m" for shorter filenames
        """
        if self.final_taps == 2_000_000:
            return "2m"
        if self.final_taps == 640_000:
            return "2m"  # ファイル名互換性のため2mを維持
        return str(self.final_taps)

    @property
    def phase_label(self) -> str:
        """ファイル名用の位相タイプラベル

        C++ expects "min_phase" for minimum phase filters
        """
        if self.phase_type == PhaseType.MINIMUM:
            return "min_phase"
        if self.phase_type == PhaseType.HYBRID:
            return "hybrid_phase"
        return self.phase_type.value

    @property
    def base_name(self) -> str:
        if self.output_prefix:
            return self.output_prefix
        return f"filter_{self.family}_{self.upsample_ratio}x_{self.taps_label}_{self.phase_label}"


class FilterDesigner:
    """フィルタ設計を担当するクラス"""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def design_linear_phase(self) -> np.ndarray:
        """ベースとなる線形位相FIRフィルタを設計する"""
        print("基準線形位相FIRフィルタ設計中...")
        print(f"  指定タップ数: {self.config.n_taps}")
        print(f"  出力サンプルレート: {self.config.output_rate} Hz")
        print(f"  通過帯域: 0-{self.config.passband_end} Hz")
        print(f"  阻止帯域: {self.config.stopband_start}+ Hz")

        cutoff_freq = (self.config.passband_end + self.config.stopband_start) / 2
        nyquist = self.config.output_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        print(f"  カットオフ周波数: {cutoff_freq} Hz (正規化: {normalized_cutoff:.6f})")
        print(f"  Kaiser β: {self.config.kaiser_beta}")

        # 偶数タップの場合は+1して奇数長を作り、後段の最小位相変換/ハイブリッド化でトリミング
        numtaps = (
            self.config.n_taps
            if self.config.n_taps % 2 == 1
            else self.config.n_taps + 1
        )

        h_linear = signal.firwin(
            numtaps=numtaps,
            cutoff=normalized_cutoff,
            window=("kaiser", self.config.kaiser_beta),
            fs=1.0,
            scale=True,
        )

        print(f"  実際のタップ数: {len(h_linear)}")
        return h_linear

    def convert_to_minimum_phase(self, h_linear: np.ndarray) -> np.ndarray:
        """線形位相フィルタを最小位相フィルタに変換する

        CuPyが利用可能な場合はGPU高速化版を使用する。
        """
        print("\n最小位相変換中...")

        n_fft = 2 ** int(np.ceil(np.log2(len(h_linear) * 8)))
        print(f"  FFTサイズ: {n_fft:,}")

        # GPU高速化（CuPyが利用可能な場合）
        if (
            CUPY_AVAILABLE
            and self.config.minimum_phase_method == MinimumPhaseMethod.HOMOMORPHIC
        ):
            print("  🚀 GPU高速化（CuPy）を使用")
            h_min_phase = self._convert_to_minimum_phase_gpu(h_linear, n_fft)
        else:
            if not CUPY_AVAILABLE:
                print("  ⚠️ CuPyが利用できません。CPU版を使用（時間がかかります）")
            else:
                print(
                    f"  CPU版を使用（method={self.config.minimum_phase_method.value}）"
                )

            h_min_phase = signal.minimum_phase(
                h_linear, method=self.config.minimum_phase_method.value, n_fft=n_fft
            )

        # 元のタップ数に合わせる
        if len(h_min_phase) > self.config.n_taps:
            h_min_phase = h_min_phase[: self.config.n_taps]
        elif len(h_min_phase) < self.config.n_taps:
            h_min_phase = np.pad(
                h_min_phase, (0, self.config.n_taps - len(h_min_phase))
            )

        print(f"  最小位相係数タップ数: {len(h_min_phase)}")
        return h_min_phase

    def design_hybrid_phase(self, h_linear: np.ndarray) -> np.ndarray:
        """ハイブリッド位相フィルタを設計する"""
        print("\nハイブリッド位相フィルタ合成中...")
        h_min_phase = self.convert_to_minimum_phase(h_linear)

        n_fft = 2 ** int(np.ceil(np.log2(self.config.n_taps * 4)))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.config.output_rate)

        H_min = np.fft.rfft(h_min_phase, n=n_fft)
        H_linear = np.fft.rfft(h_linear, n=n_fft)

        magnitude = np.maximum(np.abs(H_linear), 1e-12)
        phase_min = np.unwrap(np.angle(H_min))
        phase_linear = -2 * np.pi * freqs * self.config.hybrid_delay_seconds

        low_weight = self._hybrid_lowpass_weight(freqs)
        high_weight = 1.0 - low_weight

        phase_hybrid = low_weight * phase_min + high_weight * phase_linear
        H_hybrid = magnitude * np.exp(1j * phase_hybrid)

        h_time = np.fft.irfft(H_hybrid, n=n_fft).real
        h_time = h_time[: self.config.n_taps]
        print(
            f"  ハイブリッド: クロスオーバー {self.config.hybrid_crossover_hz} Hz, "
            f"遅延 {self.config.hybrid_delay_ms} ms"
        )
        return h_time

    def _hybrid_lowpass_weight(self, freqs: np.ndarray) -> np.ndarray:
        """クロスオーバー周波数で滑らかに接続するための重みを計算"""
        crossover = self.config.hybrid_crossover_hz
        width = self.config.hybrid_transition_hz
        start = max(0.0, crossover - width / 2.0)
        end = crossover + width / 2.0

        weights = np.ones_like(freqs)
        weights[freqs >= end] = 0.0
        transition_mask = (freqs > start) & (freqs < end)
        if np.any(transition_mask):
            phase = (freqs[transition_mask] - start) / max(end - start, 1e-9)
            weights[transition_mask] = 0.5 * (1 + np.cos(np.pi * phase))
        return weights

    def _convert_to_minimum_phase_gpu(
        self, h_linear: np.ndarray, n_fft: int
    ) -> np.ndarray:
        """CuPyを使用したGPU高速化版の最小位相変換（ホモモルフィック法）

        scipy.signal.minimum_phase のホモモルフィック法をGPU上で実装。
        """
        import time

        start_time = time.time()

        # GPU上のメモリに転送
        h_gpu = cp.asarray(h_linear, dtype=cp.float64)
        h_padded = cp.zeros(n_fft, dtype=cp.float64)
        h_padded[: len(h_linear)] = h_gpu

        # 1. FFTで周波数領域へ
        H = cp_fft.fft(h_padded)

        # 2. 対数マグニチュード（ホモモルフィック法）
        # 数値安定性のため小さな値を追加
        eps = cp.finfo(cp.float64).eps
        log_H = cp.log(cp.maximum(cp.abs(H), eps))

        # 3. ケプストラム（逆FFT）
        cepstrum = cp_fft.ifft(log_H).real

        # 4. 因果的ケプストラムを作成（最小位相のため）
        # cepstrum[0] はそのまま、cepstrum[1:n_fft//2] は2倍、cepstrum[n_fft//2+1:] は0
        causal_cepstrum = cp.zeros_like(cepstrum)
        causal_cepstrum[0] = cepstrum[0]
        if n_fft % 2 == 0:
            causal_cepstrum[1 : n_fft // 2] = 2 * cepstrum[1 : n_fft // 2]
            causal_cepstrum[n_fft // 2] = cepstrum[n_fft // 2]
        else:
            causal_cepstrum[1 : (n_fft + 1) // 2] = 2 * cepstrum[1 : (n_fft + 1) // 2]

        # 5. FFTで周波数領域へ戻り、指数関数で元に戻す
        H_min = cp.exp(cp_fft.fft(causal_cepstrum))

        # 6. 逆FFTで時間領域へ
        h_min_phase_gpu = cp_fft.ifft(H_min).real

        # CPU側に転送して半分の長さを返す（scipy.minimum_phaseと同じ）
        h_min_phase = cp.asnumpy(h_min_phase_gpu[: (len(h_linear) + 1) // 2])

        elapsed = time.time() - start_time
        print(f"  GPU処理時間: {elapsed:.2f}秒")

        return h_min_phase

    def design(self) -> tuple[np.ndarray, np.ndarray | None]:
        """
        設定に基づいてフィルタを設計する

        Returns:
            tuple: (最終フィルタ係数, 基準線形位相係数 or None)
        """
        # 1. 基準線形位相フィルタを設計
        h_linear = self.design_linear_phase()

        if self.config.phase_type == PhaseType.MINIMUM:
            h_min_phase = self.convert_to_minimum_phase(h_linear)
            return h_min_phase, h_linear
        if self.config.phase_type == PhaseType.HYBRID:
            h_hybrid = self.design_hybrid_phase(h_linear)
            return h_hybrid, h_linear
        raise ValueError(f"Unsupported phase type: {self.config.phase_type}")


class FilterValidator:
    """フィルタ係数の検証を担当するクラス"""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def validate(self, h: np.ndarray) -> dict[str, Any]:
        """フィルタ係数が仕様を満たしているか検証する"""
        print("\n仕様検証中...")

        w, H = signal.freqz(h, worN=16384, fs=self.config.output_rate)
        H_db = 20 * np.log10(np.abs(H) + 1e-12)

        # 通過帯域のリップル計算
        passband_mask = w <= self.config.passband_end
        passband_db = H_db[passband_mask]
        passband_ripple_db = np.max(passband_db) - np.min(passband_db)

        # 阻止帯域の減衰量計算
        stopband_mask = w >= self.config.stopband_start
        stopband_attenuation = np.min(H_db[stopband_mask])

        # 位相特性の検証
        peak_idx = np.argmax(np.abs(h))
        mid_point = len(h) // 2
        energy_first_half = np.sum(h[:mid_point] ** 2)
        energy_second_half = np.sum(h[mid_point:] ** 2)
        energy_ratio = energy_first_half / (energy_second_half + 1e-12)

        peak_threshold = int(len(h) * 0.01)
        is_peak_at_front = peak_idx < peak_threshold
        is_energy_causal = energy_ratio > 10

        # 線形位相の対称性チェック
        is_symmetric = self._check_symmetry(h)

        fast_window = min(len(h), self.config.hybrid_fast_window_samples)
        fast_energy = float(np.sum(h[:fast_window] ** 2))
        total_energy = float(np.sum(h**2) + 1e-24)
        fast_energy_ratio = fast_energy / total_energy if total_energy > 0 else 0.0
        fast_target_ratio = (
            HYBRID_FAST_WINDOW_TARGET
            if self.config.phase_type == PhaseType.HYBRID
            else None
        )

        results = {
            "passband_ripple_db": float(passband_ripple_db),
            "stopband_attenuation_db": float(abs(stopband_attenuation)),
            "peak_position": int(peak_idx),
            "peak_threshold_samples": int(peak_threshold),
            "energy_ratio_first_to_second_half": float(energy_ratio),
            "meets_stopband_spec": bool(
                abs(stopband_attenuation) >= self.config.stopband_attenuation_db
            ),
            "is_minimum_phase": bool(is_peak_at_front and is_energy_causal),
            "is_symmetric": is_symmetric,
            "phase_type": self.config.phase_type.value,
            "actual_taps": len(h),
            "fast_window_samples": int(fast_window),
            "fast_window_energy_ratio": float(fast_energy_ratio),
            "fast_window_target_ratio": fast_target_ratio,
            "meets_fast_window_spec": bool(
                fast_energy_ratio >= fast_target_ratio
                if fast_target_ratio is not None
                else True
            ),
        }

        self._print_results(results, stopband_attenuation)
        return results

    def _check_symmetry(self, h: np.ndarray, tolerance: float = 1e-10) -> bool:
        """線形位相フィルタの対称性をチェック"""
        return bool(np.allclose(h, h[::-1], atol=tolerance))

    def _print_results(
        self, results: dict[str, Any], stopband_attenuation: float
    ) -> None:
        print(f"  位相タイプ: {results['phase_type']}")
        print(f"  実際のタップ数: {results['actual_taps']}")
        print(f"  通過帯域リップル: {results['passband_ripple_db']:.3f} dB")
        print(
            f"  阻止帯域減衰: {abs(stopband_attenuation):.1f} dB (目標: {self.config.stopband_attenuation_db} dB)"
        )
        print(
            f"  阻止帯域スペック: {'合格' if results['meets_stopband_spec'] else '不合格'}"
        )
        print(
            f"  ピーク位置: サンプル {results['peak_position']} "
            f"(先頭1%={results['peak_threshold_samples']}サンプル以内: "
            f"{'Y' if results['peak_position'] < results['peak_threshold_samples'] else 'N'})"
        )
        print(
            f"  エネルギー比(前半/後半): {results['energy_ratio_first_to_second_half']:.1f}"
        )
        fast_pct = results["fast_window_energy_ratio"] * 100.0
        fast_samples = results["fast_window_samples"]
        fast_target = results.get("fast_window_target_ratio")
        if fast_target is not None:
            target_pct = fast_target * 100.0
            status = "合格" if results["meets_fast_window_spec"] else "要確認"
            print(
                f"  Fast window ({fast_samples} taps) energy: {fast_pct:.2f}% "
                f"(target ≥ {target_pct:.1f}%) → {status}"
            )
        else:
            print(
                f"  Fast window ({fast_samples} taps) energy: {fast_pct:.2f}% (参考値)"
            )

        if self.config.phase_type == PhaseType.MINIMUM:
            status = "確認" if results["is_minimum_phase"] else "未確認"
            print(f"  最小位相特性: {status}")
        elif self.config.phase_type == PhaseType.HYBRID:
            print(
                f"  ハイブリッド: crossover={self.config.hybrid_crossover_hz} Hz, "
                f"delay={self.config.hybrid_delay_ms} ms"
            )


class FilterExporter:
    """フィルタ係数のエクスポートを担当するクラス"""

    def __init__(
        self, config: FilterConfig, output_dir: str = "data/coefficients"
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)

    def export(
        self, h: np.ndarray, metadata: dict[str, Any], skip_header: bool = False
    ) -> str:
        """フィルタ係数をエクスポートする"""
        print(f"\n係数エクスポート中... ({self.output_dir})")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        base_name = self.config.base_name

        # 1. バイナリ形式（float32）
        self._export_binary(h, base_name)

        # 2. C++ヘッダファイル
        if not skip_header:
            self._export_header(h, metadata, base_name)

        # 3. メタデータJSON
        self._export_metadata(metadata, base_name)

        return base_name

    def _export_binary(self, h: np.ndarray, base_name: str) -> None:
        h_float32 = h.astype(np.float32)
        binary_path = self.output_dir / f"{base_name}.bin"
        h_float32.tofile(binary_path)
        file_size_mb = binary_path.stat().st_size / (1024 * 1024)
        print(f"  保存: {binary_path} ({file_size_mb:.2f} MB)")

    def _export_header(
        self, h: np.ndarray, metadata: dict[str, Any], base_name: str
    ) -> None:
        header_path = self.output_dir / "filter_coefficients.h"
        with open(header_path, "w") as f:
            f.write("// Auto-generated filter coefficients\n")
            f.write("// GPU Audio Upsampler - Phase 1\n")
            f.write(f"// Generated: {metadata['generation_date']}\n\n")
            f.write("#ifndef FILTER_COEFFICIENTS_H\n")
            f.write("#define FILTER_COEFFICIENTS_H\n\n")
            f.write("#include <cstddef>\n\n")
            f.write(f"constexpr size_t FILTER_TAPS = {len(h)};\n")
            f.write(
                f"constexpr int SAMPLE_RATE_INPUT = {metadata['sample_rate_input']};\n"
            )
            f.write(
                f"constexpr int SAMPLE_RATE_OUTPUT = {metadata['sample_rate_output']};\n"
            )
            f.write(f"constexpr int UPSAMPLE_RATIO = {metadata['upsample_ratio']};\n\n")
            f.write("// Filter coefficients are stored in external .bin files.\n")
            f.write(f"// Default binary: {base_name}.bin\n\n")
            f.write("#endif // FILTER_COEFFICIENTS_H\n")
        print(f"  保存: {header_path}")

    def _export_metadata(self, metadata: dict[str, Any], base_name: str) -> None:
        metadata_path = self.output_dir / f"{base_name}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  保存: {metadata_path}")


class FilterPlotter:
    """フィルタ特性のプロットを担当するクラス"""

    def __init__(
        self, config: FilterConfig, output_dir: str = "plots/analysis"
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)

    def plot(
        self,
        h_final: np.ndarray,
        h_linear: np.ndarray | None = None,
        filter_name: str | None = None,
    ) -> None:
        """フィルタ特性をプロットする"""
        print(f"\nプロット生成中... ({self.output_dir})")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{filter_name}_" if filter_name else ""

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        self._plot_impulse_response(h_final, h_linear, prefix)
        self._plot_frequency_response(h_final, h_linear, prefix)
        self._plot_phase_response(h_final, h_linear, prefix)

    def _plot_impulse_response(
        self, h_final: np.ndarray, h_linear: np.ndarray | None, prefix: str
    ) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 最終フィルタのインパルス応答
        display_range = min(4000, len(h_final))
        t = np.arange(display_range)
        h_display = h_final[:display_range]

        axes[0].plot(t, h_display, linewidth=0.5, color="orange")
        title = f"{self.config.phase_type.value.title()} Phase Impulse Response"
        axes[0].set_title(title, fontsize=12)
        axes[0].set_xlabel("Sample")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(0, color="r", linestyle="--", alpha=0.5, label="t=0")
        axes[0].legend()

        # 基準線形位相との比較（存在する場合）
        if h_linear is not None:
            center = len(h_linear) // 2
            display_range_lin = min(2000, center)
            t_linear = np.arange(-display_range_lin, display_range_lin)
            h_linear_center = h_linear[
                center - display_range_lin : center + display_range_lin
            ]

            axes[1].plot(t_linear, h_linear_center, linewidth=0.5)
            axes[1].set_title(
                "Linear Phase Impulse Response (Center Region)", fontsize=12
            )
            axes[1].set_xlabel("Sample")
            axes[1].set_ylabel("Amplitude")
            axes[1].grid(True, alpha=0.3)
            axes[1].axvline(0, color="r", linestyle="--", alpha=0.5, label="Center")
            axes[1].legend()
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}impulse_response.png", dpi=150)
        print(f"  保存: {prefix}impulse_response.png")
        plt.close()

    def _plot_frequency_response(
        self, h_final: np.ndarray, h_linear: np.ndarray | None, prefix: str
    ) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        w_final, H_final = signal.freqz(h_final, worN=16384, fs=self.config.output_rate)
        H_final_db = 20 * np.log10(np.abs(H_final) + 1e-12)

        # 全体表示
        axes[0].plot(
            w_final / 1000,
            H_final_db,
            label=f"{self.config.phase_type.value.title()} Phase",
            linewidth=1,
            alpha=0.7,
        )

        if h_linear is not None:
            w_lin, H_lin = signal.freqz(
                h_linear, worN=16384, fs=self.config.output_rate
            )
            H_lin_db = 20 * np.log10(np.abs(H_lin) + 1e-12)
            axes[0].plot(
                w_lin / 1000, H_lin_db, label="Linear Phase", linewidth=1, alpha=0.5
            )

        axes[0].set_title("Magnitude Response (Full Range)", fontsize=12)
        axes[0].set_xlabel("Frequency (kHz)")
        axes[0].set_ylabel("Magnitude (dB)")
        axes[0].set_ylim(-200, 5)
        axes[0].axhline(
            -180, color="r", linestyle="--", alpha=0.5, label="-180dB Target"
        )
        axes[0].axvline(
            self.config.passband_end / 1000,
            color="g",
            linestyle="--",
            alpha=0.5,
            label="Passband End",
        )
        axes[0].axvline(
            self.config.stopband_start / 1000,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label="Stopband Start",
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 通過帯域詳細
        passband_mask = w_final <= self.config.passband_end * 1.1
        axes[1].plot(
            w_final[passband_mask] / 1000,
            H_final_db[passband_mask],
            linewidth=1,
            color="orange",
        )
        axes[1].set_title("Magnitude Response (Passband Detail)", fontsize=12)
        axes[1].set_xlabel("Frequency (kHz)")
        axes[1].set_ylabel("Magnitude (dB)")
        axes[1].axvline(
            self.config.passband_end / 1000,
            color="g",
            linestyle="--",
            alpha=0.5,
            label="Passband End",
        )
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}frequency_response.png", dpi=150)
        print(f"  保存: {prefix}frequency_response.png")
        plt.close()

    def _plot_phase_response(
        self, h_final: np.ndarray, h_linear: np.ndarray | None, prefix: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(14, 6))

        w, H_final = signal.freqz(h_final, worN=8192, fs=self.config.output_rate)
        phase_final = np.unwrap(np.angle(H_final))

        ax.plot(
            w / 1000,
            phase_final,
            label=f"{self.config.phase_type.value.title()} Phase",
            linewidth=1,
            alpha=0.7,
        )

        if h_linear is not None:
            _, H_lin = signal.freqz(h_linear, worN=8192, fs=self.config.output_rate)
            phase_lin = np.unwrap(np.angle(H_lin))
            ax.plot(w / 1000, phase_lin, label="Linear Phase", linewidth=1, alpha=0.5)

        ax.set_title("Phase Response", fontsize=12)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Phase (radians)")
        ax.axvline(
            self.config.passband_end / 1000,
            color="g",
            linestyle="--",
            alpha=0.5,
            label="Passband End",
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}phase_response.png", dpi=150)
        print(f"  保存: {prefix}phase_response.png")
        plt.close()


class FilterGenerator:
    """フィルタ生成のオーケストレーションを担当するクラス"""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self.designer = FilterDesigner(config)
        self.validator = FilterValidator(config)
        self.exporter = FilterExporter(config)
        self.plotter = FilterPlotter(config)

    def generate(
        self, filter_name: str | None = None, skip_header: bool = False
    ) -> tuple[str, int]:
        """フィルタを生成する

        Returns:
            tuple: (base_name, actual_taps) - ファイル名のベースと実タップ数
        """
        # 0. タップ数の検証（polyphase要件）
        validate_tap_count(self.config.n_taps, self.config.upsample_ratio)

        # 1. フィルタ設計
        h_final, h_linear = self.designer.design()

        # 2. 係数正規化
        h_final, normalization_info = normalize_coefficients(
            h_final,
            target_dc_gain=self.config.target_dc_gain,
            dc_gain_factor=self.config.dc_gain_factor,
        )

        # 3. 仕様検証
        validation_results = self.validator.validate(h_final)
        validation_results["normalization"] = normalization_info

        # 4. プロット生成
        self.plotter.plot(h_final, h_linear, filter_name)

        # 5. メタデータ作成
        metadata = self._create_metadata(validation_results)

        # 6. 係数エクスポート
        base_name = self.exporter.export(h_final, metadata, skip_header)

        # 7. 最終レポート
        self._print_report(validation_results, normalization_info, base_name)

        # 実タップ数はフィルタ長から取得（validation_resultsに記録済み）
        actual_taps = validation_results["actual_taps"]

        return base_name, actual_taps

    def _create_metadata(self, validation_results: dict[str, Any]) -> dict[str, Any]:
        return {
            "generation_date": datetime.now().isoformat(),
            "n_taps_specified": self.config.n_taps,
            "n_taps_actual": validation_results.get(
                "actual_taps", self.config.final_taps
            ),
            "sample_rate_input": self.config.input_rate,
            "sample_rate_output": self.config.output_rate,
            "upsample_ratio": self.config.upsample_ratio,
            "passband_end_hz": self.config.passband_end,
            "stopband_start_hz": self.config.stopband_start,
            "target_stopband_attenuation_db": self.config.stopband_attenuation_db,
            "kaiser_beta": self.config.kaiser_beta,
            "phase_type": self.config.phase_type.value,
            "minimum_phase_method": self.config.minimum_phase_method.value,
            "hybrid_crossover_hz": self.config.hybrid_crossover_hz,
            "hybrid_transition_hz": self.config.hybrid_transition_hz,
            "hybrid_delay_ms": self.config.hybrid_delay_ms,
            "hybrid_fast_window_samples": self.config.hybrid_fast_window_samples,
            "hybrid_fast_window_target_ratio": HYBRID_FAST_WINDOW_TARGET,
            "target_dc_gain": self.config.target_dc_gain,
            "output_basename": self.config.base_name,
            "validation_results": validation_results,
        }

    def _print_report(
        self,
        validation_results: dict[str, Any],
        normalization_info: dict[str, Any],
        base_name: str,
    ) -> None:
        actual_taps = validation_results.get("actual_taps", self.config.final_taps)
        print("\n" + "=" * 70)
        if actual_taps != self.config.n_taps:
            print(
                f"完了 - {self.config.n_taps:,}→{actual_taps:,}タップフィルタ（パディング）"
            )
        else:
            print(f"完了 - {actual_taps:,}タップフィルタ")
        print("=" * 70)
        print(f"位相タイプ: {self.config.phase_type.value.title()} Phase")
        print(f"阻止帯域減衰: {validation_results['stopband_attenuation_db']:.1f} dB")
        spec_status = "合格" if validation_results["meets_stopband_spec"] else "不合格"
        print(f"  {spec_status} (目標: {self.config.stopband_attenuation_db} dB以上)")
        print(
            "係数正規化: "
            f"目標DC={normalization_info['target_dc_gain']:.6f}, "
            f"結果DC={normalization_info['normalized_dc_gain']:.6f}"
        )
        if "fast_window_energy_ratio" in validation_results:
            ratio = validation_results["fast_window_energy_ratio"] * 100.0
            fast_samples = validation_results["fast_window_samples"]
            fast_target = validation_results.get("fast_window_target_ratio")
            if fast_target is not None:
                target = fast_target * 100.0
                status = "✅" if validation_results["meets_fast_window_spec"] else "⚠️"
                print(
                    f"{status} Fast window energy ({fast_samples} taps): "
                    f"{ratio:.2f}% (target ≥ {target:.1f}%)"
                )
            else:
                print(
                    f"Fast window energy ({fast_samples} taps): {ratio:.2f}% (参考値)"
                )
        max_coef = normalization_info.get("max_coefficient_amplitude", 0)
        print(f"最大係数振幅: {max_coef:.6f}")
        if max_coef > 1.0:
            print("  ⚠️ CUDA側で補正が必要（#260参照）")
        print(
            f"係数ファイル: data/coefficients/{base_name}.bin ({actual_taps:,} coeffs)"
        )
        print("検証プロット: plots/analysis/")
        print("=" * 70)


# ==============================================================================
# 後方互換性のためのグローバル変数と関数
# ==============================================================================

# デフォルト定数（後方互換性のため維持）
N_TAPS = 640_000
SAMPLE_RATE_INPUT = 44100
UPSAMPLE_RATIO = 16
SAMPLE_RATE_OUTPUT = SAMPLE_RATE_INPUT * UPSAMPLE_RATIO
PASSBAND_END = 20000
STOPBAND_START = 22050
STOPBAND_ATTENUATION_DB = 160  # 24bit品質に十分
KAISER_BETA = 28
OUTPUT_PREFIX = None


def validate_tap_count(taps: int, upsample_ratio: int) -> None:
    """タップ数がアップサンプリング比率の倍数であることを確認する"""
    if taps % upsample_ratio != 0:
        raise ValueError(
            f"タップ数 {taps:,} はアップサンプリング比率 {upsample_ratio} の倍数である必要があります。"
            f"\n  推奨: {(taps // upsample_ratio) * upsample_ratio:,} または "
            f"{((taps // upsample_ratio) + 1) * upsample_ratio:,}"
        )
    print(f"タップ数 {taps:,} は {upsample_ratio} の倍数です")


def compute_padded_taps(n_taps: int, upsample_ratio: int) -> int:
    """比率の倍数になる最小のタップ数を計算する

    GPUポリフェーズ分割のため、タップ数は比率の倍数が必要。
    線形位相フィルタは設計時にこの値を使用する。

    Returns:
        int: 比率の倍数になる最小のタップ数 (>= n_taps)
    """
    if n_taps % upsample_ratio == 0:
        return n_taps
    return ((n_taps // upsample_ratio) + 1) * upsample_ratio


def normalize_coefficients(
    h: np.ndarray,
    target_dc_gain: float = 1.0,
    dc_gain_factor: float = 0.99,  # DCゲイン係数（音量統一用）
) -> tuple[np.ndarray, dict[str, Any]]:
    """フィルタ係数を正規化する（DCゲイン統一 + L1ノルム出力版）

    Args:
        h: フィルタ係数配列
        target_dc_gain: 目標DCゲイン（アップサンプル比L）
        dc_gain_factor: DCゲイン係数（デフォルト0.99 = -0.09dB）

    Note:
        全レートで音量を統一するため、DCゲイン = L × dc_gain_factor に設定。
        L1ノルムはグローバル安全ゲイン計算用にメタデータに出力。
    """
    if h.size == 0:
        raise ValueError("フィルタ係数が空です。")

    if target_dc_gain <= 0:
        raise ValueError("DCゲインのターゲットは正の値である必要があります。")

    if not 0 < dc_gain_factor <= 1.0:
        raise ValueError("dc_gain_factorは0より大きく1.0以下である必要があります。")

    dc_gain = float(np.sum(h))

    if abs(dc_gain) < 1e-12:
        raise ValueError("DCゲインが0に近すぎます。フィルター係数が不正です。")

    # DCゲインを target × dc_gain_factor に正規化
    actual_target = target_dc_gain * dc_gain_factor
    scale = actual_target / dc_gain
    h_normalized = h * scale

    final_dc_gain = float(np.sum(h_normalized))
    max_amplitude = float(np.max(np.abs(h_normalized)))

    # L1ノルム計算（グローバル安全ゲイン計算用）
    l1_norm = float(np.sum(np.abs(h_normalized)))

    info = {
        "original_dc_gain": dc_gain,
        "target_dc_gain": float(target_dc_gain),
        "dc_gain_factor": dc_gain_factor,
        "normalized_dc_gain": final_dc_gain,
        "applied_scale": float(scale),
        "l1_norm": l1_norm,
        "l1_norm_ratio": l1_norm / target_dc_gain,
        "max_coefficient_amplitude": max_amplitude,
        "normalization_applied": True,
    }

    print("\n係数正規化:")
    print(
        f"  目標DCゲイン: {target_dc_gain:.6f} × {dc_gain_factor} = {actual_target:.6f}"
    )
    print(f"  元のDCゲイン: {dc_gain:.6f}")
    print(f"  正規化スケール: {scale:.6f}x")
    print(f"  最終DCゲイン: {final_dc_gain:.6f}")
    print(f"  L1ノルム: {l1_norm:.6f} (L1/L = {l1_norm / target_dc_gain:.6f})")
    print(f"  最大係数振幅: {max_amplitude:.6f}")

    return h_normalized, info


def design_linear_phase_filter() -> np.ndarray:
    """線形位相FIRフィルタを設計する（後方互換性）"""
    config = FilterConfig(
        n_taps=N_TAPS,
        input_rate=SAMPLE_RATE_INPUT,
        upsample_ratio=UPSAMPLE_RATIO,
        passband_end=PASSBAND_END,
        stopband_start=STOPBAND_START,
        kaiser_beta=KAISER_BETA,
    )
    designer = FilterDesigner(config)
    return designer.design_linear_phase()


def convert_to_minimum_phase(h_linear: np.ndarray) -> np.ndarray:
    """線形位相フィルタを最小位相フィルタに変換する（後方互換性）"""
    config = FilterConfig(
        n_taps=N_TAPS,
        input_rate=SAMPLE_RATE_INPUT,
        upsample_ratio=UPSAMPLE_RATIO,
    )
    designer = FilterDesigner(config)
    return designer.convert_to_minimum_phase(h_linear)


def validate_specifications(h: np.ndarray) -> dict[str, Any]:
    """フィルタ係数が仕様を満たしているか検証する（後方互換性）"""
    config = FilterConfig(
        n_taps=N_TAPS,
        input_rate=SAMPLE_RATE_INPUT,
        upsample_ratio=UPSAMPLE_RATIO,
        passband_end=PASSBAND_END,
        stopband_start=STOPBAND_START,
        stopband_attenuation_db=STOPBAND_ATTENUATION_DB,
    )
    validator = FilterValidator(config)
    return validator.validate(h)


def plot_responses(
    h_linear: np.ndarray,
    h_min_phase: np.ndarray,
    output_dir: str = "plots/analysis",
    filter_name: str | None = None,
) -> None:
    """フィルタ特性をプロットする（後方互換性）"""
    config = FilterConfig(
        n_taps=N_TAPS,
        input_rate=SAMPLE_RATE_INPUT,
        upsample_ratio=UPSAMPLE_RATIO,
        passband_end=PASSBAND_END,
        stopband_start=STOPBAND_START,
    )
    plotter = FilterPlotter(config, output_dir)
    plotter.plot(h_min_phase, h_linear, filter_name)


def export_coefficients(
    h: np.ndarray,
    metadata: dict[str, Any],
    output_dir: str = "data/coefficients",
    skip_header: bool = False,
) -> str:
    """フィルタ係数をエクスポートする（後方互換性）"""
    config = FilterConfig(
        n_taps=N_TAPS,
        input_rate=SAMPLE_RATE_INPUT,
        upsample_ratio=UPSAMPLE_RATIO,
        output_prefix=OUTPUT_PREFIX,
    )
    exporter = FilterExporter(config, output_dir)
    return exporter.export(h, metadata, skip_header)


def generate_multi_rate_header(
    filter_infos: list[tuple[str, str, int, dict[str, Any]]],
    output_dir: str = "data/coefficients",
) -> None:
    """全フィルタ情報をまとめたC++ヘッダファイルを生成する

    Args:
        filter_infos: [(name, base_name, actual_taps, cfg), ...] のリスト
        output_dir: 出力ディレクトリ
    """
    output_path = Path(output_dir)
    header_path = output_path / "filter_coefficients.h"

    with open(header_path, "w") as f:
        f.write("// Auto-generated multi-rate filter coefficients\n")
        f.write("// GPU Audio Upsampler - Multi-Rate Support\n")
        f.write(f"// Generated: {datetime.now().isoformat()}\n\n")
        f.write("#ifndef FILTER_COEFFICIENTS_H\n")
        f.write("#define FILTER_COEFFICIENTS_H\n\n")
        f.write("#include <cstddef>\n")
        f.write("#include <cstdint>\n\n")
        f.write("// Multi-rate filter configurations\n")
        f.write("struct FilterConfig {\n")
        f.write("    const char* name;\n")
        f.write("    const char* filename;\n")
        f.write(
            "    size_t taps;        // Actual tap count (matches .bin file length)\n"
        )
        f.write("    int32_t input_rate;\n")
        f.write("    int32_t output_rate;\n")
        f.write("    int32_t ratio;\n")
        f.write("};\n\n")
        f.write(f"constexpr size_t FILTER_COUNT = {len(filter_infos)};\n\n")
        f.write("constexpr FilterConfig FILTER_CONFIGS[FILTER_COUNT] = {\n")
        for name, base_name, actual_taps, cfg in filter_infos:
            output_rate = cfg["input_rate"] * cfg["ratio"]
            f.write(
                f'    {{"{name}", "{base_name}.bin", {actual_taps}, '
                f'{cfg["input_rate"]}, {output_rate}, {cfg["ratio"]}}},\n'
            )
        f.write("};\n\n")
        f.write("#endif // FILTER_COEFFICIENTS_H\n")

    print(f"\nマルチレートヘッダファイル生成: {header_path}")


def calculate_safe_gain(
    filter_infos: list[tuple[str, str, int, dict[str, Any]]],
    safety_margin: float = 0.97,
    coefficients_dir: str = "data/coefficients",
) -> dict[str, Any]:
    """全フィルタからグローバル安全ゲインを計算する

    Args:
        filter_infos: [(name, base_name, actual_taps, cfg), ...] のリスト
        safety_margin: 安全マージン M（デフォルト0.97 = -0.26dB）
        coefficients_dir: 係数ディレクトリ

    Returns:
        dict: {
            "l1_max": float,
            "l1_max_filter": str,
            "max_coef_max": float,
            "max_coef_max_filter": str,
            "safety_margin": float,
            "recommended_gain": float,
            "details": list[dict],
        }
    """
    coeff_path = Path(coefficients_dir)
    details = []
    l1_max = 0.0
    l1_max_filter = ""
    max_coef_max = 0.0
    max_coef_max_filter = ""

    for name, base_name, _, _ in filter_infos:
        json_path = coeff_path / f"{base_name}.json"
        if not json_path.exists():
            print(f"  警告: {json_path} が見つかりません。スキップします。")
            continue

        with open(json_path, encoding="utf-8") as f:
            metadata = json.load(f)

        norm_info = metadata.get("validation_results", {}).get("normalization", {})
        l1_norm = norm_info.get("l1_norm")
        max_coef = norm_info.get("max_coefficient_amplitude")

        # None または無効な値のチェック（安全なFloat変換）
        if l1_norm is None or not isinstance(l1_norm, (int, float)):
            print(f"  警告: {name} のL1ノルムが無効です。スキップします。")
            continue
        if max_coef is None or not isinstance(max_coef, (int, float)):
            print(f"  警告: {name} のmax_coefficientが無効です。スキップします。")
            continue

        # 明示的にfloatに変換（int/float混在対策）
        l1_norm = float(l1_norm)
        max_coef = float(max_coef)

        details.append(
            {
                "name": name,
                "l1_norm": l1_norm,
                "max_coef": max_coef,
            }
        )

        if l1_norm > l1_max:
            l1_max = l1_norm
            l1_max_filter = name
        if max_coef > max_coef_max:
            max_coef_max = max_coef
            max_coef_max_filter = name

    # 安全ゲイン計算（max_coefベース）
    # H = M / max_coef_max
    # これにより max_coef × H ≤ M < 1.0 を保証
    if max_coef_max > 0:
        recommended_gain = float(safety_margin / max_coef_max)
    else:
        recommended_gain = 1.0

    # gain が 1.0 を超える場合は 1.0 に制限（増幅は不要）
    if recommended_gain > 1.0:
        recommended_gain = 1.0

    return {
        "l1_max": l1_max,
        "l1_max_filter": l1_max_filter,
        "max_coef_max": max_coef_max,
        "max_coef_max_filter": max_coef_max_filter,
        "safety_margin": float(safety_margin),
        "recommended_gain": recommended_gain,
        "details": details,
    }


def print_safe_gain_recommendation(safe_gain_info: dict[str, Any]) -> None:
    """安全ゲインの推奨値を表示する"""
    print("\n" + "=" * 70)
    print("GLOBAL SAFE GAIN RECOMMENDATION")
    print("=" * 70)
    print(f"L1_max: {safe_gain_info['l1_max']:.2f} ({safe_gain_info['l1_max_filter']})")
    print(
        f"max_coef_max: {safe_gain_info['max_coef_max']:.6f} "
        f"({safe_gain_info['max_coef_max_filter']})"
    )
    print(f"Safety margin M: {safe_gain_info['safety_margin']}")
    print()

    gain = safe_gain_info["recommended_gain"]
    if gain < 1.0:
        print("⚠️  max_coef > 1.0 detected. Gain adjustment required.")
        print(f"Recommended config.json gain: {gain:.4f}")
        print()
        print("To apply, set in config.json:")
        print(f'  "gain": {gain:.4f}')
    else:
        print("✅ All filters have max_coef <= 1.0. No gain adjustment needed.")
        print('config.json gain can remain at: "gain": 1.0')

    print("=" * 70)


# ==============================================================================
# CLI用関数
# ==============================================================================


def generate_single_filter(
    args: argparse.Namespace, filter_name: str | None = None, skip_header: bool = False
) -> tuple[str, int]:
    """単一フィルタを生成する

    Returns:
        tuple: (base_name, actual_taps) - ファイル名のベースと実タップ数
    """
    global SAMPLE_RATE_INPUT, UPSAMPLE_RATIO, SAMPLE_RATE_OUTPUT
    global PASSBAND_END, STOPBAND_START, STOPBAND_ATTENUATION_DB, KAISER_BETA
    global N_TAPS, OUTPUT_PREFIX

    # グローバル変数を更新（後方互換性のため）
    SAMPLE_RATE_INPUT = args.input_rate
    UPSAMPLE_RATIO = args.upsample_ratio
    SAMPLE_RATE_OUTPUT = SAMPLE_RATE_INPUT * UPSAMPLE_RATIO
    PASSBAND_END = args.passband_end
    STOPBAND_START = (
        args.stopband_start if args.stopband_start else (SAMPLE_RATE_INPUT // 2)
    )
    STOPBAND_ATTENUATION_DB = args.stopband_attenuation
    KAISER_BETA = args.kaiser_beta
    N_TAPS = args.taps
    OUTPUT_PREFIX = args.output_prefix

    # 設定を作成
    config = FilterConfig(
        n_taps=args.taps,
        input_rate=args.input_rate,
        upsample_ratio=args.upsample_ratio,
        passband_end=args.passband_end,
        stopband_start=args.stopband_start,
        stopband_attenuation_db=args.stopband_attenuation,
        kaiser_beta=args.kaiser_beta,
        phase_type=PhaseType(args.phase_type),
        minimum_phase_method=MinimumPhaseMethod(args.minimum_phase_method),
        output_prefix=args.output_prefix,
        hybrid_crossover_hz=args.hybrid_crossover_hz,
        hybrid_transition_hz=args.hybrid_transition_hz,
        hybrid_delay_ms=args.hybrid_delay_ms,
        hybrid_fast_window_samples=args.hybrid_fast_window,
    )

    generator = FilterGenerator(config)
    return generator.generate(filter_name, skip_header)


def _generate_filter_worker(
    worker_args: tuple,
) -> tuple[str, str, int, dict, str | None]:
    """並列処理用のワーカー関数

    Args:
        worker_args: (name, cfg, args_dict) のタプル

    Returns:
        (name, base_name, actual_taps, cfg, error_message) のタプル
        成功時は error_message = None
    """

    name, cfg, args_dict = worker_args

    try:
        # FilterConfigを直接作成（グローバル変数に依存しない）
        config = FilterConfig(
            n_taps=args_dict["taps"],
            input_rate=cfg["input_rate"],
            upsample_ratio=cfg["ratio"],
            passband_end=args_dict["passband_end"],
            stopband_start=cfg["stopband"],
            stopband_attenuation_db=args_dict["stopband_attenuation"],
            kaiser_beta=args_dict["kaiser_beta"],
            phase_type=PhaseType(args_dict["phase_type"]),
            minimum_phase_method=MinimumPhaseMethod(args_dict["minimum_phase_method"]),
            output_prefix=None,
            hybrid_crossover_hz=args_dict["hybrid_crossover_hz"],
            hybrid_transition_hz=args_dict["hybrid_transition_hz"],
            hybrid_delay_ms=args_dict["hybrid_delay_ms"],
            hybrid_fast_window_samples=args_dict["hybrid_fast_window_samples"],
        )

        generator = FilterGenerator(config)
        base_name, actual_taps = generator.generate(filter_name=name, skip_header=True)
        return (name, base_name, actual_taps, cfg, None)
    except Exception as e:
        return (name, "", 0, cfg, str(e))


def generate_all_filters(args: argparse.Namespace) -> None:
    """全フィルタを一括生成する"""
    import copy

    if args.family == "44k":
        configs = {k: v for k, v in MULTI_RATE_CONFIGS.items() if k.startswith("44k")}
    elif args.family == "48k":
        configs = {k: v for k, v in MULTI_RATE_CONFIGS.items() if k.startswith("48k")}
    else:
        configs = MULTI_RATE_CONFIGS

    total = len(configs)
    print("=" * 70)
    print(f"Multi-Rate Filter Generation - {total} filters")
    print(f"Phase Type: {args.phase_type}")
    if hasattr(args, "parallel") and args.parallel:
        workers = (
            args.workers
            if hasattr(args, "workers") and args.workers
            else os.cpu_count()
        )
        print(f"Parallel Mode: {workers} workers")
    print("=" * 70)
    print("\nTarget configurations:")
    for name, cfg in configs.items():
        output_rate = cfg["input_rate"] * cfg["ratio"]
        print(f"  {name}: {cfg['input_rate']}Hz × {cfg['ratio']}x → {output_rate}Hz")

    if args.output_prefix:
        print("\n注意: --output-prefix は --generate-all 時は無視されます")
    print()

    # argsを辞書に変換（pickleできるようにする）
    args_dict = {
        "taps": args.taps,
        "passband_end": args.passband_end,
        "stopband_attenuation": args.stopband_attenuation,
        "kaiser_beta": args.kaiser_beta,
        "phase_type": args.phase_type,
        "minimum_phase_method": args.minimum_phase_method,
        "hybrid_crossover_hz": args.hybrid_crossover_hz,
        "hybrid_transition_hz": args.hybrid_transition_hz,
        "hybrid_delay_ms": args.hybrid_delay_ms,
        "hybrid_fast_window_samples": args.hybrid_fast_window,
    }

    results = []
    filter_infos = []

    # 並列処理の判定
    use_parallel = hasattr(args, "parallel") and args.parallel
    workers = (
        args.workers if hasattr(args, "workers") and args.workers else os.cpu_count()
    )

    if use_parallel and total > 1:
        # マルチプロセス並列処理
        print(f"\n並列処理開始（{workers}ワーカー）...")
        worker_args_list = [(name, cfg, args_dict) for name, cfg in configs.items()]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for name, base_name, actual_taps, cfg, error in executor.map(
                _generate_filter_worker, worker_args_list
            ):
                if error:
                    results.append((name, f"Failed: {error}"))
                    print(f"  ❌ {name}: {error}")
                else:
                    results.append((name, "Success"))
                    filter_infos.append((name, base_name, actual_taps, cfg))
                    print(f"  ✅ {name}: completed")
    else:
        # 逐次処理（GPUが1つの場合はこちらの方が効率的）
        for i, (name, cfg) in enumerate(configs.items(), 1):
            print("\n" + "=" * 70)
            print(f"[{i}/{total}] Generating {name}...")
            print("=" * 70)

            filter_args = copy.copy(args)
            filter_args.input_rate = cfg["input_rate"]
            filter_args.upsample_ratio = cfg["ratio"]
            filter_args.stopband_start = cfg["stopband"]
            filter_args.output_prefix = None

            try:
                base_name, actual_taps = generate_single_filter(
                    filter_args, filter_name=name, skip_header=True
                )
                results.append((name, "Success"))
                filter_infos.append((name, base_name, actual_taps, cfg))
            except Exception as e:
                results.append((name, f"Failed: {e}"))
                print(f"ERROR: {e}")

    if filter_infos:
        generate_multi_rate_header(filter_infos)

        # グローバル安全ゲインを計算して推奨値を表示
        safe_gain_info = calculate_safe_gain(filter_infos)
        print_safe_gain_recommendation(safe_gain_info)

    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    for name, status in results:
        print(f"  {name}: {status}")
    print("=" * 70)

    success_count = sum(1 for _, s in results if s == "Success")
    print(f"\nCompleted: {success_count}/{total} filters generated successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FIR filter coefficients with selectable phase type.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single minimum phase filter (default, recommended)
  %(prog)s --input-rate 44100 --upsample-ratio 16

  # Generate hybrid phase filter (150 Hz crossover, 10 ms delay)
  %(prog)s --phase-type hybrid --hybrid-crossover-hz 120

  # Generate all 8 filter configurations
  %(prog)s --generate-all

  # Generate only 44.1kHz family
  %(prog)s --generate-all --family 44k

  # Generate all filters in parallel (CPU multiprocessing)
  %(prog)s --generate-all --parallel

  # Generate with specific number of workers
  %(prog)s --generate-all --parallel --workers 4

Phase Types:
  minimum  - No pre-ringing, frequency-dependent delay (RECOMMENDED)
  hybrid   - Minimum phase below crossover, linear phase above with aligned delay

GPU Acceleration:
  Install CuPy for GPU-accelerated minimum phase conversion:
    uv pip install cupy-cuda12x  # For CUDA 12.x
  Or add to pyproject.toml: uv sync --extra gpu
""",
    )
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Generate all 8 filter configurations (44k/48k × 16x/8x/4x/2x)",
    )
    parser.add_argument(
        "--family",
        type=str,
        choices=["44k", "48k", "all"],
        default="all",
        help="Rate family to generate (only with --generate-all). Default: all",
    )
    parser.add_argument(
        "--input-rate",
        type=int,
        default=44100,
        help="Input sample rate (Hz). Default: 44100",
    )
    parser.add_argument(
        "--upsample-ratio",
        type=int,
        default=16,
        help="Upsampling ratio. Default: 16",
    )
    parser.add_argument(
        "--taps",
        type=int,
        default=640_000,
        help="Number of filter taps. Default: 640000 (640k). Auto-adjusted to ratio multiple.",
    )
    parser.add_argument(
        "--passband-end",
        type=int,
        default=20000,
        help="Passband end frequency (Hz). Default: 20000",
    )
    parser.add_argument(
        "--stopband-start",
        type=int,
        default=None,
        help="Stopband start frequency (Hz). Default: auto (input Nyquist)",
    )
    parser.add_argument(
        "--stopband-attenuation",
        type=int,
        default=160,
        help="Target stopband attenuation (dB). Default: 160 (sufficient for 24-bit)",
    )
    parser.add_argument(
        "--kaiser-beta",
        type=float,
        default=28.0,
        help="Kaiser window beta. Default: 28 (32bit Float実装の量子ノイズ限界に合わせた最適値)",
    )
    parser.add_argument(
        "--phase-type",
        type=str,
        choices=["minimum", "hybrid"],
        default="minimum",
        help="Phase type: minimum (recommended) or hybrid. Default: minimum",
    )
    parser.add_argument(
        "--minimum-phase-method",
        type=str,
        choices=["homomorphic", "hilbert"],
        default="homomorphic",
        help="Minimum phase conversion method. Default: homomorphic",
    )
    parser.add_argument(
        "--hybrid-crossover-hz",
        type=float,
        default=HYBRID_DEFAULT_CROSSOVER_HZ,
        help="Hybrid crossover frequency separating minimum and linear regions (Hz). Default: 150",
    )
    parser.add_argument(
        "--hybrid-transition-hz",
        type=float,
        default=HYBRID_DEFAULT_TRANSITION_HZ,
        help="Hybrid transition width around the crossover (Hz). Default: 40",
    )
    parser.add_argument(
        "--hybrid-delay-ms",
        type=float,
        default=HYBRID_DEFAULT_DELAY_MS,
        help="Absolute delay applied to the linear-phase portion (ms). Default: 10",
    )
    parser.add_argument(
        "--hybrid-fast-window",
        type=int,
        default=HYBRID_DEFAULT_FAST_WINDOW,
        help="Fast-partition window size used for energy checks (samples). Default: 32768",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file basename (without extension). Default: auto",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for --generate-all (CPU multiprocessing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel mode. Default: CPU count",
    )
    return parser.parse_args()


def main() -> None:
    """メイン処理"""
    args = parse_args()

    if args.generate_all:
        generate_all_filters(args)
    else:
        print("=" * 70)
        print("GPU Audio Upsampler - Filter Coefficient Generation")
        print("=" * 70)
        generate_single_filter(args)


if __name__ == "__main__":
    main()
