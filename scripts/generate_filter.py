#!/usr/bin/env python3
"""
GPU Audio Upsampler - Multi-Rate Filter Coefficient Generation

FIRフィルタを生成し、検証する。位相タイプ（最小位相/線形位相/混合位相）を選択可能。

サポートするアップサンプリング比率:
- 16x: 44.1kHz → 705.6kHz, 48kHz → 768kHz
- 8x:  88.2kHz → 705.6kHz, 96kHz → 768kHz
- 4x:  176.4kHz → 705.6kHz, 192kHz → 768kHz
- 2x:  352.8kHz → 705.6kHz, 384kHz → 768kHz

位相タイプ:
- minimum: 最小位相（プリリンギング排除、周波数依存遅延）
- linear: 線形位相（プリリンギングあり、全周波数で一定遅延）
- mixed: 混合位相（最小位相と線形位相のブレンド）

仕様:
- タップ数: 2,000,000 (2M) デフォルト
- 通過帯域: 0-20,000 Hz
- 阻止帯域: 入力Nyquist周波数以降
- 阻止帯域減衰: -197 dB以下
- 窓関数: Kaiser (β ≈ 55)

注意:
- タップ数はアップサンプリング比率の倍数であること
- クリッピング防止のため係数は正規化される
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class PhaseType(Enum):
    """フィルタの位相タイプ"""

    MINIMUM = "minimum"  # 最小位相: プリリンギングなし、周波数依存遅延
    LINEAR = "linear"  # 線形位相: プリリンギングあり、一定遅延
    MIXED = "mixed"  # 混合位相: 最小位相と線形位相のブレンド


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


@dataclass
class FilterConfig:
    """フィルタ生成の設定"""

    n_taps: int = 2_000_000
    input_rate: int = 44100
    upsample_ratio: int = 16
    passband_end: int = 20000
    stopband_start: int | None = None  # Noneの場合は入力Nyquist周波数
    stopband_attenuation_db: int = 197
    kaiser_beta: float = 55.0
    phase_type: PhaseType = PhaseType.MINIMUM
    mix_ratio: float = 0.5  # 混合位相用（0.0=線形, 1.0=最小）
    minimum_phase_method: MinimumPhaseMethod = MinimumPhaseMethod.HOMOMORPHIC
    output_prefix: str | None = None

    def __post_init__(self) -> None:
        if self.stopband_start is None:
            self.stopband_start = self.input_rate // 2

    @property
    def output_rate(self) -> int:
        return self.input_rate * self.upsample_ratio

    @property
    def family(self) -> str:
        return "44k" if self.input_rate % 44100 == 0 else "48k"

    @property
    def taps_label(self) -> str:
        if self.n_taps % 1_000_000 == 0:
            return f"{self.n_taps // 1_000_000}m"
        return str(self.n_taps)

    @property
    def base_name(self) -> str:
        if self.output_prefix:
            return self.output_prefix
        phase_suffix = self.phase_type.value
        if self.phase_type == PhaseType.MIXED:
            phase_suffix = f"mixed{int(self.mix_ratio * 100)}"
        return f"filter_{self.family}_{self.upsample_ratio}x_{self.taps_label}_{phase_suffix}"


class FilterDesigner:
    """フィルタ設計を担当するクラス"""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def design_linear_phase(self) -> np.ndarray:
        """線形位相FIRフィルタを設計する"""
        print("線形位相FIRフィルタ設計中...")
        print(f"  タップ数: {self.config.n_taps}")
        print(f"  出力サンプルレート: {self.config.output_rate} Hz")
        print(f"  通過帯域: 0-{self.config.passband_end} Hz")
        print(f"  阻止帯域: {self.config.stopband_start}+ Hz")

        cutoff_freq = (self.config.passband_end + self.config.stopband_start) / 2
        nyquist = self.config.output_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        print(f"  カットオフ周波数: {cutoff_freq} Hz (正規化: {normalized_cutoff:.6f})")
        print(f"  Kaiser β: {self.config.kaiser_beta}")

        # タイプIフィルタのため奇数タップに
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
        """線形位相フィルタを最小位相フィルタに変換する"""
        print("\n最小位相変換中...")

        n_fft = 2 ** int(np.ceil(np.log2(len(h_linear) * 8)))
        print(
            f"  警告: FFTサイズ {n_fft:,} は非常に大きいため、処理に時間がかかります（数分～数十分）"
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
        print(f"  FFTサイズ: {n_fft}")
        return h_min_phase

    def create_mixed_phase(
        self, h_linear: np.ndarray, h_min_phase: np.ndarray
    ) -> np.ndarray:
        """混合位相フィルタを生成する（インパルス応答のブレンド）"""
        print(f"\n混合位相フィルタ生成中（mix_ratio={self.config.mix_ratio:.2f}）...")

        # 線形位相の中心をt=0に揃えてからブレンド
        linear_center = len(h_linear) // 2
        h_linear_shifted = np.zeros(self.config.n_taps)

        # 線形位相の前半部分をコピー（必要な範囲のみ）
        copy_len = min(linear_center, self.config.n_taps)
        h_linear_shifted[:copy_len] = h_linear[linear_center : linear_center + copy_len]

        # mix_ratio: 1.0 = 完全最小位相, 0.0 = 完全線形位相
        h_mixed = (
            self.config.mix_ratio * h_min_phase
            + (1.0 - self.config.mix_ratio) * h_linear_shifted
        )

        print(f"  混合位相係数タップ数: {len(h_mixed)}")
        return h_mixed

    def design(self) -> tuple[np.ndarray, np.ndarray | None]:
        """
        設定に基づいてフィルタを設計する

        Returns:
            tuple: (最終フィルタ係数, 線形位相係数 or None)
        """
        # 1. 線形位相フィルタを設計（ベース）
        h_linear = self.design_linear_phase()

        if self.config.phase_type == PhaseType.LINEAR:
            # 線形位相をそのまま使用（タップ数を調整）
            if len(h_linear) > self.config.n_taps:
                # 中心を保持してトリム
                center = len(h_linear) // 2
                start = center - self.config.n_taps // 2
                h_final = h_linear[start : start + self.config.n_taps]
            else:
                h_final = h_linear
            return h_final, h_linear

        # 2. 最小位相変換
        h_min_phase = self.convert_to_minimum_phase(h_linear)

        if self.config.phase_type == PhaseType.MINIMUM:
            return h_min_phase, h_linear

        # 3. 混合位相
        h_mixed = self.create_mixed_phase(h_linear, h_min_phase)
        return h_mixed, h_linear


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

        if self.config.phase_type == PhaseType.MINIMUM:
            status = "確認" if results["is_minimum_phase"] else "未確認"
            print(f"  最小位相特性: {status}")
        elif self.config.phase_type == PhaseType.LINEAR:
            status = "確認" if results["is_symmetric"] else "未確認"
            print(f"  線形位相特性（対称性）: {status}")


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

        # 線形位相との比較（存在する場合）
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
    ) -> str:
        """フィルタを生成する"""
        # 0. タップ数の検証
        validate_tap_count(self.config.n_taps, self.config.upsample_ratio)

        # 1. フィルタ設計
        h_final, h_linear = self.designer.design()

        # 2. 係数正規化
        h_final, normalization_info = normalize_coefficients(h_final)

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

        return base_name

    def _create_metadata(self, validation_results: dict[str, Any]) -> dict[str, Any]:
        return {
            "generation_date": datetime.now().isoformat(),
            "n_taps": self.config.n_taps,
            "sample_rate_input": self.config.input_rate,
            "sample_rate_output": self.config.output_rate,
            "upsample_ratio": self.config.upsample_ratio,
            "passband_end_hz": self.config.passband_end,
            "stopband_start_hz": self.config.stopband_start,
            "target_stopband_attenuation_db": self.config.stopband_attenuation_db,
            "kaiser_beta": self.config.kaiser_beta,
            "phase_type": self.config.phase_type.value,
            "mix_ratio": self.config.mix_ratio
            if self.config.phase_type == PhaseType.MIXED
            else None,
            "minimum_phase_method": self.config.minimum_phase_method.value,
            "output_basename": self.config.base_name,
            "validation_results": validation_results,
        }

    def _print_report(
        self,
        validation_results: dict[str, Any],
        normalization_info: dict[str, Any],
        base_name: str,
    ) -> None:
        print("\n" + "=" * 70)
        print(f"Phase 1 完了 - {self.config.n_taps:,}タップフィルタ")
        print("=" * 70)
        print(f"位相タイプ: {self.config.phase_type.value.title()} Phase")
        print(f"{self.config.n_taps:,}タップFIRフィルタ生成完了")
        print(f"阻止帯域減衰: {validation_results['stopband_attenuation_db']:.1f} dB")
        spec_status = "合格" if validation_results["meets_stopband_spec"] else "不合格"
        print(f"  {spec_status} (目標: {self.config.stopband_attenuation_db} dB以上)")
        print(f"係数正規化: DCゲイン={normalization_info['normalized_dc_gain']:.6f}")
        print(f"係数ファイル: data/coefficients/{base_name}.bin")
        print("検証プロット: plots/analysis/")
        print("=" * 70)


# ==============================================================================
# 後方互換性のためのグローバル変数と関数
# ==============================================================================

# デフォルト定数（後方互換性のため維持）
N_TAPS = 2_000_000
SAMPLE_RATE_INPUT = 44100
UPSAMPLE_RATIO = 16
SAMPLE_RATE_OUTPUT = SAMPLE_RATE_INPUT * UPSAMPLE_RATIO
PASSBAND_END = 20000
STOPBAND_START = 22050
STOPBAND_ATTENUATION_DB = 197
KAISER_BETA = 55
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


def normalize_coefficients(h: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """フィルタ係数を正規化してクリッピングを防止する"""
    dc_gain = np.sum(h)

    if abs(dc_gain) < 1e-12:
        raise ValueError("DCゲインが0に近すぎます。フィルター係数が不正です。")

    h_normalized = h / dc_gain
    max_amplitude = np.max(np.abs(h_normalized))

    info = {
        "original_dc_gain": float(dc_gain),
        "normalized_dc_gain": float(np.sum(h_normalized)),
        "max_coefficient_amplitude": float(max_amplitude),
        "normalization_applied": True,
    }

    print("\n係数正規化:")
    print(f"  元のDCゲイン: {dc_gain:.6f}")
    print(f"  正規化後DCゲイン: {np.sum(h_normalized):.6f}")
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
    filter_infos: list[tuple[str, str, dict[str, Any]]],
    output_dir: str = "data/coefficients",
    taps: int = 2_000_000,
) -> None:
    """全フィルタ情報をまとめたC++ヘッダファイルを生成する"""
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
        f.write(f"constexpr size_t FILTER_TAPS = {taps};\n\n")
        f.write("// Multi-rate filter configurations\n")
        f.write("struct FilterConfig {\n")
        f.write("    const char* name;\n")
        f.write("    const char* filename;\n")
        f.write("    int32_t input_rate;\n")
        f.write("    int32_t output_rate;\n")
        f.write("    int32_t ratio;\n")
        f.write("};\n\n")
        f.write(f"constexpr size_t FILTER_COUNT = {len(filter_infos)};\n\n")
        f.write("constexpr FilterConfig FILTER_CONFIGS[FILTER_COUNT] = {\n")
        for name, base_name, cfg in filter_infos:
            output_rate = cfg["input_rate"] * cfg["ratio"]
            f.write(
                f'    {{"{name}", "{base_name}.bin", '
                f'{cfg["input_rate"]}, {output_rate}, {cfg["ratio"]}}},\n'
            )
        f.write("};\n\n")
        f.write("#endif // FILTER_COEFFICIENTS_H\n")

    print(f"\nマルチレートヘッダファイル生成: {header_path}")


# ==============================================================================
# CLI用関数
# ==============================================================================


def generate_single_filter(
    args: argparse.Namespace, filter_name: str | None = None, skip_header: bool = False
) -> str:
    """単一フィルタを生成する"""
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
        mix_ratio=args.mix_ratio,
        minimum_phase_method=MinimumPhaseMethod(args.minimum_phase_method),
        output_prefix=args.output_prefix,
    )

    generator = FilterGenerator(config)
    return generator.generate(filter_name, skip_header)


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
    print("=" * 70)
    print("\nTarget configurations:")
    for name, cfg in configs.items():
        output_rate = cfg["input_rate"] * cfg["ratio"]
        print(f"  {name}: {cfg['input_rate']}Hz × {cfg['ratio']}x → {output_rate}Hz")

    if args.output_prefix:
        print("\n注意: --output-prefix は --generate-all 時は無視されます")
    print()

    results = []
    filter_infos = []

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
            base_name = generate_single_filter(
                filter_args, filter_name=name, skip_header=True
            )
            results.append((name, "Success"))
            filter_infos.append((name, base_name, cfg))
        except Exception as e:
            results.append((name, f"Failed: {e}"))
            print(f"ERROR: {e}")

    if filter_infos:
        generate_multi_rate_header(filter_infos, taps=args.taps)

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
  # Generate single minimum phase filter (default)
  %(prog)s --input-rate 44100 --upsample-ratio 16

  # Generate linear phase filter
  %(prog)s --phase-type linear

  # Generate mixed phase filter (50%% minimum, 50%% linear)
  %(prog)s --phase-type mixed --mix-ratio 0.5

  # Generate all 8 filter configurations
  %(prog)s --generate-all

  # Generate only 44.1kHz family with linear phase
  %(prog)s --generate-all --family 44k --phase-type linear
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
        default=2_000_000,
        help="Number of filter taps. Default: 2000000 (2M)",
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
        default=197,
        help="Target stopband attenuation (dB). Default: 197",
    )
    parser.add_argument(
        "--kaiser-beta",
        type=float,
        default=55.0,
        help="Kaiser window beta. Default: 55",
    )
    parser.add_argument(
        "--phase-type",
        type=str,
        choices=["minimum", "linear", "mixed"],
        default="minimum",
        help="Phase type: minimum (no pre-ringing), linear (symmetric), mixed (blend). Default: minimum",
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.5,
        help="Mix ratio for mixed phase (0.0=linear, 1.0=minimum). Default: 0.5",
    )
    parser.add_argument(
        "--minimum-phase-method",
        type=str,
        choices=["homomorphic", "hilbert"],
        default="homomorphic",
        help="Minimum phase conversion method. Default: homomorphic",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file basename (without extension). Default: auto",
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
