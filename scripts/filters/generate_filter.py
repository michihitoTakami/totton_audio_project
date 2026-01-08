#!/usr/bin/env python3
"""Shared filter generation utilities for the Totton Audio Project toolkit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class MinimumPhaseMethod(Enum):
    """最小位相変換に利用できる手法"""

    HOMOMORPHIC = "homomorphic"
    HILBERT = "hilbert"


MULTI_RATE_CONFIGS = {
    "44k_16x": {"input_rate": 44100, "ratio": 16, "stopband": 22050},
    "44k_8x": {"input_rate": 88200, "ratio": 8, "stopband": 44100},
    "44k_4x": {"input_rate": 176400, "ratio": 4, "stopband": 88200},
    "44k_2x": {"input_rate": 352800, "ratio": 2, "stopband": 176400},
    "48k_16x": {"input_rate": 48000, "ratio": 16, "stopband": 24000},
    "48k_8x": {"input_rate": 96000, "ratio": 8, "stopband": 48000},
    "48k_4x": {"input_rate": 192000, "ratio": 4, "stopband": 96000},
    "48k_2x": {"input_rate": 384000, "ratio": 2, "stopband": 192000},
}


@dataclass
class FilterConfig:
    """フィルタ生成共通設定"""

    n_taps: int = 640_000
    input_rate: int = 44100
    upsample_ratio: int = 16
    passband_end: int = 20000
    stopband_start: int | None = None
    stopband_attenuation_db: int = 160
    kaiser_beta: float = 28.0
    minimum_phase_method: MinimumPhaseMethod = MinimumPhaseMethod.HOMOMORPHIC
    target_dc_gain: float | None = None
    dc_gain_factor: float = 0.99
    output_prefix: str | None = None
    phase_suffix: str = "min_phase"

    def __post_init__(self) -> None:
        if self.n_taps <= 0:
            raise ValueError("タップ数は正の整数である必要があります")
        if self.input_rate <= 0:
            raise ValueError("入力レートは正の整数である必要があります")
        if self.upsample_ratio <= 0:
            raise ValueError("アップサンプリング比率は正の整数である必要があります")
        if self.kaiser_beta < 0:
            raise ValueError("Kaiser βは非負である必要があります")

        nyquist = self.input_rate // 2
        if self.passband_end > nyquist:
            raise ValueError(
                "パスバンド終端は入力ナイキスト周波数以下である必要があります"
            )

        if self.stopband_start is None:
            self.stopband_start = nyquist
        elif self.stopband_start <= self.passband_end:
            raise ValueError(
                "ストップバンド開始はパスバンド終端より大きい必要があります"
            )

        output_nyquist = self.input_rate * self.upsample_ratio // 2
        if self.stopband_start >= output_nyquist:
            raise ValueError(
                "ストップバンド開始は出力ナイキスト周波数未満である必要があります"
            )

        if self.target_dc_gain is None:
            self.target_dc_gain = float(self.upsample_ratio)
        if self.target_dc_gain <= 0:
            raise ValueError("DCゲインのターゲットは正の値である必要があります")
        if not 0 < self.dc_gain_factor <= 1.0:
            raise ValueError("dc_gain_factorは0より大きく1.0以下でなければなりません")

    @property
    def output_rate(self) -> int:
        return self.input_rate * self.upsample_ratio

    @property
    def family(self) -> str:
        return "44k" if self.input_rate % 44100 == 0 else "48k"

    @property
    def final_taps(self) -> int:
        return self.n_taps

    @property
    def taps_label(self) -> str:
        if self.final_taps == 2_000_000:
            return "2m"
        if self.final_taps == 640_000:
            return "2m"
        return str(self.final_taps)

    @property
    def base_name(self) -> str:
        if self.output_prefix:
            return self.output_prefix
        return f"filter_{self.family}_{self.upsample_ratio}x_{self.taps_label}_{self.phase_suffix}"


def kaiser_window(numtaps: int, beta: float) -> np.ndarray:
    """Kaiser窓を生成する"""
    return np.kaiser(numtaps, beta)


def compute_frequency_response(
    h: np.ndarray, fs: int, worN: int = 16384
) -> tuple[np.ndarray, np.ndarray]:
    """周波数応答を計算する"""
    return signal.freqz(h, worN=worN, fs=fs)


def save_coefficients(
    h: np.ndarray,
    metadata: dict[str, Any],
    config: FilterConfig,
    output_dir: str = "data/coefficients",
    skip_header: bool = False,
) -> str:
    exporter = FilterExporter(config, output_dir)
    return exporter.export(h, metadata, skip_header)


def plot_analysis(
    h_final: np.ndarray,
    config: FilterConfig,
    h_linear: np.ndarray | None = None,
    filter_name: str | None = None,
    output_dir: str = "plots/analysis",
) -> None:
    plotter = FilterPlotter(config, output_dir)
    plotter.plot(h_final, h_linear, filter_name)


class FilterExporter:
    """フィルタ係数をエクスポートする"""

    def __init__(
        self, config: FilterConfig, output_dir: str = "data/coefficients"
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)

    def export(
        self, h: np.ndarray, metadata: dict[str, Any], skip_header: bool = False
    ) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base_name = self.config.base_name
        self._export_binary(h, base_name)
        if not skip_header:
            self._export_header(h, metadata, base_name)
        self._export_metadata(metadata, base_name)
        return base_name

    def _export_binary(self, h: np.ndarray, base_name: str) -> None:
        binary_path = self.output_dir / f"{base_name}.bin"
        h.astype(np.float32).tofile(binary_path)
        size_mb = binary_path.stat().st_size / (1024.0 * 1024.0)
        print(f"  保存: {binary_path} ({size_mb:.2f} MB)")

    def _export_header(
        self, h: np.ndarray, metadata: dict[str, Any], base_name: str
    ) -> None:
        header_path = self.output_dir / "filter_coefficients.h"
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(
                f"// Auto-generated {self.config.phase_suffix} filter coefficients\n"
            )
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
            f.write("// Filter coefficients stored in external .bin files.\n")
            f.write(f"// Default binary: {base_name}.bin\n\n")
            f.write("#endif // FILTER_COEFFICIENTS_H\n")
        print(f"  保存: {header_path}")

    def _export_metadata(self, metadata: dict[str, Any], base_name: str) -> None:
        metadata_path = self.output_dir / f"{base_name}.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj: Any) -> Any:
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        serializable_metadata = convert_numpy_types(metadata)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
        print(f"  保存: {metadata_path}")


class FilterPlotter:
    """フィルタ特性のプロットを出力する"""

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
        print(f"\nプロット生成中... ({self.output_dir})")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{filter_name}_" if filter_name else ""
        self._plot_impulse_response(h_final, h_linear, prefix)
        self._plot_frequency_response(h_final, h_linear, prefix)
        self._plot_phase_response(h_final, h_linear, prefix)

    def _plot_impulse_response(
        self, h_final: np.ndarray, h_linear: np.ndarray | None, prefix: str
    ) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        display_range = min(4000, len(h_final))
        t = np.arange(display_range)
        axes[0].plot(t, h_final[:display_range], linewidth=0.5, color="orange")
        axes[0].set_title("Impulse Response", fontsize=12)
        axes[0].set_xlabel("Sample")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(0, color="r", linestyle="--", alpha=0.5, label="t=0")
        axes[0].legend()

        if h_linear is not None:
            center = len(h_linear) // 2
            display_range_lin = min(2000, center)
            t_linear = np.arange(-display_range_lin, display_range_lin)
            h_linear_center = h_linear[
                center - display_range_lin : center + display_range_lin
            ]
            axes[1].plot(t_linear, h_linear_center, linewidth=0.5, label="Linear Phase")
            axes[1].set_title("Linear Phase Comparison", fontsize=12)
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

        axes[0].plot(
            w_final / 1000, H_final_db, label="Designed", linewidth=1, alpha=0.7
        )
        axes[0].set_title("Magnitude Response", fontsize=12)
        axes[0].set_xlabel("Frequency (kHz)")
        axes[0].set_ylabel("Magnitude (dB)")
        axes[0].set_ylim(-200, 5)
        axes[0].axhline(
            -180, color="r", linestyle="--", alpha=0.5, label="-180dB Target"
        )
        axes[0].axvline(
            self.config.passband_end / 1000, color="g", linestyle="--", alpha=0.5
        )
        axes[0].axvline(
            self.config.stopband_start / 1000, color="orange", linestyle="--", alpha=0.5
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        passband_mask = w_final <= self.config.passband_end * 1.1
        axes[1].plot(
            w_final[passband_mask] / 1000, H_final_db[passband_mask], color="orange"
        )
        axes[1].set_title("Passband Detail", fontsize=12)
        axes[1].set_xlabel("Frequency (kHz)")
        axes[1].set_ylabel("Magnitude (dB)")
        axes[1].axvline(
            self.config.passband_end / 1000, color="g", linestyle="--", alpha=0.5
        )
        axes[1].grid(True, alpha=0.3)
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
        ax.plot(w / 1000, phase_final, label="Designed", linewidth=1, alpha=0.7)

        if h_linear is not None:
            _, H_lin = signal.freqz(h_linear, worN=8192, fs=self.config.output_rate)
            phase_lin = np.unwrap(np.angle(H_lin))
            ax.plot(w / 1000, phase_lin, label="Linear Phase", linewidth=1, alpha=0.5)

        ax.set_title("Phase Response", fontsize=12)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Phase (rad)")
        ax.axvline(
            self.config.passband_end / 1000, color="g", linestyle="--", alpha=0.5
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}phase_response.png", dpi=150)
        print(f"  保存: {prefix}phase_response.png")
        plt.close()


class FilterValidator:
    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def validate(self, h: np.ndarray) -> dict[str, Any]:
        print("\n仕様検証中...")
        w, H = signal.freqz(h, worN=16384, fs=self.config.output_rate)
        H_db = 20 * np.log10(np.abs(H) + 1e-12)
        passband_mask = w <= self.config.passband_end
        passband_db = H_db[passband_mask]
        passband_ripple = np.max(passband_db) - np.min(passband_db)
        input_band_peak = (
            float(np.max(np.abs(H[passband_mask]))) if np.any(passband_mask) else 0.0
        )
        input_band_peak_normalized = (
            float(input_band_peak / float(self.config.upsample_ratio))
            if self.config.upsample_ratio
            else 0.0
        )

        stopband_mask = w >= self.config.stopband_start
        stopband_attenuation = np.min(H_db[stopband_mask])

        peak_idx = int(np.argmax(np.abs(h)))
        midpoint = len(h) // 2
        energy_ratio = float(
            np.sum(h[:midpoint] ** 2) / (np.sum(h[midpoint:] ** 2) + 1e-12)
        )
        peak_threshold = int(len(h) * 0.01)
        is_peak_at_front = peak_idx < peak_threshold
        is_energy_causal = energy_ratio > 10
        is_symmetric = bool(np.allclose(h, h[::-1], atol=1e-10))

        results = {
            "passband_ripple_db": float(passband_ripple),
            "input_band_peak": input_band_peak,
            "input_band_peak_normalized": input_band_peak_normalized,
            "stopband_attenuation_db": float(abs(stopband_attenuation)),
            "peak_position": peak_idx,
            "peak_threshold_samples": peak_threshold,
            "energy_ratio_first_to_second_half": float(energy_ratio),
            "meets_stopband_spec": abs(stopband_attenuation)
            >= self.config.stopband_attenuation_db,
            "is_minimum_phase": bool(is_peak_at_front and is_energy_causal),
            "is_symmetric": is_symmetric,
            "actual_taps": len(h),
        }
        self._print_results(results, stopband_attenuation)
        return results

    def _print_results(
        self, results: dict[str, Any], stopband_attenuation: float
    ) -> None:
        print(f"  実際のタップ数: {results['actual_taps']}")
        print(f"  通過帯域リップル: {results['passband_ripple_db']:.3f} dB")
        print(
            "  入力帯域ピーク: "
            f"{results.get('input_band_peak_normalized', 0.0):.6f} "
            f"(raw {results.get('input_band_peak', 0.0):.6f})"
        )
        print(
            f"  阻止帯域減衰: {results['stopband_attenuation_db']:.1f} dB (目標: {self.config.stopband_attenuation_db} dB)"
        )
        print(
            f"  阻止帯域スペック: {'合格' if results['meets_stopband_spec'] else '不合格'}"
        )
        print(
            f"  ピーク位置: サンプル {results['peak_position']} "
            f"(先頭1%={results['peak_threshold_samples']}サンプル以内: {'Y' if results['peak_position'] < results['peak_threshold_samples'] else 'N'})"
        )
        print(
            f"  エネルギー比(前半/後半): {results['energy_ratio_first_to_second_half']:.1f}"
        )
        status = "確認" if results["is_minimum_phase"] else "未確認"
        print(f"  最小位相特性: {status}")
        print(f"  対称性: {'〇' if results['is_symmetric'] else '×'}")


def validate_tap_count(taps: int, upsample_ratio: int) -> None:
    if taps % upsample_ratio != 0:
        raise ValueError(
            f"タップ数 {taps:,} はアップサンプリング比率 {upsample_ratio} の倍数である必要があります。"
        )
    print(f"タップ数 {taps:,} は {upsample_ratio} の倍数です")


def compute_padded_taps(n_taps: int, upsample_ratio: int) -> int:
    if n_taps % upsample_ratio == 0:
        return n_taps
    return ((n_taps // upsample_ratio) + 1) * upsample_ratio


def normalize_coefficients(
    h: np.ndarray,
    target_dc_gain: float = 1.0,
    dc_gain_factor: float = 0.99,
) -> tuple[np.ndarray, dict[str, Any]]:
    if h.size == 0:
        raise ValueError("フィルタ係数が空です")
    if target_dc_gain <= 0:
        raise ValueError("DCゲインのターゲットは正の値でなければなりません")
    if not 0 < dc_gain_factor <= 1.0:
        raise ValueError("dc_gain_factorは0より大きく1.0以下でなければなりません")

    dc_gain = float(np.sum(h))
    if abs(dc_gain) < 1e-12:
        raise ValueError("DCゲインが0に近すぎます。係数を確認してください。")

    target_gain = target_dc_gain * dc_gain_factor
    scale = target_gain / dc_gain
    h_normalized = h * scale

    final_dc = float(np.sum(h_normalized))
    max_coef = float(np.max(np.abs(h_normalized)))
    l1_norm = float(np.sum(np.abs(h_normalized)))

    info = {
        "original_dc_gain": dc_gain,
        "target_dc_gain": float(target_dc_gain),
        "dc_gain_factor": dc_gain_factor,
        "normalized_dc_gain": final_dc,
        "applied_scale": float(scale),
        "l1_norm": l1_norm,
        "l1_norm_ratio": l1_norm / target_dc_gain,
        "max_coefficient_amplitude": max_coef,
        "normalization_applied": True,
    }

    print("\n係数正規化:")
    print(
        f"  目標DCゲイン: {target_dc_gain:.6f} × {dc_gain_factor} = {target_gain:.6f}"
    )
    print(f"  元のDCゲイン: {dc_gain:.6f}")
    print(f"  正規化スケール: {scale:.6f}x")
    print(f"  最終DCゲイン: {final_dc:.6f}")
    print(f"  L1ノルム: {l1_norm:.6f} (L1/L = {info['l1_norm_ratio']:.6f})")
    print(f"  最大係数振幅: {max_coef:.6f}")

    return h_normalized, info


def generate_multi_rate_header(
    filter_infos: list[tuple[str, str, int, dict[str, Any]]],
    output_dir: str = "data/coefficients",
) -> None:
    output_path = Path(output_dir)
    header_path = output_path / "filter_coefficients.h"
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated multi-rate filter coefficients\n")
        f.write("// GPU Audio Upsampler - Multi-Rate Support\n")
        f.write(f"// Generated: {datetime.now().isoformat()}\n\n")
        f.write("#ifndef FILTER_COEFFICIENTS_H\n")
        f.write("#define FILTER_COEFFICIENTS_H\n\n")
        f.write("#include <cstddef>\n")
        f.write("#include <cstdint>\n\n")
        f.write("struct FilterConfig {\n")
        f.write("    const char* name;\n")
        f.write("    const char* filename;\n")
        f.write("    size_t taps;\n")
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
    coeff_path = Path(coefficients_dir)
    l1_max = 0.0
    l1_max_filter = ""
    max_coef_max = 0.0
    max_coef_max_filter = ""
    details: list[dict[str, Any]] = []

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
        if l1_norm is None or max_coef is None:
            print(f"  警告: {name} のノルム/最大値が取得できません。")
            continue
        l1_norm = float(l1_norm)
        max_coef = float(max_coef)
        details.append({"name": name, "l1_norm": l1_norm, "max_coef": max_coef})
        if l1_norm > l1_max:
            l1_max = l1_norm
            l1_max_filter = name
        if max_coef > max_coef_max:
            max_coef_max = max_coef
            max_coef_max_filter = name

    recommended_gain = float(safety_margin / max_coef_max) if max_coef_max > 0 else 1.0
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
    print("\n" + "=" * 70)
    print("GLOBAL SAFE GAIN RECOMMENDATION")
    print("=" * 70)
    print(f"L1_max: {safe_gain_info['l1_max']:.2f} ({safe_gain_info['l1_max_filter']})")
    print(
        f"max_coef_max: {safe_gain_info['max_coef_max']:.6f} ({safe_gain_info['max_coef_max_filter']})"
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
