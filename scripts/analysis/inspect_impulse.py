#!/usr/bin/env python3
"""
最小位相フィルタのインパルス応答を可視化し、低遅延パーティション構成を評価するツール。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.partition_analysis import (
    PartitionConfig,
    build_partition_plan,
    estimate_settling_samples,
    load_partition_config,
    partition_energy_summary,
)

HYBRID_CROSSOVER_HZ = 100.0
HYBRID_CROSSOVER_LABEL = f"{HYBRID_CROSSOVER_HZ:.0f} Hz"
GROUP_DELAY_SUMMARY_KEY = f"group_delay_{int(HYBRID_CROSSOVER_HZ)}hz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="低遅延パーティション対応インパルス解析ツール"
    )
    parser.add_argument(
        "--coeff",
        type=Path,
        default=Path("data/coefficients/filter_44k_16x_640k_linear_phase.bin"),
        help="解析対象のFIR係数(.bin)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="係数メタデータ(.json)。出力レートや倍率計算に使用",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="partitionedConvolution設定を引き込むconfig.json",
    )
    parser.add_argument("--fast-partition-taps", type=int, default=None)
    parser.add_argument("--min-partition-taps", type=int, default=None)
    parser.add_argument("--max-partitions", type=int, default=None)
    parser.add_argument("--tail-fft-multiple", type=int, default=None)
    parser.add_argument(
        "--enable-partition",
        dest="partition_enabled",
        action="store_true",
        help="強制的にpartitionモードを有効化",
    )
    parser.add_argument(
        "--disable-partition",
        dest="partition_enabled",
        action="store_false",
        help="強制的にpartitionモードを無効化",
    )
    parser.set_defaults(partition_enabled=None)
    parser.add_argument(
        "--output-rate",
        type=int,
        default=None,
        help="出力サンプルレートを手動指定（metadata未指定時に使用）",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("plots/analysis/impulse_detail.png"),
        help="生成するPNGファイル",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="解析結果をJSONで保存するパス",
    )
    parser.add_argument(
        "--window", type=int, default=2000, help="広域プロットのサンプル数"
    )
    parser.add_argument(
        "--zoom", type=int, default=200, help="ズーム表示するサンプル数"
    )
    parser.add_argument(
        "--show", action="store_true", help="matplotlibのGUI表示を有効化"
    )
    return parser.parse_args()


def _load_metadata(meta_path: Optional[Path]) -> Dict[str, Any]:
    if not meta_path:
        return {}
    with meta_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _shade_partitions(ax, summary):
    if not summary:
        return
    cursor = 0
    for entry in summary:
        end = cursor + entry["taps"]
        color = "#ffbe0b" if entry["role"] == "fast" else "#3a86ff"
        ax.axvspan(cursor, end, color=color, alpha=0.08, linewidth=0)
        cursor = end


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _fast_partition_energy_pct(cumulative_energy: np.ndarray, plan) -> Optional[float]:
    if not plan.enabled or plan.realtime_taps <= 0 or cumulative_energy.size == 0:
        return None
    idx = min(plan.realtime_taps - 1, cumulative_energy.size - 1)
    return float(cumulative_energy[idx] * 100.0)


def _pre_ringing_ratio_pct(coeffs_sq_cumsum: np.ndarray, peak_idx: int) -> float:
    if coeffs_sq_cumsum.size == 0:
        return 0.0
    if peak_idx <= 0:
        return 0.0
    peak_idx = min(peak_idx, coeffs_sq_cumsum.size - 1)
    total_energy = float(coeffs_sq_cumsum[-1])
    if total_energy == 0:
        return 0.0
    pre_energy = float(coeffs_sq_cumsum[peak_idx - 1])
    return (pre_energy / total_energy) * 100.0


def _compute_group_delay_curve(coeffs: np.ndarray, sample_rate: Optional[int]):
    if not sample_rate or sample_rate <= 0:
        return None, None
    base_len = max(len(coeffs), 1024)
    fft_len = 1 << int(np.ceil(np.log2(base_len)))
    fft_len = min(max(fft_len * 4, 4096), 4_194_304)
    response = np.fft.rfft(coeffs, n=fft_len)
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / sample_rate)
    omega = 2 * np.pi * freqs
    phase = np.unwrap(np.angle(response))
    with np.errstate(divide="ignore", invalid="ignore"):
        group_delay = -np.gradient(phase, omega, edge_order=2)
    group_delay[0] = np.nan
    mag = np.abs(response)
    threshold = np.max(mag) * 1e-6
    group_delay[mag < threshold] = np.nan
    return freqs, group_delay * 1000.0  # milliseconds


def _group_delay_metrics(
    freqs,
    group_delay_ms,
    center_hz: float = HYBRID_CROSSOVER_HZ,
    window_hz: float = 40.0,
):
    if freqs is None or group_delay_ms is None:
        return None
    half_window = window_hz / 2.0
    mask = (
        (freqs >= center_hz - half_window)
        & (freqs <= center_hz + half_window)
        & ~np.isnan(group_delay_ms)
    )
    if not np.any(mask):
        return None
    segment = group_delay_ms[mask]
    return {
        "center_hz": center_hz,
        "window_hz": window_hz,
        "mean_ms": float(np.nanmean(segment)),
        "span_ms": float(np.nanmax(segment) - np.nanmin(segment)),
        "min_ms": float(np.nanmin(segment)),
        "max_ms": float(np.nanmax(segment)),
    }


def main():
    args = _parse_args()

    coeff_path = args.coeff
    coeffs = np.fromfile(coeff_path, dtype=np.float32)
    if coeffs.size == 0:
        raise ValueError(f"{coeff_path} から係数を読み込めませんでした。")
    coeffs64 = coeffs.astype(np.float64, copy=False)
    energy_cumsum = np.cumsum(coeffs64**2)
    total_energy = float(energy_cumsum[-1])
    if total_energy == 0.0:
        raise ValueError("係数エネルギーが0です。フィルタ内容を確認してください。")

    metadata = _load_metadata(args.metadata)
    output_rate = args.output_rate or metadata.get("sample_rate_output")

    config = load_partition_config(args.config, base=PartitionConfig())
    if args.partition_enabled is not None:
        config = config.merge_overrides(enabled=args.partition_enabled)
    config = config.merge_overrides(
        fast_partition_taps=args.fast_partition_taps,
        min_partition_taps=args.min_partition_taps,
        max_partitions=args.max_partitions,
        tail_fft_multiple=args.tail_fft_multiple,
    )

    plan = build_partition_plan(len(coeffs), config)
    partition_summary = partition_energy_summary(coeffs, plan)
    fast_window, settling_window = estimate_settling_samples(plan)

    print("=" * 60)
    print(f"フィルタ係数ファイル: {coeff_path}")
    if metadata:
        print(
            f"入力レート {metadata.get('sample_rate_input')} Hz → 出力レート {metadata.get('sample_rate_output')} Hz "
            f"(×{metadata.get('upsample_ratio')})"
        )
    print(f"タップ数: {len(coeffs)}")

    first_100 = coeffs[:100]
    peak_idx = int(np.argmax(np.abs(coeffs[: min(1000, coeffs.size)])))
    peak_value = float(coeffs[peak_idx])
    print(f"ピーク位置: サンプル {peak_idx} (値 {peak_value:.6f})")
    print("先頭10サンプル:")
    for i in range(min(10, coeffs.size)):
        print(f"  h[{i:02d}] = {coeffs[i]:.8f}")

    energy_first_100 = float(np.sum(first_100.astype(np.float64) ** 2))
    print(
        f"先頭100サンプルのエネルギー比率: {energy_first_100 / total_energy * 100:.2f}%"
    )

    cumulative_normalized = energy_cumsum / total_energy
    pre_ringing_pct = _pre_ringing_ratio_pct(energy_cumsum, peak_idx)
    fast_partition_energy_pct = _fast_partition_energy_pct(cumulative_normalized, plan)
    group_delay_freqs, group_delay_ms = _compute_group_delay_curve(coeffs, output_rate)
    group_delay_summary = _group_delay_metrics(
        group_delay_freqs, group_delay_ms, center_hz=HYBRID_CROSSOVER_HZ
    )

    if plan.enabled:
        print("\n[Partition Plan]")
        print(
            f"  設定: fast={config.fast_partition_taps} taps, min={config.min_partition_taps}, "
            f"max={config.max_partitions}, tailFFT×{config.tail_fft_multiple}"
        )
        for entry in partition_summary:
            print(
                f"  - {entry['role']}#{entry['index']}: {entry['taps']} taps, FFT {entry['fft_size']}, "
                f"valid {entry['valid_output']} → energy {entry['energy_pct']:.3f}%"
            )
        if output_rate:
            fast_ms = fast_window / output_rate * 1000.0
            settling_ms = settling_window / output_rate * 1000.0
            print(
                f"  推定遅延: fast {fast_window} samples ≈ {fast_ms:.2f} ms / "
                f"全パーティション {settling_window} samples ≈ {settling_ms:.2f} ms"
            )
        else:
            print(f"  推定遅延サンプル: fast {fast_window}, tail合流 {settling_window}")
    else:
        print("\nPartitionモードは無効です（--enable-partitionで強制可能）。")

    if fast_partition_energy_pct is not None:
        print(f"Fastパーティション累積エネルギー: {fast_partition_energy_pct:.2f}%")
    print(f"プリリンギングエネルギー比（ピーク前）: {pre_ringing_pct:.4f}%")
    if group_delay_summary:
        window_half = group_delay_summary["window_hz"] / 2.0
        print(
            f"{HYBRID_CROSSOVER_LABEL}帯域の群遅延: "
            f"平均 {group_delay_summary['mean_ms']:.3f} ms / "
            f"変動幅 {group_delay_summary['span_ms']:.3f} ms "
            f"(±{window_half:.0f} Hz)"
        )

    # プロット生成
    plot_rows = 4 if group_delay_freqs is not None and group_delay_ms is not None else 3
    fig, axes = plt.subplots(plot_rows, 1, figsize=(13, 11 + 2.5 * (plot_rows - 3)))
    axes = np.atleast_1d(axes)
    window = min(args.window, coeffs.size)
    zoom = min(args.zoom, coeffs.size)

    axes[0].plot(coeffs[:window], linewidth=0.8)
    axes[0].set_title(f"Impulse Response (first {window} samples)")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(peak_idx, color="r", linestyle="--", alpha=0.6, label="Peak")
    axes[0].legend()
    _shade_partitions(axes[0], partition_summary)

    axes[1].plot(coeffs[:zoom], linewidth=1.2, marker="o", markersize=3)
    axes[1].set_title(f"Impulse Response (first {zoom} samples zoom)")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(peak_idx, color="r", linestyle="--", alpha=0.6)
    _shade_partitions(axes[1], partition_summary)

    axes[2].plot(cumulative_normalized[:window], linewidth=1.0, color="#2ec4b6")
    axes[2].set_title("Cumulative energy (normalized)")
    axes[2].set_xlabel("Sample")
    axes[2].set_ylabel("Energy ratio")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(
        pre_ringing_pct / 100.0,
        color="#ff006e",
        linestyle=":",
        alpha=0.6,
        label="Pre-ringing ratio",
    )
    if plan.enabled and plan.realtime_taps > 0:
        axes[2].axvline(
            plan.realtime_taps,
            color="#8338ec",
            linestyle="--",
            alpha=0.7,
            label="Fast boundary",
        )
    _shade_partitions(axes[2], partition_summary)
    axes[2].legend()

    if plot_rows == 4:
        ax_gd = axes[3]
        ax_gd.plot(group_delay_freqs, group_delay_ms, linewidth=0.8, color="#1d8a99")
        ax_gd.set_xlim(
            0, min(500, group_delay_freqs[-1] if group_delay_freqs.size else 500)
        )
        ax_gd.set_xlabel("Frequency (Hz)")
        ax_gd.set_ylabel("Group delay (ms)")
        ax_gd.set_title("Group delay curve")
        ax_gd.axvline(
            HYBRID_CROSSOVER_HZ,
            color="#ffbe0b",
            linestyle="--",
            alpha=0.7,
            label=f"{HYBRID_CROSSOVER_LABEL} focus",
        )
        if group_delay_summary:
            ax_gd.axhspan(
                group_delay_summary["min_ms"],
                group_delay_summary["max_ms"],
                color="#e0fbfc",
                alpha=0.3,
                label=f"{HYBRID_CROSSOVER_LABEL} window range",
            )
        ax_gd.grid(True, alpha=0.3)
        ax_gd.legend()

    plt.tight_layout()
    _ensure_parent(args.output_plot)
    plt.savefig(args.output_plot, dpi=150)
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    print(f"\nプロット保存: {args.output_plot}")

    if args.summary_json:
        summary_payload: Dict[str, Any] = {
            "coeff_path": str(coeff_path),
            "tap_count": int(coeffs.size),
            "peak_index": peak_idx,
            "peak_value": peak_value,
            "energy_first_100_pct": energy_first_100 / total_energy * 100.0,
            "fast_partition_energy_pct": fast_partition_energy_pct,
            "pre_ringing_pct": pre_ringing_pct,
            GROUP_DELAY_SUMMARY_KEY: group_delay_summary,
            "metadata": metadata,
            "partition_plan": {
                "enabled": plan.enabled,
                "total_taps": plan.total_taps,
                "realtime_taps": plan.realtime_taps,
                "fast_window_samples": fast_window,
                "settling_window_samples": settling_window,
                "summary": partition_summary,
            },
        }
        _ensure_parent(args.summary_json)
        with args.summary_json.open("w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, indent=2, ensure_ascii=False)
        print(f"サマリJSONを保存しました: {args.summary_json}")


if __name__ == "__main__":
    main()
