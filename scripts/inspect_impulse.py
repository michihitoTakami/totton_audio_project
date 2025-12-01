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

from scripts.partition_analysis import (
    PartitionConfig,
    build_partition_plan,
    estimate_settling_samples,
    load_partition_config,
    partition_energy_summary,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="低遅延パーティション対応インパルス解析ツール")
    parser.add_argument(
        "--coeff",
        type=Path,
        default=Path("data/coefficients/filter_44k_16x_2m_hybrid_phase.bin"),
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
    parser.add_argument("--window", type=int, default=2000, help="広域プロットのサンプル数")
    parser.add_argument("--zoom", type=int, default=200, help="ズーム表示するサンプル数")
    parser.add_argument("--show", action="store_true", help="matplotlibのGUI表示を有効化")
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


def main():
    args = _parse_args()

    coeff_path = args.coeff
    coeffs = np.fromfile(coeff_path, dtype=np.float32)
    if coeffs.size == 0:
        raise ValueError(f"{coeff_path} から係数を読み込めませんでした。")

    metadata = _load_metadata(args.metadata)
    output_rate = args.output_rate or metadata.get("sample_rate_output")
    upsample_ratio = metadata.get("upsample_ratio")

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

    energy_total = float(np.sum(coeffs.astype(np.float64) ** 2))
    energy_first_100 = float(np.sum(first_100.astype(np.float64) ** 2))
    print(f"先頭100サンプルのエネルギー比率: {energy_first_100 / energy_total * 100:.2f}%")

    if plan.enabled:
        print("\n[Partition Plan]")
        print(f"  設定: fast={config.fast_partition_taps} taps, min={config.min_partition_taps}, "
              f"max={config.max_partitions}, tailFFT×{config.tail_fft_multiple}")
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
            print(
                f"  推定遅延サンプル: fast {fast_window}, tail合流 {settling_window}"
            )
    else:
        print("\nPartitionモードは無効です（--enable-partitionで強制可能）。")

    # プロット生成
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
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

    cumulative = np.cumsum(coeffs.astype(np.float64) ** 2)
    cumulative /= cumulative[-1]
    axes[2].plot(cumulative[:window], linewidth=1.0, color="#2ec4b6")
    axes[2].set_title("Cumulative energy (normalized)")
    axes[2].set_xlabel("Sample")
    axes[2].set_ylabel("Energy ratio")
    axes[2].grid(True, alpha=0.3)
    _shade_partitions(axes[2], partition_summary)

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
            "energy_first_100_pct": energy_first_100 / energy_total * 100.0,
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
