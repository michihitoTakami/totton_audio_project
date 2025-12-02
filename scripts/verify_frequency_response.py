#!/usr/bin/env python3
"""
Verify frequency response of upsampled audio.
低遅延パーティションモード用に、出力信号の先頭区間をスキップして解析できるよう拡張。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile

from scripts.partition_analysis import (
    PartitionConfig,
    build_partition_plan,
    estimate_settling_samples,
    load_partition_config,
)


def _read_wav(path: Path) -> Tuple[int, np.ndarray]:
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]
    return sr, data.astype(np.float32)


def _slice_signal(
    data: np.ndarray,
    sample_rate: int,
    start_s: float = 0.0,
    duration_s: Optional[float] = None,
):
    start_idx = max(0, int(start_s * sample_rate))
    if duration_s is None:
        end_idx = data.shape[0]
    else:
        end_idx = min(data.shape[0], start_idx + max(1, int(duration_s * sample_rate)))
    if end_idx <= start_idx:
        end_idx = min(data.shape[0], start_idx + 1)
    return data[start_idx:end_idx]


def _window_fft(signal: np.ndarray):
    if signal.size < 2:
        raise ValueError("FFT対象のサンプル数が不足しています")
    window = np.hanning(signal.size)
    return np.fft.rfft(signal * window)


def _compute_spectrum(signal: np.ndarray, sample_rate: int):
    fft = _window_fft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    mag_db = 20 * np.log10(np.abs(fft) + 1e-12)
    return freqs, mag_db


def analyze_frequency_response(
    data_in: np.ndarray,
    sr_in: int,
    data_out: np.ndarray,
    sr_out: int,
):
    """Compare spectra of input/output arrays and return metrics with plot data."""
    print(f"Input:  {len(data_in)} samples @ {sr_in} Hz")
    print(f"Output: {len(data_out)} samples @ {sr_out} Hz")
    print(f"Upsample ratio: {sr_out / sr_in:.2f}x")
    print()

    freqs_in, mag_in_db = _compute_spectrum(data_in, sr_in)
    freqs_out, mag_out_db = _compute_spectrum(data_out, sr_out)

    peak_idx_in = int(np.argmax(mag_in_db))
    peak_idx_out = int(np.argmax(mag_out_db))
    peak_freq_in = freqs_in[peak_idx_in]
    peak_freq_out = freqs_out[peak_idx_out]
    peak_mag_in = mag_in_db[peak_idx_in]
    peak_mag_out = mag_out_db[peak_idx_out]

    print(f"Input peak:  {peak_freq_in:.2f} Hz @ {peak_mag_in:.2f} dB")
    print(f"Output peak: {peak_freq_out:.2f} Hz @ {peak_mag_out:.2f} dB")
    freq_match = abs(peak_freq_in - peak_freq_out) < 10
    print(f"Peak frequency match (<10 Hz): {freq_match}")
    print()

    stopband_start = 22050
    stopband_mask = freqs_out > stopband_start
    stopband_delta = None
    if np.any(stopband_mask):
        stopband_energy = mag_out_db[stopband_mask]
        max_stopband_energy = float(np.max(stopband_energy))
        stopband_delta = peak_mag_out - max_stopband_energy
        print(
            f"Stopband (>{stopband_start} Hz) max energy: {max_stopband_energy:.2f} dB"
        )
        print(f"Stopband attenuation: {stopband_delta:.2f} dB")
    print()

    plot_data = {
        "freqs_in": freqs_in,
        "mag_in_db": mag_in_db,
        "freqs_out": freqs_out,
        "mag_out_db": mag_out_db,
        "peak_freq_in": peak_freq_in,
        "peak_freq_out": peak_freq_out,
    }

    return (
        {
            "peak_freq_in": float(peak_freq_in),
            "peak_freq_out": float(peak_freq_out),
            "peak_mag_in": float(peak_mag_in),
            "peak_mag_out": float(peak_mag_out),
            "stopband_delta": stopband_delta,
            "frequency_match": freq_match,
        },
        plot_data,
    )


def _render_frequency_plot(
    plot_data: dict,
    output_plot: Optional[Path],
    reference_plot: Optional[dict],
    reference_label: str,
    delta_plot: Optional[dict],
):
    plot_rows = 2 + (1 if delta_plot else 0)
    fig, axes = plt.subplots(plot_rows, 1, figsize=(12, 8 + 2.5 * (plot_rows - 2)))
    axes = np.atleast_1d(axes)

    freqs_in = plot_data["freqs_in"]
    mag_in_db = plot_data["mag_in_db"]
    axes[0].plot(freqs_in / 1000, mag_in_db, linewidth=0.5, label="Input")
    axes[0].set_xlim(0, freqs_in[-1] / 1000)
    axes[0].set_ylim(np.max(mag_in_db) - 100, np.max(mag_in_db) + 10)
    axes[0].axvline(
        plot_data["peak_freq_in"] / 1000,
        color="r",
        linestyle="--",
        alpha=0.5,
        label="Peak",
    )
    axes[0].set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title(f"Input Spectrum ({int(freqs_in[-1] * 2)} Hz Nyquist)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    freqs_out = plot_data["freqs_out"]
    mag_out_db = plot_data["mag_out_db"]
    axes[1].plot(freqs_out / 1000, mag_out_db, linewidth=0.5, label="Hybrid output")
    axes[1].set_xlim(0, min(30, freqs_out[-1] / 1000))
    axes[1].set_ylim(np.max(mag_out_db) - 100, np.max(mag_out_db) + 10)
    axes[1].axvline(
        plot_data["peak_freq_out"] / 1000,
        color="r",
        linestyle="--",
        alpha=0.5,
        label="Peak",
    )
    axes[1].axvline(20, color="g", linestyle="--", alpha=0.4, label="20 kHz")
    axes[1].axvline(22.05, color="orange", linestyle="--", alpha=0.4, label="22.05 kHz")
    if reference_plot:
        axes[1].plot(
            reference_plot["freqs"] / 1000,
            reference_plot["mag_db"],
            linewidth=0.5,
            alpha=0.8,
            label=reference_label,
        )
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_title("Output Spectrum")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if delta_plot:
        ax_delta = axes[2]
        ax_delta.plot(
            delta_plot["frequencies"] / 1000,
            delta_plot["delta_db"],
            linewidth=0.6,
            color="#8338ec",
        )
        ax_delta.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_delta.set_xlabel("Frequency (kHz)")
        ax_delta.set_ylabel("Δ Hybrid - Reference (dB)")
        ax_delta.set_title("Hybrid vs Reference magnitude delta")
        ax_delta.grid(True, alpha=0.3)

    plt.tight_layout()
    destination = output_plot or Path("plots/analysis/frequency_verification.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {destination}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Verify frequency response of upsampled audio"
    )
    parser.add_argument("input", type=Path, help="Input WAV file")
    parser.add_argument("output", type=Path, help="Output WAV file (upsampled)")
    parser.add_argument("--plot", type=Path, default=None, help="出力プロット (png)")
    parser.add_argument(
        "--metadata", type=Path, default=None, help="フィルタ係数メタデータ(JSON)"
    )
    parser.add_argument(
        "--tap-count", type=int, default=None, help="フィルタタップ数を直接指定"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="partitionedConvolution設定を読み込むconfig.json",
    )
    parser.add_argument("--fast-partition-taps", type=int, default=None)
    parser.add_argument("--min-partition-taps", type=int, default=None)
    parser.add_argument("--max-partitions", type=int, default=None)
    parser.add_argument("--tail-fft-multiple", type=int, default=None)
    parser.add_argument(
        "--settling-seconds",
        type=float,
        default=None,
        help="出力解析前にスキップする秒数（未指定時はpartition計画から自動算出）",
    )
    parser.add_argument(
        "--analysis-window-seconds",
        type=float,
        default=None,
        help="この秒数だけをFFT対象にする（未指定時は全体）",
    )
    parser.add_argument(
        "--compare-fast-tail",
        action="store_true",
        help="fastパーティションのみのスペクトルとtail合成後を比較して差分を表示",
    )
    parser.add_argument(
        "--partition-enabled",
        action="store_true",
        help="config.jsonが無い場合でもpartition計画を強制有効化",
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        default=None,
        help="比較対象となる旧最小位相などのWAVを指定するとハイブリッドとの差分を算出",
    )
    parser.add_argument(
        "--reference-label",
        type=str,
        default="Legacy min-phase",
        help="比較対象のラベル文字列",
    )
    return parser.parse_args()


def _load_metadata(path: Optional[Path]) -> dict:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _compute_partition_windows(args, tap_count: Optional[int], output_rate: int):
    if tap_count is None:
        return None, None, None
    config = load_partition_config(args.config, base=PartitionConfig())
    if args.partition_enabled:
        config = config.merge_overrides(enabled=True)
    config = config.merge_overrides(
        fast_partition_taps=args.fast_partition_taps,
        min_partition_taps=args.min_partition_taps,
        max_partitions=args.max_partitions,
        tail_fft_multiple=args.tail_fft_multiple,
    )
    plan = build_partition_plan(tap_count, config)
    if not plan.enabled:
        return plan, None, None
    fast_window, settling_window = estimate_settling_samples(plan)
    if output_rate <= 0:
        return plan, fast_window, settling_window
    fast_seconds = fast_window / output_rate
    settling_seconds = settling_window / output_rate
    return plan, fast_seconds, settling_seconds


def _compare_fast_tail(
    data_out: np.ndarray,
    sr_out: int,
    fast_seconds: Optional[float],
    settling_seconds: Optional[float],
    window_seconds: Optional[float],
):
    if fast_seconds is None or settling_seconds is None:
        return None
    fast_duration = window_seconds or max(fast_seconds, 0.05)
    fast_slice = _slice_signal(data_out, sr_out, start_s=0.0, duration_s=fast_duration)
    tail_slice = _slice_signal(
        data_out, sr_out, start_s=settling_seconds, duration_s=window_seconds
    )
    if fast_slice.size < 2 or tail_slice.size < 2:
        return None
    fast_fft = 20 * np.log10(np.abs(_window_fft(fast_slice)) + 1e-12)
    tail_fft = 20 * np.log10(np.abs(_window_fft(tail_slice)) + 1e-12)
    min_len = min(fast_fft.size, tail_fft.size)
    delta = float(np.max(np.abs(tail_fft[:min_len] - fast_fft[:min_len])))
    rms_fast = float(np.sqrt(np.mean(fast_slice**2)))
    rms_tail = float(np.sqrt(np.mean(tail_slice**2)))
    print(
        f"Fast vs Tail スペクトル最大差: {delta:.2f} dB (fast RMS {rms_fast:.6f}, tail RMS {rms_tail:.6f})"
    )
    return {
        "spectral_delta_db": delta,
        "rms_fast": rms_fast,
        "rms_tail": rms_tail,
    }


def _compare_reference_output(
    data_ref: np.ndarray,
    sr_ref: int,
    slice_start: float,
    duration_s: Optional[float],
    freqs_target: np.ndarray,
    mag_target_db: np.ndarray,
    sr_target: int,
):
    slice_ref = _slice_signal(
        data_ref, sr_ref, start_s=slice_start, duration_s=duration_s
    )
    if slice_ref.size < 2:
        print("Reference出力が短すぎるため比較をスキップします。")
        return None, None
    freqs_ref, mag_ref_db = _compute_spectrum(slice_ref, sr_ref)
    reference_plot = {"freqs": freqs_ref, "mag_db": mag_ref_db}

    if sr_ref != sr_target:
        print(
            f"Referenceサンプルレート {sr_ref} Hz と出力 {sr_target} Hz が一致しません。"
            " スペクトル差分は表示せず、プロットのみ重ねます。"
        )
        return reference_plot, None

    max_freq = min(freqs_ref[-1], freqs_target[-1])
    mask_target = freqs_target <= max_freq
    interp_ref = np.interp(freqs_target[mask_target], freqs_ref, mag_ref_db)
    delta = mag_target_db[mask_target] - interp_ref
    if delta.size == 0:
        return reference_plot, None

    delta_stats = {
        "frequencies": freqs_target[mask_target],
        "delta_db": delta,
        "max_abs_db": float(np.max(np.abs(delta))),
        "rms_db": float(np.sqrt(np.mean(delta**2))),
    }
    return reference_plot, delta_stats


def main():
    args = _parse_args()

    print("=" * 60)
    print("Frequency Response Verification")
    print("=" * 60)
    print()

    sr_in, data_in = _read_wav(args.input)
    sr_out, data_out = _read_wav(args.output)

    metadata = _load_metadata(args.metadata)
    tap_count = (
        args.tap_count
        or metadata.get("n_taps_actual")
        or metadata.get("n_taps_specified")
    )

    plan, fast_seconds, settling_seconds = _compute_partition_windows(
        args, tap_count, sr_out
    )
    skip_seconds = args.settling_seconds
    if skip_seconds is None and settling_seconds:
        skip_seconds = settling_seconds
    if skip_seconds:
        print(f"Output解析前に {skip_seconds:.3f} 秒スキップ（tail合流後を評価）")

    sliced_output = _slice_signal(
        data_out,
        sr_out,
        start_s=skip_seconds or 0.0,
        duration_s=args.analysis_window_seconds,
    )
    if args.analysis_window_seconds:
        comparable_duration = args.analysis_window_seconds * (sr_in / sr_out)
        sliced_input = _slice_signal(
            data_in, sr_in, start_s=0.0, duration_s=comparable_duration
        )
    else:
        sliced_input = data_in

    results, plot_data = analyze_frequency_response(
        sliced_input, sr_in, sliced_output, sr_out
    )

    reference_plot = None
    reference_delta = None
    if args.reference_output:
        sr_ref, data_ref = _read_wav(args.reference_output)
        reference_plot, reference_delta = _compare_reference_output(
            data_ref,
            sr_ref,
            slice_start=skip_seconds or 0.0,
            duration_s=args.analysis_window_seconds,
            freqs_target=plot_data["freqs_out"],
            mag_target_db=plot_data["mag_out_db"],
            sr_target=sr_out,
        )
        if reference_delta:
            print(
                f"Hybrid vs {args.reference_label}: "
                f"max Δ {reference_delta['max_abs_db']:.2f} dB / "
                f"RMS Δ {reference_delta['rms_db']:.2f} dB"
            )
            results["reference_delta_max_db"] = reference_delta["max_abs_db"]
            results["reference_delta_rms_db"] = reference_delta["rms_db"]

    fast_tail_stats = None
    if args.compare_fast_tail:
        fast_tail_stats = _compare_fast_tail(
            data_out,
            sr_out,
            fast_seconds,
            settling_seconds,
            args.analysis_window_seconds,
        )

    _render_frequency_plot(
        plot_data, args.plot, reference_plot, args.reference_label, reference_delta
    )

    print("=" * 60)
    if results["frequency_match"]:
        print("✓ Frequency response verification PASSED")
    else:
        print("✗ Frequency response verification FAILED")
    print("=" * 60)

    if plan and plan.enabled:
        print(f"Partition plan: {plan.describe()}")

    if fast_tail_stats:
        print(f"Fast/Tail サマリ: Δmax={fast_tail_stats['spectral_delta_db']:.2f} dB")


if __name__ == "__main__":
    main()
