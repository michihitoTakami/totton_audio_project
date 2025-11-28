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


def _slice_signal(data: np.ndarray, sample_rate: int, start_s: float = 0.0, duration_s: Optional[float] = None):
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


def analyze_frequency_response(
    data_in: np.ndarray,
    sr_in: int,
    data_out: np.ndarray,
    sr_out: int,
    output_plot: Optional[Path] = None,
):
    """Compare spectra of input/output arrays."""
    print(f"Input:  {len(data_in)} samples @ {sr_in} Hz")
    print(f"Output: {len(data_out)} samples @ {sr_out} Hz")
    print(f"Upsample ratio: {sr_out / sr_in:.2f}x")
    print()

    fft_in = _window_fft(data_in)
    fft_out = _window_fft(data_out)

    freqs_in = np.fft.rfftfreq(len(data_in), 1 / sr_in)
    freqs_out = np.fft.rfftfreq(len(data_out), 1 / sr_out)

    mag_in_db = 20 * np.log10(np.abs(fft_in) + 1e-12)
    mag_out_db = 20 * np.log10(np.abs(fft_out) + 1e-12)

    peak_idx_in = int(np.argmax(np.abs(fft_in)))
    peak_idx_out = int(np.argmax(np.abs(fft_out)))
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
        print(f"Stopband (>{stopband_start} Hz) max energy: {max_stopband_energy:.2f} dB")
        print(f"Stopband attenuation: {stopband_delta:.2f} dB")
    print()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(freqs_in / 1000, mag_in_db, linewidth=0.5)
    axes[0].set_xlim(0, sr_in / 2000)
    axes[0].set_ylim(np.max(mag_in_db) - 100, np.max(mag_in_db) + 10)
    axes[0].axvline(peak_freq_in / 1000, color="r", linestyle="--", alpha=0.5, label=f"Peak {peak_freq_in:.1f} Hz")
    axes[0].set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title(f"Input Spectrum ({sr_in} Hz)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(freqs_out / 1000, mag_out_db, linewidth=0.5)
    axes[1].set_xlim(0, min(30, sr_out / 2000))
    axes[1].set_ylim(np.max(mag_out_db) - 100, np.max(mag_out_db) + 10)
    axes[1].axvline(peak_freq_out / 1000, color="r", linestyle="--", alpha=0.5, label=f"Peak {peak_freq_out:.1f} Hz")
    axes[1].axvline(20, color="g", linestyle="--", alpha=0.5, label="20 kHz")
    axes[1].axvline(22.05, color="orange", linestyle="--", alpha=0.5, label="22.05 kHz")
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_title(f"Output Spectrum ({sr_out} Hz)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    destination = output_plot or Path("plots/analysis/frequency_verification.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {destination}")

    return {
        "peak_freq_in": float(peak_freq_in),
        "peak_freq_out": float(peak_freq_out),
        "peak_mag_in": float(peak_mag_in),
        "peak_mag_out": float(peak_mag_out),
        "stopband_delta": stopband_delta,
        "frequency_match": freq_match,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Verify frequency response of upsampled audio")
    parser.add_argument("input", type=Path, help="Input WAV file")
    parser.add_argument("output", type=Path, help="Output WAV file (upsampled)")
    parser.add_argument("--plot", type=Path, default=None, help="出力プロット (png)")
    parser.add_argument("--metadata", type=Path, default=None, help="フィルタ係数メタデータ(JSON)")
    parser.add_argument("--tap-count", type=int, default=None, help="フィルタタップ数を直接指定")
    parser.add_argument("--config", type=Path, default=None, help="partitionedConvolution設定を読み込むconfig.json")
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
    tail_slice = _slice_signal(data_out, sr_out, start_s=settling_seconds, duration_s=window_seconds)
    if fast_slice.size < 2 or tail_slice.size < 2:
        return None
    fast_fft = 20 * np.log10(np.abs(_window_fft(fast_slice)) + 1e-12)
    tail_fft = 20 * np.log10(np.abs(_window_fft(tail_slice)) + 1e-12)
    min_len = min(fast_fft.size, tail_fft.size)
    delta = float(np.max(np.abs(tail_fft[:min_len] - fast_fft[:min_len])))
    rms_fast = float(np.sqrt(np.mean(fast_slice**2)))
    rms_tail = float(np.sqrt(np.mean(tail_slice**2)))
    print(f"Fast vs Tail スペクトル最大差: {delta:.2f} dB (fast RMS {rms_fast:.6f}, tail RMS {rms_tail:.6f})")
    return {
        "spectral_delta_db": delta,
        "rms_fast": rms_fast,
        "rms_tail": rms_tail,
    }


def main():
    args = _parse_args()

    print("=" * 60)
    print("Frequency Response Verification")
    print("=" * 60)
    print()

    sr_in, data_in = _read_wav(args.input)
    sr_out, data_out = _read_wav(args.output)

    metadata = _load_metadata(args.metadata)
    tap_count = args.tap_count or metadata.get("n_taps_actual") or metadata.get("n_taps_specified")

    plan, fast_seconds, settling_seconds = _compute_partition_windows(args, tap_count, sr_out)
    skip_seconds = args.settling_seconds
    if skip_seconds is None and settling_seconds:
        skip_seconds = settling_seconds
    if skip_seconds:
        print(f"Output解析前に {skip_seconds:.3f} 秒スキップ（tail合流後を評価）")

    sliced_output = _slice_signal(data_out, sr_out, start_s=skip_seconds or 0.0, duration_s=args.analysis_window_seconds)
    if args.analysis_window_seconds:
        comparable_duration = args.analysis_window_seconds * (sr_in / sr_out)
        sliced_input = _slice_signal(data_in, sr_in, start_s=0.0, duration_s=comparable_duration)
    else:
        sliced_input = data_in

    results = analyze_frequency_response(sliced_input, sr_in, sliced_output, sr_out, args.plot)

    fast_tail_stats = None
    if args.compare_fast_tail:
        fast_tail_stats = _compare_fast_tail(data_out, sr_out, fast_seconds, settling_seconds, args.analysis_window_seconds)

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
