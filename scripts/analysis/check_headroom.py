#!/usr/bin/env python3
"""
Headroom and clipping diagnostics utilities.

Provides three capabilities:
1. Inspect FIR filter coefficients for theoretical peak overshoot.
2. Generate full-scale PCM signals for stress testing.
3. Analyze captured PCM files (input/output) with histogram + clipping stats.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf

MIN_DBFS = -200.0


def linear_to_db(linear: float) -> float:
    if linear <= 0.0:
        return MIN_DBFS
    return 20.0 * math.log10(linear)


def compute_filter_stats(coeffs: np.ndarray) -> Dict[str, Any]:
    if coeffs.size == 0:
        raise ValueError("Filter coefficient array is empty")
    max_coeff = float(np.max(np.abs(coeffs)))
    headroom_db = max(0.0, linear_to_db(max_coeff))
    return {
        "taps": int(coeffs.size),
        "max_linear": max_coeff,
        "max_dbfs": linear_to_db(max_coeff),
        "required_headroom_db": headroom_db,
    }


def analyze_buffer(buffer: np.ndarray, bins: int = 100) -> Dict[str, Any]:
    if buffer.ndim == 1:
        data = buffer[:, np.newaxis]
    else:
        data = buffer

    if data.size == 0:
        return {
            "channels": 0,
            "total_samples": 0,
            "peak_linear": 0.0,
            "peak_dbfs": MIN_DBFS,
            "clip_count": 0,
            "clip_rate": 0.0,
            "per_channel_peaks": [],
            "histogram": {"bins": [], "counts": []},
        }

    abs_data = np.abs(data)
    peak_linear = float(np.max(abs_data, initial=0.0))
    clip_mask = abs_data >= 1.0
    clip_count = int(np.count_nonzero(clip_mask))
    total_samples = int(abs_data.size)
    clip_rate = (clip_count / total_samples) if total_samples else 0.0

    per_channel_peaks = [
        float(np.max(abs_data[:, idx], initial=0.0)) for idx in range(abs_data.shape[1])
    ]

    hist_counts, hist_edges = np.histogram(abs_data, bins=bins, range=(0.0, 1.2))

    return {
        "channels": int(abs_data.shape[1]),
        "total_samples": total_samples,
        "peak_linear": peak_linear,
        "peak_dbfs": linear_to_db(peak_linear),
        "clip_count": clip_count,
        "clip_rate": clip_rate,
        "per_channel_peaks": per_channel_peaks,
        "histogram": {
            "bins": hist_edges.tolist(),
            "counts": hist_counts.tolist(),
        },
    }


def analyze_wave(path: Path, bins: int = 100) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    audio, sr = sf.read(path)
    audio = audio.astype(np.float32)
    stats = analyze_buffer(audio, bins=bins)
    stats["sample_rate"] = int(sr)
    stats["duration_sec"] = float(len(audio) / sr if sr else 0.0)
    stats["path"] = str(path)
    return stats


def generate_fullscale_sine(
    output_path: Path,
    sample_rate: int,
    duration_sec: float,
    frequency_hz: float,
    amplitude: float,
) -> Path:
    if amplitude <= 0.0 or amplitude > 1.0:
        raise ValueError("Amplitude must be within (0, 1]")
    total_samples = int(sample_rate * duration_sec)
    t = np.arange(total_samples, dtype=np.float64) / sample_rate
    wave = (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)
    sf.write(output_path, wave, sample_rate)
    return output_path


def load_filter(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.fromfile(path, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clip/headroom diagnostics toolkit")
    parser.add_argument("--filter", type=Path, help="Path to FIR coefficient binary")
    parser.add_argument(
        "--generate-sine",
        type=Path,
        help="Optional output path to render a full-scale sine for stress testing",
    )
    parser.add_argument(
        "--sine-frequency", type=float, default=997.0, help="Sine frequency in Hz"
    )
    parser.add_argument(
        "--sine-duration",
        type=float,
        default=5.0,
        help="Duration in seconds for generated sine",
    )
    parser.add_argument(
        "--sine-sample-rate",
        type=int,
        default=44100,
        help="Sample rate for generated sine",
    )
    parser.add_argument(
        "--sine-amplitude",
        type=float,
        default=0.999,
        help="Sine amplitude (set to <1 to avoid immediate clipping)",
    )
    parser.add_argument(
        "--input-wav", type=Path, help="Optional captured input WAV to inspect"
    )
    parser.add_argument(
        "--output-wav", type=Path, help="Optional captured output WAV to inspect"
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=120,
        help="Number of histogram bins for PCM analysis",
    )
    parser.add_argument(
        "--fail-on-clip",
        action="store_true",
        help="Exit with status 1 if analyzed output contains clipped samples",
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=None,
        help="Exit with status 1 if analyzed output peak exceeds this linear value",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report destination")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: Dict[str, Any] = {}

    if args.filter:
        coeffs = load_filter(args.filter)
        stats = compute_filter_stats(coeffs)
        results["filter"] = stats
        print(
            f"[filter] taps={stats['taps']}  max_linear={stats['max_linear']:.6f} "
            f"max_dbfs={stats['max_dbfs']:.3f} dB  "
            f"required_headroom={stats['required_headroom_db']:.3f} dB"
        )

    if args.generate_sine:
        out_path = generate_fullscale_sine(
            args.generate_sine,
            sample_rate=args.sine_sample_rate,
            duration_sec=args.sine_duration,
            frequency_hz=args.sine_frequency,
            amplitude=args.sine_amplitude,
        )
        print(
            f"[generate] wrote {out_path} ({args.sine_sample_rate} Hz, {args.sine_duration}s)"
        )

    if args.input_wav:
        input_stats = analyze_wave(args.input_wav, bins=args.hist_bins)
        results["input"] = input_stats
        print(
            f"[input] {args.input_wav} peak={input_stats['peak_linear']:.6f} "
            f"({input_stats['peak_dbfs']:.2f} dBFS)  "
            f"clip_count={input_stats['clip_count']}"
        )

    if args.output_wav:
        output_stats = analyze_wave(args.output_wav, bins=args.hist_bins)
        results["output"] = output_stats
        print(
            f"[output] {args.output_wav} peak={output_stats['peak_linear']:.6f} "
            f"({output_stats['peak_dbfs']:.2f} dBFS)  "
            f"clip_count={output_stats['clip_count']}"
        )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[report] wrote {args.report}")

    exit_code = 0
    output_stats = results.get("output")
    if isinstance(output_stats, dict):
        if args.fail_on_clip and output_stats.get("clip_count", 0) > 0:
            print("[fail] clipped samples detected in output analysis")
            exit_code = 1
        if args.fail_threshold is not None:
            peak = output_stats.get("peak_linear", 0.0)
            if peak >= args.fail_threshold:
                print(
                    f"[fail] output peak {peak:.6f} exceeds threshold "
                    f"{args.fail_threshold:.6f}"
                )
                exit_code = 1

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
