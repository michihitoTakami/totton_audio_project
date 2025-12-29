#!/usr/bin/env python3
"""
De-limiter Loudness Comparison Tool

Generates a visual comparison report showing the effectiveness of de-limiting:
- Waveform comparison (Before/After)
- Loudness metrics (LUFS, True Peak, PLR, Crest Factor)
- Histogram comparison

Usage:
    uv run python scripts/delimiter/compare_loudness.py \
        --before input.wav --after output.wav --output report.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Optional: pyloudnorm for LUFS measurement
try:
    import pyloudnorm as pyln

    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False


@dataclass
class AudioMetrics:
    """Audio loudness and dynamics metrics."""

    path: str
    sample_rate: int
    duration_sec: float
    frames: int
    channels: int

    # Peak metrics
    peak_linear: float
    peak_dbfs: float
    true_peak_dbtp: float  # True Peak (inter-sample)

    # Loudness metrics
    lufs: float | None  # Integrated loudness
    rms_linear: float
    rms_dbfs: float

    # Dynamics metrics
    plr: float | None  # Peak-to-Loudness Ratio (dB)
    crest_factor_db: float  # Peak/RMS ratio in dB

    # Clipping
    clip_count: int
    clip_rate: float


def linear_to_db(linear: float, min_db: float = -200.0) -> float:
    """Convert linear amplitude to dB."""
    if linear <= 0.0:
        return min_db
    return 20.0 * np.log10(linear)


def calculate_true_peak(audio: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Calculate true peak using oversampling."""
    from scipy.signal import resample_poly

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    max_peak = 0.0
    for ch in range(audio.shape[1]):
        # Oversample to detect inter-sample peaks
        oversampled = resample_poly(audio[:, ch], oversample, 1)
        peak = float(np.max(np.abs(oversampled)))
        max_peak = max(max_peak, peak)

    return max_peak


def calculate_lufs(audio: np.ndarray, sr: int) -> float | None:
    """Calculate integrated loudness in LUFS."""
    if not HAS_PYLOUDNORM:
        return None

    try:
        meter = pyln.Meter(sr)
        if audio.ndim == 1:
            # Mono: duplicate to stereo for measurement
            audio = np.column_stack([audio, audio])
        return float(meter.integrated_loudness(audio))
    except Exception:
        return None


def analyze_audio(path: Path) -> AudioMetrics:
    """Analyze audio file and compute metrics."""
    audio, sr = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    frames = audio.shape[0]
    channels = audio.shape[1]
    duration_sec = frames / sr

    # Peak metrics
    peak_linear = float(np.max(np.abs(audio)))
    peak_dbfs = linear_to_db(peak_linear)
    true_peak = calculate_true_peak(audio, sr)
    true_peak_dbtp = linear_to_db(true_peak)

    # RMS
    rms_linear = float(np.sqrt(np.mean(np.square(audio))))
    rms_dbfs = linear_to_db(rms_linear)

    # LUFS
    lufs = calculate_lufs(audio, sr)

    # PLR (Peak-to-Loudness Ratio)
    plr = None
    if lufs is not None and lufs > -70:
        plr = true_peak_dbtp - lufs

    # Crest factor
    crest_factor_db = peak_dbfs - rms_dbfs

    # Clipping
    clip_count = int(np.count_nonzero(np.abs(audio) >= 1.0))
    clip_rate = clip_count / audio.size if audio.size > 0 else 0.0

    return AudioMetrics(
        path=str(path),
        sample_rate=sr,
        duration_sec=duration_sec,
        frames=frames,
        channels=channels,
        peak_linear=peak_linear,
        peak_dbfs=peak_dbfs,
        true_peak_dbtp=true_peak_dbtp,
        lufs=lufs,
        rms_linear=rms_linear,
        rms_dbfs=rms_dbfs,
        plr=plr,
        crest_factor_db=crest_factor_db,
        clip_count=clip_count,
        clip_rate=clip_rate,
    )


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load audio and convert to mono."""
    audio, sr = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio, sr


def create_comparison_report(
    before: AudioMetrics,
    after: AudioMetrics,
    before_audio: np.ndarray,
    after_audio: np.ndarray,
    sr: int,
    output_path: Path,
    title: str = "De-limiter Effect Report",
) -> None:
    """Create visual comparison report."""
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Layout: 3 rows
    # Row 1: Waveforms (before/after)
    # Row 2: Zoomed waveforms (peak region)
    # Row 3: Metrics table + Histogram

    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)

    # Time axis
    time_before = np.arange(len(before_audio)) / sr
    time_after = np.arange(len(after_audio)) / sr

    # Row 1: Full waveforms
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_before, before_audio, linewidth=0.3, color="steelblue")
    ax1.set_title("Before (Input)", fontsize=12)
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1.1, 1.1)
    ax1.axhline(y=1.0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.axhline(y=-1.0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_after, after_audio, linewidth=0.3, color="forestgreen")
    ax2.set_title("After (De-limited)", fontsize=12)
    ax2.set_xlabel("Time (sec)")
    ax2.set_ylabel("Amplitude")
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.axhline(y=-1.0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.grid(True, alpha=0.3)

    # Row 2: Zoomed view (find peak region)
    peak_idx_before = int(np.argmax(np.abs(before_audio)))
    zoom_samples = int(0.05 * sr)  # 50ms window
    zoom_start = max(0, peak_idx_before - zoom_samples // 2)
    zoom_end = min(len(before_audio), zoom_start + zoom_samples)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        time_before[zoom_start:zoom_end],
        before_audio[zoom_start:zoom_end],
        linewidth=1.0,
        color="steelblue",
    )
    ax3.set_title("Before - Peak Region (50ms)", fontsize=12)
    ax3.set_xlabel("Time (sec)")
    ax3.set_ylabel("Amplitude")
    ax3.set_ylim(-1.1, 1.1)
    ax3.axhline(y=1.0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    ax3.axhline(y=-1.0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    ax3.fill_between(
        time_before[zoom_start:zoom_end],
        before_audio[zoom_start:zoom_end],
        alpha=0.3,
        color="steelblue",
    )
    ax3.grid(True, alpha=0.3)

    # Same region for after
    zoom_end_after = min(len(after_audio), zoom_end)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(
        time_after[zoom_start:zoom_end_after],
        after_audio[zoom_start:zoom_end_after],
        linewidth=1.0,
        color="forestgreen",
    )
    ax4.set_title("After - Peak Region (50ms)", fontsize=12)
    ax4.set_xlabel("Time (sec)")
    ax4.set_ylabel("Amplitude")
    ax4.set_ylim(-1.1, 1.1)
    ax4.axhline(y=1.0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    ax4.axhline(y=-1.0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    ax4.fill_between(
        time_after[zoom_start:zoom_end_after],
        after_audio[zoom_start:zoom_end_after],
        alpha=0.3,
        color="forestgreen",
    )
    ax4.grid(True, alpha=0.3)

    # Row 3 Left: Metrics table
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis("off")

    def fmt_val(val: float | None, suffix: str = "", precision: int = 2) -> str:
        if val is None:
            return "N/A"
        return f"{val:.{precision}f}{suffix}"

    def fmt_change(before_val: float | None, after_val: float | None) -> str:
        if before_val is None or after_val is None:
            return ""
        diff = after_val - before_val
        arrow = "+" if diff >= 0 else ""
        return f"({arrow}{diff:.1f})"

    table_data = [
        [
            "Metric",
            "Before",
            "After",
            "Change",
        ],
        [
            "True Peak",
            fmt_val(before.true_peak_dbtp, " dBTP"),
            fmt_val(after.true_peak_dbtp, " dBTP"),
            fmt_change(before.true_peak_dbtp, after.true_peak_dbtp),
        ],
        [
            "LUFS",
            fmt_val(before.lufs, " LUFS"),
            fmt_val(after.lufs, " LUFS"),
            fmt_change(before.lufs, after.lufs),
        ],
        [
            "PLR",
            fmt_val(before.plr, " dB"),
            fmt_val(after.plr, " dB"),
            fmt_change(before.plr, after.plr),
        ],
        [
            "Crest Factor",
            fmt_val(before.crest_factor_db, " dB"),
            fmt_val(after.crest_factor_db, " dB"),
            fmt_change(before.crest_factor_db, after.crest_factor_db),
        ],
        [
            "RMS",
            fmt_val(before.rms_dbfs, " dBFS"),
            fmt_val(after.rms_dbfs, " dBFS"),
            fmt_change(before.rms_dbfs, after.rms_dbfs),
        ],
        [
            "Clip Count",
            str(before.clip_count),
            str(after.clip_count),
            f"({after.clip_count - before.clip_count:+d})",
        ],
    ]

    # Create table
    table = ax5.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.25, 0.25, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight PLR row (key metric)
    for j in range(4):
        table[(3, j)].set_facecolor("#E2EFDA")

    ax5.set_title("Loudness Metrics Comparison", fontsize=12, pad=20)

    # Row 3 Right: Histogram
    ax6 = fig.add_subplot(gs[2, 1])

    bins = np.linspace(0, 1.2, 100)
    ax6.hist(
        np.abs(before_audio),
        bins=bins,
        alpha=0.6,
        label="Before",
        color="steelblue",
        density=True,
    )
    ax6.hist(
        np.abs(after_audio),
        bins=bins,
        alpha=0.6,
        label="After",
        color="forestgreen",
        density=True,
    )
    ax6.axvline(x=1.0, color="red", linestyle="--", linewidth=1, label="Clip threshold")
    ax6.set_xlabel("Absolute Amplitude")
    ax6.set_ylabel("Density")
    ax6.set_title("Amplitude Distribution", fontsize=12)
    ax6.legend(loc="upper right")
    ax6.set_xlim(0, 1.2)
    ax6.grid(True, alpha=0.3)

    # Add interpretation text
    plr_change = (
        (after.plr - before.plr) if before.plr is not None and after.plr is not None else None
    )
    if plr_change is not None:
        if plr_change > 2.0:
            interpretation = "Significant dynamic range restoration detected."
        elif plr_change > 0.5:
            interpretation = "Moderate dynamic range improvement."
        elif plr_change > -0.5:
            interpretation = "Minimal change in dynamics."
        else:
            interpretation = "Warning: Dynamics may have been reduced."
    else:
        interpretation = "PLR could not be calculated (LUFS measurement failed)."

    fig.text(
        0.5,
        0.02,
        f"Interpretation: {interpretation}",
        ha="center",
        fontsize=11,
        style="italic",
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Report saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="De-limiter Loudness Comparison Tool")
    p.add_argument("--before", type=Path, required=True, help="Input (before) WAV file")
    p.add_argument("--after", type=Path, required=True, help="Output (after) WAV file")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("delimiter_comparison.png"),
        help="Output report image path",
    )
    p.add_argument("--title", type=str, default="De-limiter Effect Report", help="Report title")
    p.add_argument("--json", type=Path, default=None, help="Optional JSON metrics output")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.before.exists():
        print(f"Error: Before file not found: {args.before}", file=sys.stderr)
        return 1
    if not args.after.exists():
        print(f"Error: After file not found: {args.after}", file=sys.stderr)
        return 1

    if not HAS_PYLOUDNORM:
        print(
            "Warning: pyloudnorm not installed. LUFS and PLR will not be calculated.",
            file=sys.stderr,
        )
        print("  Install with: uv sync --extra delimiter", file=sys.stderr)

    print(f"Analyzing: {args.before}")
    before_metrics = analyze_audio(args.before)
    before_audio, sr = load_audio_mono(args.before)

    print(f"Analyzing: {args.after}")
    after_metrics = analyze_audio(args.after)
    after_audio, _ = load_audio_mono(args.after)

    # Print summary
    print("\n=== Metrics Summary ===")
    print(f"{'Metric':<20} {'Before':>15} {'After':>15} {'Change':>10}")
    print("-" * 60)
    print(
        f"{'True Peak':<20} {before_metrics.true_peak_dbtp:>12.2f} dBTP "
        f"{after_metrics.true_peak_dbtp:>12.2f} dBTP "
        f"{after_metrics.true_peak_dbtp - before_metrics.true_peak_dbtp:>+8.1f}"
    )
    if before_metrics.lufs is not None and after_metrics.lufs is not None:
        print(
            f"{'LUFS':<20} {before_metrics.lufs:>12.2f} LUFS "
            f"{after_metrics.lufs:>12.2f} LUFS "
            f"{after_metrics.lufs - before_metrics.lufs:>+8.1f}"
        )
    if before_metrics.plr is not None and after_metrics.plr is not None:
        print(
            f"{'PLR':<20} {before_metrics.plr:>14.2f} dB "
            f"{after_metrics.plr:>14.2f} dB "
            f"{after_metrics.plr - before_metrics.plr:>+8.1f}"
        )
    print(
        f"{'Crest Factor':<20} {before_metrics.crest_factor_db:>14.2f} dB "
        f"{after_metrics.crest_factor_db:>14.2f} dB "
        f"{after_metrics.crest_factor_db - before_metrics.crest_factor_db:>+8.1f}"
    )
    print(
        f"{'Clip Count':<20} {before_metrics.clip_count:>15} "
        f"{after_metrics.clip_count:>15} "
        f"{after_metrics.clip_count - before_metrics.clip_count:>+9d}"
    )

    # Generate report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_comparison_report(
        before_metrics,
        after_metrics,
        before_audio,
        after_audio,
        sr,
        args.output,
        title=args.title,
    )

    # Optional JSON output
    if args.json:

        def metrics_to_dict(m: AudioMetrics) -> dict[str, Any]:
            return {
                "path": m.path,
                "sample_rate": m.sample_rate,
                "duration_sec": m.duration_sec,
                "peak_dbfs": m.peak_dbfs,
                "true_peak_dbtp": m.true_peak_dbtp,
                "lufs": m.lufs,
                "rms_dbfs": m.rms_dbfs,
                "plr": m.plr,
                "crest_factor_db": m.crest_factor_db,
                "clip_count": m.clip_count,
            }

        report = {
            "before": metrics_to_dict(before_metrics),
            "after": metrics_to_dict(after_metrics),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON metrics saved to: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
