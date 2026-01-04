#!/usr/bin/env python3
"""
De-limiter Multi-Scale Waveform Comparison

Generates waveform comparison at multiple zoom levels:
- 10 seconds
- 1 second
- 0.1 seconds (100ms)

Shows the "nori" (brick wall) restoration clearly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load audio and convert to mono."""
    audio, sr = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio, sr


def find_loud_region(audio: np.ndarray, sr: int, window_sec: float = 1.0) -> int:
    """Find the loudest region in the audio."""
    window_samples = int(window_sec * sr)
    if len(audio) <= window_samples:
        return 0

    # Calculate RMS in sliding windows
    best_idx = 0
    best_rms = 0.0

    step = window_samples // 4
    for i in range(0, len(audio) - window_samples, step):
        rms = np.sqrt(np.mean(np.square(audio[i : i + window_samples])))
        if rms > best_rms:
            best_rms = rms
            best_idx = i

    return best_idx


def plot_waveform_pair(
    ax_before,
    ax_after,
    before_audio: np.ndarray,
    after_audio: np.ndarray,
    sr: int,
    start_sample: int,
    duration_sec: float,
    title_suffix: str,
):
    """Plot a pair of before/after waveforms."""
    duration_samples = int(duration_sec * sr)
    end_sample = min(
        start_sample + duration_samples, len(before_audio), len(after_audio)
    )

    time = np.arange(end_sample - start_sample) / sr

    before_segment = before_audio[start_sample:end_sample]
    after_segment = after_audio[start_sample:end_sample]

    # Before
    ax_before.fill_between(time, before_segment, alpha=0.7, color="steelblue")
    ax_before.plot(time, before_segment, linewidth=0.3, color="darkblue")
    ax_before.set_title(f"Before {title_suffix}", fontsize=11)
    ax_before.set_ylabel("Amplitude")
    ax_before.set_ylim(-1.1, 1.1)
    ax_before.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.8)
    ax_before.axhline(y=-1.0, color="red", linestyle="--", linewidth=1, alpha=0.8)
    ax_before.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax_before.grid(True, alpha=0.3)

    # After
    ax_after.fill_between(time, after_segment, alpha=0.7, color="forestgreen")
    ax_after.plot(time, after_segment, linewidth=0.3, color="darkgreen")
    ax_after.set_title(f"After {title_suffix}", fontsize=11)
    ax_after.set_ylabel("Amplitude")
    ax_after.set_ylim(-1.1, 1.1)
    ax_after.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.8)
    ax_after.axhline(y=-1.0, color="red", linestyle="--", linewidth=1, alpha=0.8)
    ax_after.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax_after.grid(True, alpha=0.3)

    # Peak annotations
    before_peak: float = float(np.max(np.abs(before_segment)))
    after_peak: float = float(np.max(np.abs(after_segment)))
    ax_before.text(
        0.02,
        0.98,
        f"Peak: {before_peak:.3f}",
        transform=ax_before.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax_after.text(
        0.02,
        0.98,
        f"Peak: {after_peak:.3f}",
        transform=ax_after.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def create_multiscale_report(
    before_path: Path,
    after_path: Path,
    output_path: Path,
    title: str = "De-limiter Multi-Scale Waveform Comparison",
) -> None:
    """Create multi-scale waveform comparison report."""
    before_audio, sr = load_audio_mono(before_path)
    after_audio, _ = load_audio_mono(after_path)

    # Find a loud region to focus on
    center_idx = find_loud_region(before_audio, sr, window_sec=10.0)

    # Zoom levels: 10s, 1s, 0.1s
    zoom_levels = [
        (10.0, "10 sec"),
        (1.0, "1 sec"),
        (0.1, "100 ms"),
    ]

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    for row, (duration_sec, label) in enumerate(zoom_levels):
        duration_samples = int(duration_sec * sr)
        start_sample = max(0, center_idx - duration_samples // 2)

        ax_before = fig.add_subplot(gs[row, 0])
        ax_after = fig.add_subplot(gs[row, 1])

        plot_waveform_pair(
            ax_before,
            ax_after,
            before_audio,
            after_audio,
            sr,
            start_sample,
            duration_sec,
            f"({label})",
        )

        # Add time axis label only on bottom row
        if row == len(zoom_levels) - 1:
            ax_before.set_xlabel("Time (sec)")
            ax_after.set_xlabel("Time (sec)")

    # Add explanation text
    fig.text(
        0.5,
        0.02,
        "Red dashed lines = clipping threshold (±1.0). "
        "Notice how peaks are reduced and waveform shape is restored after de-limiting.",
        ha="center",
        fontsize=10,
        style="italic",
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Multi-scale waveform report saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-scale waveform comparison")
    p.add_argument("--before", type=Path, required=True, help="Input (before) WAV file")
    p.add_argument("--after", type=Path, required=True, help="Output (after) WAV file")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("waveform_multiscale.png"),
        help="Output image path",
    )
    p.add_argument(
        "--title",
        type=str,
        default="De-limiter Multi-Scale Waveform Comparison",
        help="Report title",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.before.exists():
        print(f"Error: Before file not found: {args.before}", file=sys.stderr)
        return 1
    if not args.after.exists():
        print(f"Error: After file not found: {args.after}", file=sys.stderr)
        return 1

    create_multiscale_report(args.before, args.after, args.output, args.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
