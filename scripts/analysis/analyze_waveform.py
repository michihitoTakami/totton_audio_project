#!/usr/bin/env python3
"""
Waveform analysis script to detect discontinuities and clicks
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import sys


def detect_clicks(audio, sr, threshold_db=-40, window_ms=1):
    """
    Detect clicks by looking for sudden amplitude changes

    Args:
        audio: audio samples (mono or stereo)
        sr: sample rate
        threshold_db: amplitude threshold in dB
        window_ms: window size in milliseconds
    """
    if audio.ndim > 1:
        # Use left channel for analysis
        audio = audio[:, 0]

    # Calculate first derivative (rate of change)
    diff = np.diff(audio)

    # Calculate moving average of absolute derivative
    int(sr * window_ms / 1000)
    diff_abs = np.abs(diff)

    # Find peaks in derivative (sudden changes)
    threshold = 10 ** (threshold_db / 20)
    clicks = np.where(diff_abs > threshold)[0]

    # Group nearby clicks (within 10ms)
    if len(clicks) > 0:
        click_groups = []
        current_group = [clicks[0]]
        for i in range(1, len(clicks)):
            if clicks[i] - clicks[i - 1] < sr * 0.01:  # 10ms
                current_group.append(clicks[i])
            else:
                click_groups.append(current_group)
                current_group = [clicks[i]]
        click_groups.append(current_group)

        # Take the strongest click in each group
        click_positions = []
        click_amplitudes = []
        for group in click_groups:
            max_idx = group[np.argmax(diff_abs[group])]
            click_positions.append(max_idx)
            click_amplitudes.append(diff_abs[max_idx])

        return np.array(click_positions), np.array(click_amplitudes)

    return np.array([]), np.array([])


def analyze_discontinuities(audio, sr):
    """
    Analyze audio for discontinuities
    """
    if audio.ndim > 1:
        left = audio[:, 0]
        right = audio[:, 1]
    else:
        left = audio
        right = None

    print("\n=== Waveform Analysis ===")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(left)/sr:.2f} seconds")
    print(f"Channels: {'Stereo' if right is not None else 'Mono'}")
    print(f"Peak amplitude: {np.max(np.abs(left)):.6f}")
    print(f"RMS level: {np.sqrt(np.mean(left**2)):.6f}")

    # Check for clipping
    clipped = np.sum(np.abs(left) >= 0.99)
    if clipped > 0:
        print(f"\nWARNING: {clipped} samples clipped ({100*clipped/len(left):.4f}%)")

    # Check for DC offset
    dc_offset = np.mean(left)
    if abs(dc_offset) > 0.01:
        print(f"\nWARNING: DC offset detected: {dc_offset:.6f}")

    # Detect clicks at various thresholds
    print("\n=== Click Detection ===")
    for threshold_db in [-30, -40, -50, -60]:
        clicks, amplitudes = detect_clicks(audio, sr, threshold_db)
        if len(clicks) > 0:
            print(f"Threshold {threshold_db} dB: {len(clicks)} clicks detected")
            print(f"  First 10 click positions (samples): {clicks[:10]}")
            print(f"  First 10 click positions (time): {clicks[:10]/sr}")
            print(f"  Max click amplitude: {20*np.log10(np.max(amplitudes)):.1f} dB")
        else:
            print(f"Threshold {threshold_db} dB: No clicks detected")

    return left, right


def plot_waveform(audio, sr, output_path, start_sec=0, duration_sec=0.1):
    """
    Plot a segment of the waveform
    """
    if audio.ndim > 1:
        left = audio[:, 0]
    else:
        left = audio

    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)

    segment = left[start_sample:end_sample]
    time = np.arange(len(segment)) / sr + start_sec

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time * 1000, segment)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform ({start_sec:.2f}s - {start_sec+duration_sec:.2f}s)")
    plt.grid(True, alpha=0.3)

    # Plot derivative to show sudden changes
    plt.subplot(2, 1, 2)
    diff = np.diff(segment)
    time_diff = time[:-1]
    plt.plot(time_diff * 1000, diff)
    plt.xlabel("Time (ms)")
    plt.ylabel("Rate of Change")
    plt.title("First Derivative (detects clicks/discontinuities)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nWaveform plot saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python analyze_waveform.py <audio_file.wav> [start_sec] [duration_sec]"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

    print(f"Analyzing: {input_file}")

    # Load audio
    audio, sr = sf.read(input_file)

    # Analyze
    left, right = analyze_discontinuities(audio, sr)

    # Plot
    output_plot = input_file.replace(".wav", "_analysis.png")
    plot_waveform(audio, sr, output_plot, start_sec, duration_sec)

    # Also plot a few other sections
    for i, t in enumerate([0.5, 2.0, 5.0]):
        if t < len(audio) / sr:
            output_plot2 = input_file.replace(".wav", f"_analysis_{i+2}.png")
            plot_waveform(audio, sr, output_plot2, t, duration_sec)


if __name__ == "__main__":
    main()
