#!/usr/bin/env python3
"""
Verify frequency response of upsampled audio
Compare input and output spectra
"""

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import argparse


def analyze_frequency_response(input_file, output_file, output_plot=None):
    """
    Analyze and compare frequency response between input and output
    """
    # Read input file
    sr_in, data_in = wavfile.read(input_file)
    if len(data_in.shape) > 1:
        data_in = data_in[:, 0]  # Take left channel if stereo
    data_in = data_in.astype(np.float32)

    # Read output file
    sr_out, data_out = wavfile.read(output_file)
    if len(data_out.shape) > 1:
        data_out = data_out[:, 0]  # Take left channel if stereo
    data_out = data_out.astype(np.float32)

    print(f"Input:  {len(data_in)} samples @ {sr_in} Hz")
    print(f"Output: {len(data_out)} samples @ {sr_out} Hz")
    print(f"Upsample ratio: {sr_out / sr_in}x")
    print()

    # Compute FFT
    fft_in = np.fft.rfft(data_in)
    fft_out = np.fft.rfft(data_out)

    # Frequency bins
    freqs_in = np.fft.rfftfreq(len(data_in), 1 / sr_in)
    freqs_out = np.fft.rfftfreq(len(data_out), 1 / sr_out)

    # Magnitude spectrum (dB)
    mag_in_db = 20 * np.log10(np.abs(fft_in) + 1e-12)
    mag_out_db = 20 * np.log10(np.abs(fft_out) + 1e-12)

    # Find peaks
    peak_idx_in = np.argmax(np.abs(fft_in))
    peak_freq_in = freqs_in[peak_idx_in]
    peak_mag_in = mag_in_db[peak_idx_in]

    peak_idx_out = np.argmax(np.abs(fft_out))
    peak_freq_out = freqs_out[peak_idx_out]
    peak_mag_out = mag_out_db[peak_idx_out]

    print(f"Input peak:  {peak_freq_in:.2f} Hz @ {peak_mag_in:.2f} dB")
    print(f"Output peak: {peak_freq_out:.2f} Hz @ {peak_mag_out:.2f} dB")
    print(f"Peak frequency match: {abs(peak_freq_in - peak_freq_out) < 10} Hz")
    print()

    # Check stopband attenuation (after 22.05 kHz)
    stopband_start = 22050
    stopband_mask = freqs_out > stopband_start
    if np.any(stopband_mask):
        stopband_energy = mag_out_db[stopband_mask]
        max_stopband_energy = np.max(stopband_energy)
        print(
            f"Stopband (>{stopband_start} Hz) max energy: {max_stopband_energy:.2f} dB"
        )
        print(f"Stopband attenuation: {peak_mag_out - max_stopband_energy:.2f} dB")
    print()

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Input spectrum
    axes[0].plot(freqs_in / 1000, mag_in_db, linewidth=0.5)
    axes[0].set_xlim(0, sr_in / 2000)
    axes[0].set_ylim(np.max(mag_in_db) - 100, np.max(mag_in_db) + 10)
    axes[0].axvline(
        peak_freq_in / 1000,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Peak: {peak_freq_in:.1f} Hz",
    )
    axes[0].set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title(f"Input Spectrum ({sr_in} Hz)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Output spectrum
    axes[1].plot(freqs_out / 1000, mag_out_db, linewidth=0.5)
    axes[1].set_xlim(0, 30)  # Show up to 30 kHz to see stopband
    axes[1].set_ylim(np.max(mag_out_db) - 100, np.max(mag_out_db) + 10)
    axes[1].axvline(
        peak_freq_out / 1000,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Peak: {peak_freq_out:.1f} Hz",
    )
    axes[1].axvline(
        20, color="g", linestyle="--", alpha=0.5, label="Passband edge (20 kHz)"
    )
    axes[1].axvline(
        22.05,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label="Stopband start (22.05 kHz)",
    )
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_title(f"Output Spectrum ({sr_out} Hz)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    if output_plot:
        plt.savefig(output_plot, dpi=150)
        print(f"Plot saved: {output_plot}")
    else:
        plt.savefig("plots/analysis/frequency_verification.png", dpi=150)
        print("Plot saved: plots/analysis/frequency_verification.png")

    return {
        "peak_freq_in": peak_freq_in,
        "peak_freq_out": peak_freq_out,
        "peak_mag_in": peak_mag_in,
        "peak_mag_out": peak_mag_out,
        "frequency_match": abs(peak_freq_in - peak_freq_out) < 10,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify frequency response of upsampled audio"
    )
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("output", help="Output WAV file (upsampled)")
    parser.add_argument("--plot", help="Output plot file", default=None)

    args = parser.parse_args()

    print("=" * 60)
    print("Frequency Response Verification")
    print("=" * 60)
    print()

    results = analyze_frequency_response(args.input, args.output, args.plot)

    print("=" * 60)
    if results["frequency_match"]:
        print("✓ Frequency response verification PASSED")
    else:
        print("✗ Frequency response verification FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
