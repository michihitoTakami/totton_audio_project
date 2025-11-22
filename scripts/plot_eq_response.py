#!/usr/bin/env python3
"""
EQ Frequency Response Plotter
Visualizes the frequency response of an AutoEq/Equalizer APO format EQ profile.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def parse_eq_file(filepath: str) -> tuple[float, list[dict]]:
    """Parse AutoEq/Equalizer APO format EQ file."""
    preamp_db = 0.0
    bands: list[dict] = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse preamp
            preamp_match = re.match(
                r"Preamp:\s*([-+]?\d*\.?\d+)\s*dB?", line, re.IGNORECASE
            )
            if preamp_match:
                preamp_db = float(preamp_match.group(1))
                continue

            # Parse filter
            filter_match = re.match(
                r"Filter\s*\d+:\s*(ON|OFF)\s+(PK|LS|HS|LP|HP)\s+Fc\s+([\d.]+)\s*Hz\s+Gain\s+([-+]?\d*\.?\d+)\s*dB\s+Q\s+([\d.]+)",
                line,
                re.IGNORECASE,
            )
            if filter_match:
                enabled = filter_match.group(1).upper() == "ON"
                filter_type = filter_match.group(2).upper()
                fc = float(filter_match.group(3))
                gain = float(filter_match.group(4))
                q = float(filter_match.group(5))

                if enabled:
                    bands.append({"type": filter_type, "fc": fc, "gain": gain, "q": q})

    return preamp_db, bands


def calculate_biquad_coeffs(
    filter_type: str, fc: float, gain_db: float, q: float, fs: float
) -> tuple:
    """
    Calculate biquad filter coefficients using Audio EQ Cookbook formulas.
    Returns (b0, b1, b2, a0, a1, a2)
    """
    A = 10 ** (gain_db / 40.0)  # sqrt(10^(dB/20))
    w0 = 2 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * q)

    if filter_type == "PK":  # Peaking EQ
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    elif filter_type == "LS":  # Low Shelf
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "HS":  # High Shelf
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    else:
        # Passthrough for unsupported types
        b0, b1, b2 = 1, 0, 0
        a0, a1, a2 = 1, 0, 0

    return b0, b1, b2, a0, a1, a2


def biquad_frequency_response(
    coeffs: tuple, freqs: np.ndarray, fs: float
) -> np.ndarray:
    """Calculate frequency response of a biquad filter."""
    b0, b1, b2, a0, a1, a2 = coeffs

    # Normalize coefficients
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    # z = e^(j*w) where w = 2*pi*f/fs
    w = 2 * np.pi * freqs / fs
    np.exp(1j * w)
    z_inv = np.exp(-1j * w)
    z_inv2 = np.exp(-2j * w)

    # H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
    numerator = b0 + b1 * z_inv + b2 * z_inv2
    denominator = 1 + a1 * z_inv + a2 * z_inv2

    return numerator / denominator


def compute_eq_response(
    preamp_db: float, bands: list[dict], freqs: np.ndarray, fs: float
) -> np.ndarray:
    """Compute combined EQ frequency response."""
    # Start with preamp
    preamp_linear = 10 ** (preamp_db / 20.0)
    response = np.ones(len(freqs), dtype=complex) * preamp_linear

    # Multiply by each band's response
    for band in bands:
        coeffs = calculate_biquad_coeffs(
            band["type"], band["fc"], band["gain"], band["q"], fs
        )
        band_response = biquad_frequency_response(coeffs, freqs, fs)
        response *= band_response

    return response


def plot_eq_response(eq_file: str, output_file: str = None, sample_rate: float = 44100):
    """Plot EQ frequency response."""
    # Parse EQ file
    preamp_db, bands = parse_eq_file(eq_file)

    print(f"EQ Profile: {Path(eq_file).stem}")
    print(f"  Preamp: {preamp_db} dB")
    print(f"  Bands: {len(bands)}")
    for i, band in enumerate(bands):
        print(
            f"    {i+1}: {band['type']} Fc={band['fc']:.1f}Hz Gain={band['gain']:.1f}dB Q={band['q']:.2f}"
        )

    # Generate frequency points (log scale from 20Hz to 20kHz)
    freqs = np.logspace(np.log10(20), np.log10(20000), 1000)

    # Compute response
    response = compute_eq_response(preamp_db, bands, freqs, sample_rate)
    magnitude_db = 20 * np.log10(np.abs(response))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.semilogx(freqs, magnitude_db, "b-", linewidth=2, label="Combined EQ Response")

    # Plot individual bands (lighter)
    for band in bands:
        coeffs = calculate_biquad_coeffs(
            band["type"], band["fc"], band["gain"], band["q"], sample_rate
        )
        band_response = biquad_frequency_response(coeffs, freqs, sample_rate)
        band_db = 20 * np.log10(np.abs(band_response))
        ax.semilogx(freqs, band_db, "gray", linewidth=0.5, alpha=0.5)

    # Add preamp line
    ax.axhline(
        y=preamp_db,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Preamp ({preamp_db} dB)",
    )

    # Formatting
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Magnitude (dB)", fontsize=12)
    ax.set_title(f"EQ Frequency Response: {Path(eq_file).stem}", fontsize=14)
    ax.set_xlim(20, 20000)
    ax.set_ylim(-25, 15)
    ax.grid(True, which="both", linestyle="-", alpha=0.3)
    ax.legend(loc="upper right")

    # X-axis ticks
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.set_xticklabels(
        ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]
    )

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()

    return magnitude_db, freqs


def compare_eq_profiles(
    eq_files: list[str], output_file: str = None, sample_rate: float = 44100
):
    """Compare multiple EQ profiles on the same plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    freqs = np.logspace(np.log10(20), np.log10(20000), 1000)
    colors = plt.cm.tab10(np.linspace(0, 1, len(eq_files)))  # type: ignore[attr-defined]

    for eq_file, color in zip(eq_files, colors):
        preamp_db, bands = parse_eq_file(eq_file)
        response = compute_eq_response(preamp_db, bands, freqs, sample_rate)
        magnitude_db = 20 * np.log10(np.abs(response))

        label = f"{Path(eq_file).stem} ({len(bands)} bands, {preamp_db:+.1f}dB)"
        ax.semilogx(freqs, magnitude_db, color=color, linewidth=2, label=label)

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Magnitude (dB)", fontsize=12)
    ax.set_title("EQ Profile Comparison", fontsize=14)
    ax.set_xlim(20, 20000)
    ax.set_ylim(-25, 15)
    ax.grid(True, which="both", linestyle="-", alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.set_xticklabels(
        ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]
    )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to: {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    # Default paths
    eq_dir = Path(__file__).parent.parent / "data" / "EQ"
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        # Plot specified file
        eq_file_arg = sys.argv[1]
        output_file = output_dir / f"{Path(eq_file_arg).stem}_response.png"
        plot_eq_response(eq_file_arg, str(output_file))
    else:
        # Plot all EQ files in data/EQ directory
        eq_files: list[Path] = list(eq_dir.glob("*.txt"))

        if not eq_files:
            print(f"No EQ files found in {eq_dir}")
            sys.exit(1)

        # Plot each profile
        for eq_file in eq_files:
            output_file = output_dir / f"{eq_file.stem}_response.png"
            plot_eq_response(str(eq_file), str(output_file))
            print()

        # Compare all profiles
        if len(eq_files) > 1:
            compare_eq_profiles(
                [str(f) for f in eq_files], str(output_dir / "eq_comparison.png")
            )
