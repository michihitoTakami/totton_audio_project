#!/usr/bin/env python3
"""Plot EQ test results from CSV files."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_csv(filepath):
    """Load CSV file using numpy."""
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    return data[:, 0], data[:, 1]  # frequency, magnitude


def main():
    test_dir = Path(__file__).parent.parent / "test_output"
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    # Load CSV files
    freq_no_eq, mag_no_eq = load_csv(test_dir / "spectrum_no_eq.csv")
    freq_with_eq, mag_with_eq = load_csv(test_dir / "spectrum_with_eq.csv")
    freq_diff, mag_diff = load_csv(test_dir / "spectrum_difference.csv")

    # Filter to audible range
    mask = (freq_no_eq >= 20) & (freq_no_eq <= 20000)
    freq_no_eq, mag_no_eq = freq_no_eq[mask], mag_no_eq[mask]
    freq_with_eq, mag_with_eq = freq_with_eq[mask], mag_with_eq[mask]
    freq_diff, mag_diff = freq_diff[mask], mag_diff[mask]

    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Spectrum without EQ
    ax1 = axes[0]
    ax1.semilogx(freq_no_eq, mag_no_eq, "b-", linewidth=1)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("Impulse Response Spectrum - WITHOUT EQ")
    ax1.set_xlim(20, 20000)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spectrum with EQ
    ax2 = axes[1]
    ax2.semilogx(freq_with_eq, mag_with_eq, "r-", linewidth=1)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_title("Impulse Response Spectrum - WITH EQ (Sample_EQ)")
    ax2.set_xlim(20, 20000)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Difference (EQ effect)
    ax3 = axes[2]
    ax3.semilogx(freq_diff, mag_diff, "g-", linewidth=2, label="EQ Effect")
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax3.fill_between(
        freq_diff,
        0,
        mag_diff,
        where=mag_diff > 0,
        alpha=0.3,
        color="green",
        label="Boost",
    )
    ax3.fill_between(
        freq_diff, 0, mag_diff, where=mag_diff < 0, alpha=0.3, color="red", label="Cut"
    )
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Magnitude Difference (dB)")
    ax3.set_title("EQ Effect (With EQ - Without EQ)")
    ax3.set_xlim(20, 20000)
    ax3.set_ylim(-20, 15)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "eq_test_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    # Print summary
    print("\nEQ Effect Summary:")
    max_boost_idx = np.argmax(mag_diff)
    max_cut_idx = np.argmin(mag_diff)
    print(
        f"  Max boost: {mag_diff[max_boost_idx]:.1f} dB at {freq_diff[max_boost_idx]:.0f} Hz"
    )
    print(
        f"  Max cut:   {mag_diff[max_cut_idx]:.1f} dB at {freq_diff[max_cut_idx]:.0f} Hz"
    )


if __name__ == "__main__":
    main()
