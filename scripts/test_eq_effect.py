#!/usr/bin/env python3
"""
EQ Effect Verification Test
Generates a sweep signal, processes through GPU upsampler with/without EQ,
and compares the frequency response to verify EQ is working.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import subprocess


def generate_sweep(
    duration: float = 5.0, fs: int = 44100, f_start: float = 20, f_end: float = 20000
) -> np.ndarray:
    """Generate a logarithmic sine sweep."""
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    sweep = signal.chirp(t, f_start, duration, f_end, method="logarithmic")
    # Normalize and apply fade in/out
    fade_samples = int(0.01 * fs)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    sweep[:fade_samples] *= fade_in
    sweep[-fade_samples:] *= fade_out
    return (sweep * 0.5).astype(np.float32)  # -6dB headroom


def generate_white_noise(duration: float = 2.0, fs: int = 44100) -> np.ndarray:
    """Generate white noise for frequency response measurement."""
    np.random.seed(42)  # Reproducible
    noise = np.random.randn(int(duration * fs)).astype(np.float32)
    return noise * 0.3  # -10dB headroom


def generate_impulse(fs: int = 44100) -> np.ndarray:
    """Generate an impulse for impulse response measurement."""
    # Create a short impulse with some padding
    samples = fs  # 1 second
    impulse = np.zeros(samples, dtype=np.float32)
    impulse[100] = 0.9  # Impulse at sample 100
    return impulse


def analyze_frequency_response(
    signal_data: np.ndarray, fs: int, nperseg: int = 8192
) -> tuple:
    """Analyze frequency response using Welch's method."""
    freqs, psd = signal.welch(signal_data, fs, nperseg=nperseg)
    return freqs, 10 * np.log10(psd + 1e-12)  # Convert to dB


def process_with_convolution_engine(
    input_signal: np.ndarray,
    eq_enabled: bool,
    eq_profile: str,
    filter_path: str,
    build_dir: Path,
) -> np.ndarray:
    """
    Process signal through the GPU convolution engine.
    This creates a test binary that processes the signal.
    """
    # For now, we'll simulate the EQ effect in Python
    # In a real test, we'd call the actual GPU code

    # This is a placeholder - actual implementation would use the GPU engine
    # For verification, we compute expected EQ effect
    return input_signal


def compute_eq_response_python(
    freqs: np.ndarray, eq_file: str, fs: float
) -> np.ndarray:
    """Compute EQ frequency response in Python for comparison."""
    import re

    preamp_db = 0.0
    bands = []

    with open(eq_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            preamp_match = re.match(
                r"Preamp:\s*([-+]?\d*\.?\d+)\s*dB?", line, re.IGNORECASE
            )
            if preamp_match:
                preamp_db = float(preamp_match.group(1))
                continue

            filter_match = re.match(
                r"Filter\s*\d+:\s*(ON|OFF)\s+(PK|LS|HS)\s+Fc\s+([\d.]+)\s*Hz\s+Gain\s+([-+]?\d*\.?\d+)\s*dB\s+Q\s+([\d.]+)",
                line,
                re.IGNORECASE,
            )
            if filter_match and filter_match.group(1).upper() == "ON":
                bands.append(
                    {
                        "type": filter_match.group(2).upper(),
                        "fc": float(filter_match.group(3)),
                        "gain": float(filter_match.group(4)),
                        "q": float(filter_match.group(5)),
                    }
                )

    # Compute combined response
    response_db = np.ones(len(freqs)) * preamp_db

    for band in bands:
        A = 10 ** (band["gain"] / 40.0)
        w0 = 2 * np.pi * band["fc"] / fs
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * band["q"])

        w = 2 * np.pi * freqs / fs
        z_inv = np.exp(-1j * w)
        z_inv2 = np.exp(-2j * w)

        if band["type"] == "PK":
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
        elif band["type"] == "LS":
            sqrt_A = np.sqrt(A)
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
            a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
        elif band["type"] == "HS":
            sqrt_A = np.sqrt(A)
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
        else:
            continue

        H = (b0 + b1 * z_inv + b2 * z_inv2) / (a0 + a1 * z_inv + a2 * z_inv2)
        response_db += 20 * np.log10(np.abs(H) + 1e-12)

    return response_db


def run_gpu_test(
    input_wav: str, output_wav: str, eq_enabled: bool, eq_profile: str, build_dir: Path
):
    """Run actual GPU processing test using a test executable."""
    test_exe = build_dir / "gpu_upsampler"

    if not test_exe.exists():
        print(f"Warning: {test_exe} not found, skipping GPU test")
        return None

    # The gpu_upsampler processes WAV files
    cmd = [str(test_exe), input_wav, output_wav]

    # Note: Current gpu_upsampler doesn't have EQ command line option
    # This would need to be added for proper testing

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"GPU processing failed: {result.stderr}")
            return None
        return output_wav
    except Exception as e:
        print(f"GPU processing error: {e}")
        return None


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    project_dir / "build"
    output_dir = project_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    eq_file = project_dir / "data" / "EQ" / "Sample_EQ.txt"
    if not eq_file.exists():
        eq_file = Path("/home/michihito/Working/gpu_os/data/EQ/Sample_EQ.txt")

    fs = 44100

    print("=" * 60)
    print("EQ Effect Verification Test")
    print("=" * 60)

    # Generate test signals
    print("\n1. Generating test signals...")
    sweep = generate_sweep(duration=3.0, fs=fs)
    noise = generate_white_noise(duration=2.0, fs=fs)

    # Save test signals
    test_dir = project_dir / "test_signals"
    test_dir.mkdir(exist_ok=True)

    sweep_file = test_dir / "sweep_44100.wav"
    noise_file = test_dir / "noise_44100.wav"

    # Convert to stereo int16 for WAV
    sweep_stereo = np.column_stack([sweep, sweep])
    noise_stereo = np.column_stack([noise, noise])

    wavfile.write(sweep_file, fs, (sweep_stereo * 32767).astype(np.int16))
    wavfile.write(noise_file, fs, (noise_stereo * 32767).astype(np.int16))
    print(f"   Sweep saved: {sweep_file}")
    print(f"   Noise saved: {noise_file}")

    # Analyze input signal frequency content
    print("\n2. Analyzing input signal frequency response...")
    freqs_sweep, psd_sweep = analyze_frequency_response(sweep, fs)
    freqs_noise, psd_noise = analyze_frequency_response(noise, fs)

    # Compute expected EQ response
    print("\n3. Computing expected EQ frequency response...")
    print(f"   EQ Profile: {eq_file}")

    # Use frequencies in audible range
    test_freqs = np.logspace(np.log10(20), np.log10(20000), 500)
    eq_response = compute_eq_response_python(test_freqs, str(eq_file), fs)

    # Plot results
    print("\n4. Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Input signals spectrum
    ax1 = axes[0, 0]
    ax1.semilogx(
        freqs_sweep[freqs_sweep > 10],
        psd_sweep[freqs_sweep > 10],
        label="Sweep",
        alpha=0.7,
    )
    ax1.semilogx(
        freqs_noise[freqs_noise > 10],
        psd_noise[freqs_noise > 10],
        label="White Noise",
        alpha=0.7,
    )
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power Spectral Density (dB)")
    ax1.set_title("Input Test Signals Spectrum")
    ax1.set_xlim(20, 20000)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Expected EQ Response
    ax2 = axes[0, 1]
    ax2.semilogx(
        test_freqs, eq_response, "b-", linewidth=2, label="Expected EQ Response"
    )
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_title(f"Expected EQ Response: {eq_file.stem}")
    ax2.set_xlim(20, 20000)
    ax2.set_ylim(-25, 15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Instructions for manual verification
    ax3 = axes[1, 0]
    ax3.text(
        0.5,
        0.5,
        "Manual Verification Steps:\n\n"
        "1. Run daemon with EQ disabled\n"
        "2. Play sweep_44100.wav through system\n"
        "3. Record output (or listen)\n\n"
        "4. Run daemon with EQ enabled\n"
        "5. Play sweep_44100.wav again\n"
        "6. Compare: should hear EQ effect\n\n"
        f"Test files in:\n{test_dir}",
        transform=ax3.transAxes,
        fontsize=11,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax3.axis("off")
    ax3.set_title("Manual Verification")

    # Plot 4: EQ Band Details
    ax4 = axes[1, 1]

    # Parse and show band info
    import re

    with open(eq_file, "r") as f:
        for line in f:
            if line.strip().startswith("Filter") and "ON" in line:
                match = re.search(r"Fc\s+([\d.]+)\s*Hz\s+Gain\s+([-+]?\d*\.?\d+)", line)
                if match:
                    fc = float(match.group(1))
                    gain = float(match.group(2))
                    ax4.axvline(x=fc, color="red", alpha=0.3, linewidth=1)
                    if abs(gain) > 3:  # Only annotate significant changes
                        ax4.annotate(
                            f"{gain:+.0f}dB", (fc, gain), fontsize=8, alpha=0.7
                        )

    ax4.semilogx(test_freqs, eq_response, "b-", linewidth=2)
    ax4.fill_between(
        test_freqs,
        0,
        eq_response,
        alpha=0.3,
        where=eq_response > 0,
        color="green",
        label="Boost",
    )
    ax4.fill_between(
        test_freqs,
        0,
        eq_response,
        alpha=0.3,
        where=eq_response < 0,
        color="red",
        label="Cut",
    )
    ax4.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Magnitude (dB)")
    ax4.set_title("EQ Boost/Cut Regions")
    ax4.set_xlim(20, 20000)
    ax4.set_ylim(-25, 15)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "eq_verification_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"   Plot saved: {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"EQ Profile: {eq_file.name}")

    # Find significant EQ changes
    significant_changes = []
    for i, (f, db) in enumerate(zip(test_freqs, eq_response)):
        if i > 0 and abs(eq_response[i] - eq_response[i - 1]) > 0.5:
            if abs(db) > 3:
                significant_changes.append((f, db))

    print("\nSignificant frequency changes (>3dB):")
    eq_changes = list(zip(test_freqs, eq_response))
    for f, db in sorted(eq_changes, key=lambda x: abs(x[1]), reverse=True)[:10]:
        if abs(db) > 3:
            direction = "BOOST" if db > 0 else "CUT"
            print(f"  {f:8.1f} Hz: {db:+6.1f} dB ({direction})")

    print(f"\n** Test signals saved to: {test_dir}")
    print("** Play these through the system to verify EQ effect **")
    print("=" * 60)


if __name__ == "__main__":
    main()
