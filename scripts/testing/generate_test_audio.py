#!/usr/bin/env python3
"""
Generate test audio files for GPU upsampler testing
"""

import numpy as np
import scipy.io.wavfile as wavfile
import argparse
from pathlib import Path


def generate_sine_wave(frequency, duration, sample_rate, amplitude=0.5):
    """Generate a pure sine wave"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal.astype(np.float32)


def generate_sweep(f_start, f_end, duration, sample_rate, amplitude=0.5):
    """Generate a logarithmic frequency sweep"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Logarithmic sweep
    k = (f_end / f_start) ** (1 / duration)
    phase = 2 * np.pi * f_start * duration / np.log(k) * (k ** (t / duration) - 1)
    signal = amplitude * np.sin(phase)
    return signal.astype(np.float32)


def generate_impulse(duration, sample_rate, amplitude=1.0):
    """Generate an impulse (single sample at max amplitude)"""
    signal = np.zeros(int(sample_rate * duration), dtype=np.float32)
    signal[sample_rate // 10] = amplitude  # Impulse at 0.1 second
    return signal


def main():
    parser = argparse.ArgumentParser(description="Generate test audio files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data",
        help="Output directory for test files",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=44100, help="Sample rate (default: 44100 Hz)"
    )
    parser.add_argument(
        "--duration", type=float, default=2.0, help="Duration in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="Signal amplitude (0-1]. Set close to 1.0 for headroom tests.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    sample_rate = args.sample_rate
    duration = args.duration

    print("Generating test audio files...")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} seconds")
    print(f"  Output directory: {output_dir}")
    print()

    # 1. Pure tone (1 kHz)
    print("1. Generating 1 kHz sine wave...")
    sine_1k = generate_sine_wave(1000, duration, sample_rate, amplitude=args.amplitude)
    output_path = output_dir / f"test_sine_1khz_{sample_rate}hz.wav"
    wavfile.write(output_path, sample_rate, sine_1k)
    print(f"   Saved: {output_path}")

    # 2. High frequency tone (10 kHz)
    print("2. Generating 10 kHz sine wave...")
    sine_10k = generate_sine_wave(
        10000, duration, sample_rate, amplitude=args.amplitude
    )
    output_path = output_dir / f"test_sine_10khz_{sample_rate}hz.wav"
    wavfile.write(output_path, sample_rate, sine_10k)
    print(f"   Saved: {output_path}")

    # 3. Frequency sweep (20 Hz to 20 kHz)
    print("3. Generating frequency sweep (20 Hz - 20 kHz)...")
    sweep = generate_sweep(20, 20000, duration, sample_rate, amplitude=args.amplitude)
    output_path = output_dir / f"test_sweep_{sample_rate}hz.wav"
    wavfile.write(output_path, sample_rate, sweep)
    print(f"   Saved: {output_path}")

    # 4. Impulse response test
    print("4. Generating impulse...")
    impulse = generate_impulse(duration, sample_rate, amplitude=args.amplitude)
    output_path = output_dir / f"test_impulse_{sample_rate}hz.wav"
    wavfile.write(output_path, sample_rate, impulse)
    print(f"   Saved: {output_path}")

    # 5. Stereo test (L: 1kHz, R: 2kHz)
    print("5. Generating stereo test (L: 1kHz, R: 2kHz)...")
    left = generate_sine_wave(1000, duration, sample_rate, amplitude=args.amplitude)
    right = generate_sine_wave(2000, duration, sample_rate, amplitude=args.amplitude)
    stereo = np.stack([left, right], axis=1)
    output_path = output_dir / f"test_stereo_{sample_rate}hz.wav"
    wavfile.write(output_path, sample_rate, stereo)
    print(f"   Saved: {output_path}")

    print()
    print("All test files generated successfully!")
    print("Total files: 5")


if __name__ == "__main__":
    main()
