#ifndef EQ_TO_FIR_H
#define EQ_TO_FIR_H

#include "eq_parser.h"
#include <vector>
#include <complex>

namespace EQ {

// Biquad filter coefficients (normalized, a0 = 1)
struct BiquadCoeffs {
    double b0, b1, b2;  // Numerator
    double a1, a2;      // Denominator (a0 normalized to 1)
};

// Calculate biquad coefficients for a single EQ band
// sampleRate: the sample rate at which to design the filter (e.g., 44100)
BiquadCoeffs calculateBiquadCoeffs(const EqBand& band, double sampleRate);

// Compute frequency response of a biquad filter at given frequencies
// frequencies: array of frequencies (Hz)
// coeffs: biquad coefficients
// sampleRate: sample rate used for coefficient calculation
// Returns: complex frequency response H(f) for each frequency
std::vector<std::complex<double>> biquadFrequencyResponse(
    const std::vector<double>& frequencies,
    const BiquadCoeffs& coeffs,
    double sampleRate
);

// Compute combined frequency response of entire EQ profile
// frequencies: array of frequencies (Hz)
// profile: EQ profile with bands and preamp
// sampleRate: sample rate for filter design
// Returns: complex frequency response H_eq(f) = preamp * H_band1 * H_band2 * ...
std::vector<std::complex<double>> computeEqFrequencyResponse(
    const std::vector<double>& frequencies,
    const EqProfile& profile,
    double sampleRate
);

// Generate frequency array for FFT bins
// fftSize: size of FFT (e.g., 2^20 for 1M-tap filter)
// sampleRate: output sample rate after upsampling (e.g., 705600)
// Returns: frequency values for each FFT bin (DC to Nyquist, then negative)
std::vector<double> generateFftFrequencies(size_t fftSize, double sampleRate);

// Apply EQ response to existing filter FFT
// filterFFT: complex FFT of FIR filter (modified in place)
// eqResponse: complex EQ frequency response (same size as filterFFT)
// Computes: H_combined = H_filter * H_eq (element-wise complex multiplication)
void applyEqToFilterFft(
    std::vector<std::complex<float>>& filterFFT,
    const std::vector<std::complex<double>>& eqResponse
);

// Convenience function: compute EQ response sized for filter FFT
// fftSize: size of filter FFT
// inputSampleRate: original sample rate (44100 or 48000)
// profile: EQ profile
// Returns: EQ frequency response sized for filter FFT
std::vector<std::complex<double>> computeEqResponseForFft(
    size_t fftSize,
    double inputSampleRate,
    const EqProfile& profile
);

}  // namespace EQ

#endif  // EQ_TO_FIR_H
