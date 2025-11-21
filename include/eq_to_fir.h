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

// Generate frequency array for R2C FFT bins (positive frequencies only)
// numBins: number of FFT bins (N/2+1 for R2C output)
// fullFftSize: full FFT size N (for frequency resolution calculation)
// sampleRate: output sample rate after upsampling (e.g., 705600)
// Returns: frequency values for each FFT bin (DC to Nyquist only)
std::vector<double> generateR2cFftFrequencies(size_t numBins, size_t fullFftSize, double sampleRate);

// Compute EQ response sized for filter FFT (R2C output)
// (Internal helper, used by computeEqMagnitudeForFft)
// filterFftSize: R2C FFT output size (N/2+1)
// fullFftSize: full FFT size (N)
// outputSampleRate: sample rate after upsampling (e.g., 705600)
// profile: EQ profile
// Returns: EQ frequency response sized for filter FFT (N/2+1 bins)
std::vector<std::complex<double>> computeEqResponseForFft(
    size_t filterFftSize,
    size_t fullFftSize,
    double outputSampleRate,
    const EqProfile& profile
);

// Compute EQ magnitude response only (for minimum phase reconstruction)
// Same parameters as computeEqResponseForFft
// Returns: EQ magnitude |H_eq(f)| for each FFT bin (N/2+1 values)
// Note: Phase is discarded; use with applyEqMagnitude() for minimum phase EQ
std::vector<double> computeEqMagnitudeForFft(
    size_t filterFftSize,
    size_t fullFftSize,
    double outputSampleRate,
    const EqProfile& profile
);

}  // namespace EQ

#endif  // EQ_TO_FIR_H
