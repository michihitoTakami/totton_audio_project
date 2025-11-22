#include "eq_to_fir.h"

#include <cmath>
#include <iostream>

namespace EQ {

// Calculate biquad coefficients using Audio EQ Cookbook formulas
// Reference: https://www.w3.org/2011/audio/audio-eq-cookbook.html
BiquadCoeffs calculateBiquadCoeffs(const EqBand& band, double sampleRate) {
    BiquadCoeffs c = {1.0, 0.0, 0.0, 0.0, 0.0};

    if (!band.enabled || band.gain == 0.0) {
        // Pass-through (unity gain)
        return c;
    }

    const double pi = 3.14159265358979323846;
    double A = std::pow(10.0, band.gain / 40.0);  // sqrt(10^(dB/20))
    double w0 = 2.0 * pi * band.frequency / sampleRate;
    double cosW0 = std::cos(w0);
    double sinW0 = std::sin(w0);
    double alpha = sinW0 / (2.0 * band.q);

    double a0;

    switch (band.type) {
    case FilterType::PK: {
        // Peaking EQ
        c.b0 = 1.0 + alpha * A;
        c.b1 = -2.0 * cosW0;
        c.b2 = 1.0 - alpha * A;
        a0 = 1.0 + alpha / A;
        c.a1 = -2.0 * cosW0;
        c.a2 = 1.0 - alpha / A;
        break;
    }

    case FilterType::LS: {
        // Low Shelf
        double sqrtA = std::sqrt(A);
        double sqrtA2alpha = 2.0 * sqrtA * alpha;
        c.b0 = A * ((A + 1.0) - (A - 1.0) * cosW0 + sqrtA2alpha);
        c.b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosW0);
        c.b2 = A * ((A + 1.0) - (A - 1.0) * cosW0 - sqrtA2alpha);
        a0 = (A + 1.0) + (A - 1.0) * cosW0 + sqrtA2alpha;
        c.a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cosW0);
        c.a2 = (A + 1.0) + (A - 1.0) * cosW0 - sqrtA2alpha;
        break;
    }

    case FilterType::HS: {
        // High Shelf
        double sqrtA = std::sqrt(A);
        double sqrtA2alpha = 2.0 * sqrtA * alpha;
        c.b0 = A * ((A + 1.0) + (A - 1.0) * cosW0 + sqrtA2alpha);
        c.b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosW0);
        c.b2 = A * ((A + 1.0) + (A - 1.0) * cosW0 - sqrtA2alpha);
        a0 = (A + 1.0) - (A - 1.0) * cosW0 + sqrtA2alpha;
        c.a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosW0);
        c.a2 = (A + 1.0) - (A - 1.0) * cosW0 - sqrtA2alpha;
        break;
    }

    case FilterType::LP:
    case FilterType::HP:
    default:
        // Not implemented - return unity
        std::cerr << "EQ: Filter type " << filterTypeName(band.type)
                  << " not implemented, using bypass" << std::endl;
        return c;
    }

    // Normalize by a0
    c.b0 /= a0;
    c.b1 /= a0;
    c.b2 /= a0;
    c.a1 /= a0;
    c.a2 /= a0;

    return c;
}

std::vector<std::complex<double>> biquadFrequencyResponse(const std::vector<double>& frequencies,
                                                          const BiquadCoeffs& coeffs,
                                                          double sampleRate) {
    const double pi = 3.14159265358979323846;
    std::vector<std::complex<double>> response(frequencies.size());

    for (size_t i = 0; i < frequencies.size(); ++i) {
        double f = frequencies[i];
        // Handle negative frequencies (for full FFT spectrum)
        if (f < 0)
            f = -f;

        double w = 2.0 * pi * f / sampleRate;
        std::complex<double> z = std::exp(std::complex<double>(0, -w));  // z = e^(-jw)
        std::complex<double> z2 = z * z;

        // H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
        std::complex<double> num = coeffs.b0 + coeffs.b1 * z + coeffs.b2 * z2;
        std::complex<double> den = 1.0 + coeffs.a1 * z + coeffs.a2 * z2;

        response[i] = num / den;
    }

    return response;
}

std::vector<std::complex<double>> computeEqFrequencyResponse(const std::vector<double>& frequencies,
                                                             const EqProfile& profile,
                                                             double sampleRate) {
    size_t N = frequencies.size();
    std::vector<std::complex<double>> response(N, std::complex<double>(1.0, 0.0));

    // Apply preamp
    if (profile.preampDb != 0.0) {
        double preampLinear = std::pow(10.0, profile.preampDb / 20.0);
        for (size_t i = 0; i < N; ++i) {
            response[i] *= preampLinear;
        }
    }

    // Multiply response from each band
    for (const auto& band : profile.bands) {
        if (!band.enabled)
            continue;

        BiquadCoeffs coeffs = calculateBiquadCoeffs(band, sampleRate);
        auto bandResponse = biquadFrequencyResponse(frequencies, coeffs, sampleRate);

        for (size_t i = 0; i < N; ++i) {
            response[i] *= bandResponse[i];
        }
    }

    return response;
}

std::vector<double> generateR2cFftFrequencies(size_t numBins, size_t fullFftSize,
                                              double sampleRate) {
    // For R2C FFT, we only have positive frequencies: DC to Nyquist
    // numBins = N/2 + 1 for full FFT size N
    // Frequency resolution = sampleRate / N
    std::vector<double> frequencies(numBins);
    double df = sampleRate / static_cast<double>(fullFftSize);

    for (size_t i = 0; i < numBins; ++i) {
        frequencies[i] = i * df;  // 0, df, 2*df, ..., Nyquist
    }

    return frequencies;
}

std::vector<std::complex<double>> computeEqResponseForFft(size_t filterFftSize, size_t fullFftSize,
                                                          double outputSampleRate,
                                                          const EqProfile& profile) {
    // Generate frequencies for R2C FFT bins at OUTPUT sample rate
    // filterFftSize = N/2+1 (R2C output bins)
    // fullFftSize = N (for frequency resolution)
    // outputSampleRate = input_rate * upsample_ratio (e.g., 705600 Hz)
    auto frequencies = generateR2cFftFrequencies(filterFftSize, fullFftSize, outputSampleRate);

    // Compute EQ response at OUTPUT sample rate
    // The biquad coefficients are computed for the upsampled rate
    // so that EQ frequencies remain correct (1kHz stays at 1kHz)
    return computeEqFrequencyResponse(frequencies, profile, outputSampleRate);
}

std::vector<double> computeEqMagnitudeForFft(size_t filterFftSize, size_t fullFftSize,
                                             double outputSampleRate, const EqProfile& profile) {
    // Compute full complex response first
    auto complexResponse =
        computeEqResponseForFft(filterFftSize, fullFftSize, outputSampleRate, profile);

    // Extract magnitude only (discard phase)
    std::vector<double> magnitude(complexResponse.size());
    for (size_t i = 0; i < complexResponse.size(); ++i) {
        magnitude[i] = std::abs(complexResponse[i]);
    }

    return magnitude;
}

}  // namespace EQ
