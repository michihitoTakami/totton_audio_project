/**
 * @file test_eq_to_fir.cpp
 * @brief Unit tests for EQ to FIR conversion (biquad calculations)
 */

#include "audio/eq_to_fir.h"

#include <cmath>
#include <gtest/gtest.h>

using namespace EQ;

class EqToFirTest : public ::testing::Test {
   protected:
    static constexpr double SAMPLE_RATE = 44100.0;
    static constexpr double TOLERANCE = 1e-6;

    void SetUp() override {}
};

// ============================================================
// calculateBiquadCoeffs tests
// ============================================================

TEST_F(EqToFirTest, BiquadCoeffsDisabledBandReturnsUnity) {
    EqBand band;
    band.enabled = false;
    band.gain = 6.0;  // Would normally boost

    BiquadCoeffs c = calculateBiquadCoeffs(band, SAMPLE_RATE);

    // Unity gain: b0=1, b1=b2=a1=a2=0
    EXPECT_DOUBLE_EQ(c.b0, 1.0);
    EXPECT_DOUBLE_EQ(c.b1, 0.0);
    EXPECT_DOUBLE_EQ(c.b2, 0.0);
    EXPECT_DOUBLE_EQ(c.a1, 0.0);
    EXPECT_DOUBLE_EQ(c.a2, 0.0);
}

TEST_F(EqToFirTest, BiquadCoeffsZeroGainReturnsUnity) {
    EqBand band;
    band.enabled = true;
    band.gain = 0.0;

    BiquadCoeffs c = calculateBiquadCoeffs(band, SAMPLE_RATE);

    EXPECT_DOUBLE_EQ(c.b0, 1.0);
    EXPECT_DOUBLE_EQ(c.b1, 0.0);
    EXPECT_DOUBLE_EQ(c.b2, 0.0);
}

TEST_F(EqToFirTest, BiquadCoeffsPeakingNonZero) {
    EqBand band;
    band.enabled = true;
    band.type = FilterType::PK;
    band.frequency = 1000.0;
    band.gain = 6.0;
    band.q = 1.41;

    BiquadCoeffs c = calculateBiquadCoeffs(band, SAMPLE_RATE);

    // Coefficients should be non-trivial
    EXPECT_NE(c.b0, 1.0);
    EXPECT_NE(c.b1, 0.0);
    EXPECT_NE(c.b2, 0.0);
    EXPECT_NE(c.a1, 0.0);
    EXPECT_NE(c.a2, 0.0);
}

TEST_F(EqToFirTest, BiquadCoeffsLowShelfNonZero) {
    EqBand band;
    band.enabled = true;
    band.type = FilterType::LS;
    band.frequency = 100.0;
    band.gain = 3.0;
    band.q = 0.71;

    BiquadCoeffs c = calculateBiquadCoeffs(band, SAMPLE_RATE);

    EXPECT_NE(c.b0, 1.0);
}

TEST_F(EqToFirTest, BiquadCoeffsHighShelfNonZero) {
    EqBand band;
    band.enabled = true;
    band.type = FilterType::HS;
    band.frequency = 10000.0;
    band.gain = -3.0;
    band.q = 0.71;

    BiquadCoeffs c = calculateBiquadCoeffs(band, SAMPLE_RATE);

    EXPECT_NE(c.b0, 1.0);
}

// ============================================================
// biquadFrequencyResponse tests
// ============================================================

TEST_F(EqToFirTest, FrequencyResponseUnityAtDCForUnityCoeffs) {
    BiquadCoeffs unity = {1.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> freqs = {0.0, 100.0, 1000.0, 10000.0};

    auto response = biquadFrequencyResponse(freqs, unity, SAMPLE_RATE);

    for (size_t i = 0; i < response.size(); ++i) {
        EXPECT_NEAR(std::abs(response[i]), 1.0, TOLERANCE);
    }
}

TEST_F(EqToFirTest, FrequencyResponsePeakingHasGainAtCenter) {
    EqBand band;
    band.enabled = true;
    band.type = FilterType::PK;
    band.frequency = 1000.0;
    band.gain = 6.0;
    band.q = 1.41;

    BiquadCoeffs c = calculateBiquadCoeffs(band, SAMPLE_RATE);
    std::vector<double> freqs = {1000.0};

    auto response = biquadFrequencyResponse(freqs, c, SAMPLE_RATE);
    double magnitude = std::abs(response[0]);
    double magnitudeDb = 20.0 * std::log10(magnitude);

    // At center frequency, gain should be approximately the specified gain
    EXPECT_NEAR(magnitudeDb, 6.0, 0.5);
}

TEST_F(EqToFirTest, FrequencyResponsePeakingFlatAwayFromCenter) {
    EqBand band;
    band.enabled = true;
    band.type = FilterType::PK;
    band.frequency = 1000.0;
    band.gain = 6.0;
    band.q = 1.41;

    BiquadCoeffs c = calculateBiquadCoeffs(band, SAMPLE_RATE);

    // Far from center frequency (low and high)
    std::vector<double> freqs = {100.0, 10000.0};
    auto response = biquadFrequencyResponse(freqs, c, SAMPLE_RATE);

    // Should be close to unity (0 dB)
    for (const auto& r : response) {
        double magnitudeDb = 20.0 * std::log10(std::abs(r));
        EXPECT_NEAR(magnitudeDb, 0.0, 1.0);  // Within 1 dB of flat
    }
}

// ============================================================
// generateR2cFftFrequencies tests
// ============================================================

TEST_F(EqToFirTest, R2cFftFrequenciesStartsAtZero) {
    auto freqs = generateR2cFftFrequencies(513, 1024, SAMPLE_RATE);

    EXPECT_DOUBLE_EQ(freqs[0], 0.0);
}

TEST_F(EqToFirTest, R2cFftFrequenciesEndsAtNyquist) {
    size_t fftSize = 1024;
    size_t numBins = fftSize / 2 + 1;  // 513
    auto freqs = generateR2cFftFrequencies(numBins, fftSize, SAMPLE_RATE);

    double nyquist = SAMPLE_RATE / 2.0;
    EXPECT_NEAR(freqs.back(), nyquist, 1.0);
}

TEST_F(EqToFirTest, R2cFftFrequenciesCorrectSize) {
    size_t fftSize = 1024;
    size_t numBins = fftSize / 2 + 1;
    auto freqs = generateR2cFftFrequencies(numBins, fftSize, SAMPLE_RATE);

    EXPECT_EQ(freqs.size(), numBins);
}

TEST_F(EqToFirTest, R2cFftFrequenciesEvenlySpaced) {
    size_t fftSize = 1024;
    size_t numBins = fftSize / 2 + 1;
    auto freqs = generateR2cFftFrequencies(numBins, fftSize, SAMPLE_RATE);

    double df = SAMPLE_RATE / static_cast<double>(fftSize);

    for (size_t i = 0; i < numBins; ++i) {
        EXPECT_NEAR(freqs[i], i * df, TOLERANCE);
    }
}

// ============================================================
// computeEqFrequencyResponse tests
// ============================================================

TEST_F(EqToFirTest, EqFrequencyResponseEmptyProfileIsUnity) {
    EqProfile profile;
    std::vector<double> freqs = {100.0, 1000.0, 10000.0};

    auto response = computeEqFrequencyResponse(freqs, profile, SAMPLE_RATE);

    for (const auto& r : response) {
        EXPECT_NEAR(std::abs(r), 1.0, TOLERANCE);
    }
}

TEST_F(EqToFirTest, EqFrequencyResponsePreampApplied) {
    EqProfile profile;
    profile.preampDb = -6.0;  // -6dB = 0.5 linear
    std::vector<double> freqs = {1000.0};

    auto response = computeEqFrequencyResponse(freqs, profile, SAMPLE_RATE);

    double expectedLinear = std::pow(10.0, -6.0 / 20.0);  // ~0.5
    EXPECT_NEAR(std::abs(response[0]), expectedLinear, 0.01);
}

TEST_F(EqToFirTest, EqFrequencyResponseSingleBand) {
    EqProfile profile;
    EqBand band;
    band.enabled = true;
    band.type = FilterType::PK;
    band.frequency = 1000.0;
    band.gain = 6.0;
    band.q = 1.41;
    profile.bands.push_back(band);

    std::vector<double> freqs = {1000.0};
    auto response = computeEqFrequencyResponse(freqs, profile, SAMPLE_RATE);

    double magnitudeDb = 20.0 * std::log10(std::abs(response[0]));
    EXPECT_NEAR(magnitudeDb, 6.0, 0.5);
}

TEST_F(EqToFirTest, EqFrequencyResponseDisabledBandIgnored) {
    EqProfile profile;
    EqBand band;
    band.enabled = false;
    band.type = FilterType::PK;
    band.frequency = 1000.0;
    band.gain = 6.0;
    band.q = 1.41;
    profile.bands.push_back(band);

    std::vector<double> freqs = {1000.0};
    auto response = computeEqFrequencyResponse(freqs, profile, SAMPLE_RATE);

    // Disabled band should result in unity response
    EXPECT_NEAR(std::abs(response[0]), 1.0, TOLERANCE);
}

// ============================================================
// computeEqMagnitudeForFft tests
// ============================================================

TEST_F(EqToFirTest, EqMagnitudeForFftEmptyProfileIsUnity) {
    EqProfile profile;
    size_t fftSize = 1024;
    size_t numBins = fftSize / 2 + 1;
    double outputRate = SAMPLE_RATE * 16;  // Upsampled

    auto magnitude = computeEqMagnitudeForFft(numBins, fftSize, outputRate, profile);

    EXPECT_EQ(magnitude.size(), numBins);
    for (double m : magnitude) {
        EXPECT_NEAR(m, 1.0, TOLERANCE);
    }
}

TEST_F(EqToFirTest, EqMagnitudeForFftCorrectSize) {
    EqProfile profile;
    size_t fftSize = 2048;
    size_t numBins = fftSize / 2 + 1;
    double outputRate = SAMPLE_RATE * 16;

    auto magnitude = computeEqMagnitudeForFft(numBins, fftSize, outputRate, profile);

    EXPECT_EQ(magnitude.size(), numBins);
}

TEST_F(EqToFirTest, EqMagnitudeForFftAllPositive) {
    EqProfile profile;
    profile.preampDb = -12.0;
    EqBand band;
    band.enabled = true;
    band.type = FilterType::PK;
    band.frequency = 1000.0;
    band.gain = -10.0;
    band.q = 1.0;
    profile.bands.push_back(band);

    size_t fftSize = 1024;
    size_t numBins = fftSize / 2 + 1;
    double outputRate = SAMPLE_RATE * 16;

    auto magnitude = computeEqMagnitudeForFft(numBins, fftSize, outputRate, profile);

    for (double m : magnitude) {
        EXPECT_GT(m, 0.0);
    }
}
