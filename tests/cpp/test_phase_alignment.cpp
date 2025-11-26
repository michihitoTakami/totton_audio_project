#include "phase_alignment.h"

#include <cmath>
#include <gtest/gtest.h>

using namespace PhaseAlignment;

TEST(PhaseAlignmentTest, EnergyCentroidImpulse) {
    std::vector<float> impulse(8, 0.0f);
    impulse[5] = 2.0f;
    EXPECT_NEAR(computeEnergyCentroid(impulse), 5.0f, 1e-6f);
}

TEST(PhaseAlignmentTest, FractionalDelayBypassesZeroDelay) {
    FractionalDelayLine delay;
    std::vector<float> input = {1.0f, 0.0f, 0.0f};
    std::vector<float> output;
    delay.process(input, output);
    ASSERT_EQ(output.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]);
    }
}

TEST(PhaseAlignmentTest, FractionalDelayShiftsSamples) {
    FractionalDelayLine delay;
    // Default kernelRadius=12 gives base group delay of ~12 samples
    // delaySamples=1.0 reduces delay by 1 sample (advances the signal)
    // Total delay: ~11 samples
    delay.configure(1.0f);  // Advance signal by 1 sample
    delay.reset();

    // Input must be long enough to capture the delayed impulse
    // With kernelRadius=12 and delaySamples=1, peak appears around sample 11
    std::vector<float> input(32, 0.0f);
    input[0] = 1.0f;  // Impulse at t=0
    std::vector<float> output;
    delay.process(input, output);

    ASSERT_EQ(output.size(), input.size());

    // Find peak position
    int peakIdx = 0;
    float peakVal = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i]) > peakVal) {
            peakVal = std::fabs(output[i]);
            peakIdx = static_cast<int>(i);
        }
    }

    // Peak should be around kernelRadius - delaySamples = 12 - 1 = 11
    // (delaySamples advances the signal, reducing effective delay)
    EXPECT_NEAR(peakIdx, 11, 1) << "Peak should be delayed by ~11 samples";
    EXPECT_GT(peakVal, 0.5f) << "Peak should have significant amplitude";
    EXPECT_LT(std::fabs(output[0]), 0.1f) << "Output at t=0 should be small";
}
