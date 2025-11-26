#include <gtest/gtest.h>

#include "phase_alignment.h"

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
    delay.configure(1.0f);  // integer shift for determinism
    delay.reset();

    std::vector<float> input = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> output;
    delay.process(input, output);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_NEAR(output[1], 1.0f, 1e-3f);
    EXPECT_LT(std::fabs(output[0]), 1e-3f);
}

