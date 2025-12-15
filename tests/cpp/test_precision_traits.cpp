// Precision conversion utilities (Issue #782)
#include "gpu/precision_traits.h"

#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

using namespace ConvolutionEngine;

TEST(PrecisionTraitsTest, ConvertsCoefficientsToActivePrecision) {
    std::vector<float> source = {0.25f, -0.5f, 1.25f};
    auto converted = convertCoefficientsToPrecision<ActivePrecisionTraits>(source);

    ASSERT_EQ(converted.size(), source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(converted[i]), static_cast<double>(source[i]));
    }

    if constexpr (ActivePrecisionTraits::kIsDouble) {
        EXPECT_EQ(sizeof(converted[0]), sizeof(double));
    } else {
        EXPECT_EQ(sizeof(converted[0]), sizeof(float));
    }
}
TEST(PrecisionTraitsTest, FloatTraitsConstants) {
    using Traits = PrecisionTraits<float>;
    static_assert(!Traits::kIsDouble, "float traits should not be double");
    EXPECT_EQ(Traits::kFftTypeForward, CUFFT_R2C);
    EXPECT_EQ(Traits::kFftTypeInverse, CUFFT_C2R);
    EXPECT_FLOAT_EQ(Traits::scaleFactor(1024), 1.0f / 1024.0f);
}

TEST(PrecisionTraitsTest, DoubleTraitsConstants) {
    using Traits = PrecisionTraits<double>;
    static_assert(Traits::kIsDouble, "double traits should be double");
    EXPECT_EQ(Traits::kFftTypeForward, CUFFT_D2Z);
    EXPECT_EQ(Traits::kFftTypeInverse, CUFFT_Z2D);
    EXPECT_DOUBLE_EQ(Traits::scaleFactor(2048), 1.0 / 2048.0);
}

TEST(PrecisionTraitsTest, ActivePrecisionSelection) {
#ifdef GPU_UPSAMPLER_USE_FLOAT64
    constexpr bool expectDouble = true;
#else
    constexpr bool expectDouble = false;
#endif
    constexpr bool actualIsDouble = ActivePrecisionTraits::kIsDouble;
    EXPECT_EQ(actualIsDouble, expectDouble);
}
