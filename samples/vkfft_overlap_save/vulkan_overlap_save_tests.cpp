#include "vulkan_overlap_save.h"

#include <gtest/gtest.h>
#include <vector>

namespace {

bool runImpulseCase(std::vector<float>& out) {
    const std::vector<float> filter = {1.f, 2.f, 3.f, 2.f, 1.f};
    const std::vector<float> input = {1.f, 0.f, 0.f, 0.f};
    const uint32_t ratio = 2;
    const uint32_t fftSize = 64;
    return processOverlapSaveBuffer(input, filter, ratio, fftSize, /*chunkFrames=*/4, out);
}

bool runStereoImpulseCase(std::vector<float>& out) {
    const std::vector<float> filter = {1.f, 2.f, 3.f, 2.f, 1.f};
    const std::vector<float> input = {1.f, 2.f,  //
                                      0.f, 0.f,  //
                                      0.f, 0.f,  //
                                      0.f, 0.f};
    const uint32_t ratio = 2;
    const uint32_t fftSize = 64;
    return processOverlapSaveStereoBuffer(input, filter, ratio, fftSize, /*chunkFrames=*/4, out);
}

}  // namespace

TEST(VulkanOverlapSave, ImpulseMatchesFilter) {
    std::vector<float> output;
    if (!runImpulseCase(output)) {
        GTEST_SKIP() << "Vulkan backend not available";
    }
    ASSERT_GE(output.size(), 5u);
    const std::vector<float> expected = {1.f, 2.f, 3.f, 2.f, 1.f};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-4);
    }
}

TEST(VulkanOverlapSave, StereoChannelsIndependent) {
    std::vector<float> output;
    if (!runStereoImpulseCase(output)) {
        GTEST_SKIP() << "Vulkan backend not available";
    }
    ASSERT_EQ(output.size() % 2, 0u);
    const std::size_t frames = output.size() / 2;
    ASSERT_GE(frames, 5u);

    const std::vector<float> expectedLeft = {1.f, 2.f, 3.f, 2.f, 1.f};
    const std::vector<float> expectedRight = {2.f, 4.f, 6.f, 4.f, 2.f};
    for (std::size_t i = 0; i < expectedLeft.size(); ++i) {
        EXPECT_NEAR(output[i * 2], expectedLeft[i], 1e-4);
        EXPECT_NEAR(output[i * 2 + 1], expectedRight[i], 1e-4);
    }
}
