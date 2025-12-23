#include "vulkan/vulkan_streaming_upsampler.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

using vulkan_backend::VulkanStreamingUpsampler;

namespace {

bool hasVulkanDevice() {
    VkInstance instance = VK_NULL_HANDLE;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "vulkan_eq_test";
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        return false;
    }

    uint32_t deviceCount = 0;
    VkResult res = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    vkDestroyInstance(instance, nullptr);
    return res == VK_SUCCESS && deviceCount > 0;
}

std::string writeImpulseFile(std::size_t taps) {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 gen(static_cast<uint32_t>(now));
    std::uniform_int_distribution<int> dis(0, 999999);

    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("vk_eq_filter_" + std::to_string(now) + "_" + std::to_string(dis(gen)) + ".bin");

    std::vector<float> impulse(taps, 0.0f);
    impulse[0] = 1.0f;  // simple minimum-phase compatible impulse

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        return {};
    }
    ofs.write(reinterpret_cast<const char*>(impulse.data()),
              static_cast<std::streamsize>(impulse.size() * sizeof(float)));
    return path.string();
}

}  // namespace

TEST(VulkanStreamingEqTest, ApplyEqMagnitudeUpdatesSpectrum) {
    if (!hasVulkanDevice()) {
        GTEST_SKIP() << "No Vulkan device available";
    }

    // Use a small impulse filter to keep FFT size modest
    const std::string coeffPath = writeImpulseFile(256);
    ASSERT_FALSE(coeffPath.empty());

    VulkanStreamingUpsampler upsampler;
    VulkanStreamingUpsampler::InitParams params{};
    params.filterPathMinimum = coeffPath;
    params.filterPathLinear = coeffPath;  // test uses the same impulse for both phases
    params.initialPhase = PhaseType::Minimum;
    params.upsampleRatio = 2;
    params.blockSize = 64;
    params.inputRate = 48000;
    params.fftSizeOverride = 0;  // auto

    ASSERT_TRUE(upsampler.initialize(params));
    ASSERT_TRUE(upsampler.initializeStreaming());

    const size_t fftBins = upsampler.getFilterFftSize();
    ASSERT_GT(fftBins, 0u);

    std::vector<double> eqMag(fftBins, 0.5);  // -6 dB across band
    EXPECT_TRUE(upsampler.applyEqMagnitude(eqMag));
}

TEST(VulkanStreamingEqTest, SwitchPhaseTypeWorksAndEqReapplySucceeds) {
    if (!hasVulkanDevice()) {
        GTEST_SKIP() << "No Vulkan device available";
    }

    const std::string coeffPath = writeImpulseFile(256);
    ASSERT_FALSE(coeffPath.empty());

    VulkanStreamingUpsampler upsampler;
    VulkanStreamingUpsampler::InitParams params{};
    params.filterPathMinimum = coeffPath;
    params.filterPathLinear = coeffPath;
    params.initialPhase = PhaseType::Minimum;
    params.upsampleRatio = 2;
    params.blockSize = 64;
    params.inputRate = 48000;
    params.fftSizeOverride = 0;

    ASSERT_TRUE(upsampler.initialize(params));
    ASSERT_TRUE(upsampler.initializeStreaming());

    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);
    EXPECT_TRUE(upsampler.switchPhaseType(PhaseType::Linear));
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Linear);

    const size_t fftBins = upsampler.getFilterFftSize();
    ASSERT_GT(fftBins, 0u);
    std::vector<double> eqMag(fftBins, 1.0);
    EXPECT_TRUE(upsampler.applyEqMagnitude(eqMag));

    EXPECT_TRUE(upsampler.switchPhaseType(PhaseType::Minimum));
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);
    EXPECT_TRUE(upsampler.applyEqMagnitude(eqMag));
}

TEST(VulkanStreamingEqTest, MultiRateAndPhaseSwitchWorks) {
    if (!hasVulkanDevice()) {
        GTEST_SKIP() << "No Vulkan device available";
    }

    const std::string coeff44 = writeImpulseFile(256);
    const std::string coeff48 = writeImpulseFile(256);
    ASSERT_FALSE(coeff44.empty());
    ASSERT_FALSE(coeff48.empty());

    VulkanStreamingUpsampler upsampler;
    VulkanStreamingUpsampler::InitParams params{};
    params.filterPathMinimum44k = coeff44;
    params.filterPathMinimum48k = coeff48;
    params.filterPathLinear44k = coeff44;
    params.filterPathLinear48k = coeff48;
    params.initialPhase = PhaseType::Minimum;
    params.upsampleRatio = 2;
    params.blockSize = 64;
    params.inputRate = 44100;

    ASSERT_TRUE(upsampler.initialize(params));
    ASSERT_TRUE(upsampler.initializeStreaming());

    EXPECT_TRUE(upsampler.isMultiRateEnabled());
    EXPECT_EQ(upsampler.getCurrentInputRate(), 44100);

    // Switch to 48k family and linear phase
    EXPECT_TRUE(upsampler.switchToInputRate(48000));
    EXPECT_EQ(upsampler.getCurrentInputRate(), 48000);
    EXPECT_EQ(upsampler.getOutputSampleRate(), 96000);
    EXPECT_TRUE(upsampler.switchPhaseType(PhaseType::Linear));
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Linear);

    // Switch back to minimum phase and original family
    EXPECT_TRUE(upsampler.switchPhaseType(PhaseType::Minimum));
    EXPECT_TRUE(upsampler.switchToInputRate(44100));
    EXPECT_EQ(upsampler.getPhaseType(), PhaseType::Minimum);
    EXPECT_EQ(upsampler.getCurrentInputRate(), 44100);
}
