#include "convolution_engine.h"
#include "vulkan/vulkan_streaming_upsampler.h"

#include <chrono>
#include <cmath>
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
    appInfo.pApplicationName = "vulkan_cuda_parity_test";
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
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 gen(static_cast<uint32_t>(stamp));
    std::uniform_int_distribution<int> dist(0, 999999);

    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("vk_cuda_parity_" + std::to_string(stamp) + "_" + std::to_string(dist(gen)) + ".bin");

    std::vector<float> impulse(taps, 0.0f);
    impulse[0] = 1.0f;

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        return {};
    }
    ofs.write(reinterpret_cast<const char*>(impulse.data()),
              static_cast<std::streamsize>(impulse.size() * sizeof(float)));
    return path.string();
}

float computeRmsAndMaxDiff(const ConvolutionEngine::StreamFloatVector& a,
                           const ConvolutionEngine::StreamFloatVector& b, float& maxAbs) {
    double sumSq = 0.0;
    maxAbs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = a[i] - b[i];
        sumSq += static_cast<double>(diff) * static_cast<double>(diff);
        maxAbs = std::max(maxAbs, std::abs(diff));
    }
    return static_cast<float>(std::sqrt(sumSq / static_cast<double>(a.size())));
}

}  // namespace

#if defined(HAVE_CUDA_BACKEND)
TEST(VulkanCudaParityTest, NullResidualIsSmall) {
    if (!hasVulkanDevice()) {
        GTEST_SKIP() << "No Vulkan device available";
    }

    const std::string coeffPath = writeImpulseFile(256);
    ASSERT_FALSE(coeffPath.empty());

    constexpr int kUpsampleRatio = 2;
    constexpr int kBlockSize = 64;

    ConvolutionEngine::GPUUpsampler cudaUpsampler;
    ASSERT_TRUE(cudaUpsampler.initialize(coeffPath, kUpsampleRatio, kBlockSize));
    ASSERT_TRUE(cudaUpsampler.initializeStreaming());
    const size_t cudaStreamBlock = cudaUpsampler.getStreamValidInputPerBlock();
    ASSERT_GT(cudaStreamBlock, 0u);

    VulkanStreamingUpsampler vkUpsampler;
    VulkanStreamingUpsampler::InitParams params{};
    params.filterPathMinimum = coeffPath;
    params.filterPathLinear = coeffPath;
    params.initialPhase = PhaseType::Minimum;
    params.upsampleRatio = static_cast<uint32_t>(kUpsampleRatio);
    params.blockSize = static_cast<uint32_t>(kBlockSize);
    params.inputRate = 48000;
    params.fftSizeOverride = 0;

    ASSERT_TRUE(vkUpsampler.initialize(params));
    ASSERT_TRUE(vkUpsampler.initializeStreaming());
    const size_t vkStreamBlock = vkUpsampler.getStreamValidInputPerBlock();
    ASSERT_GT(vkStreamBlock, 0u);
    ASSERT_EQ(cudaStreamBlock, vkStreamBlock);

    std::vector<float> input(cudaStreamBlock, 0.0f);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : input) {
        v = dist(gen);
    }

    ConvolutionEngine::StreamFloatVector cudaStreamBuf(cudaStreamBlock * 2, 0.0f);
    ConvolutionEngine::StreamFloatVector cudaOutput;
    size_t cudaAccum = 0;
    ASSERT_TRUE(cudaUpsampler.processStreamBlock(input.data(), input.size(), cudaOutput,
                                                 /*stream=*/nullptr, cudaStreamBuf, cudaAccum));
    ASSERT_EQ(cudaOutput.size(), cudaUpsampler.getStreamValidInputPerBlock() * kUpsampleRatio);

    ConvolutionEngine::StreamFloatVector vkStreamBuf(vkStreamBlock * 2, 0.0f);
    ConvolutionEngine::StreamFloatVector vkOutput;
    size_t vkAccum = 0;
    ASSERT_TRUE(vkUpsampler.processStreamBlock(input.data(), input.size(), vkOutput, nullptr,
                                               vkStreamBuf, vkAccum));
    ASSERT_EQ(vkOutput.size(), vkUpsampler.getStreamValidInputPerBlock() * kUpsampleRatio);

    ASSERT_EQ(cudaOutput.size(), vkOutput.size());
    float maxAbs = 0.0f;
    const float rms = computeRmsAndMaxDiff(cudaOutput, vkOutput, maxAbs);

    constexpr float kMaxAbsTolerance = 5e-3f;
    constexpr float kRmsTolerance = 1e-3f;
    EXPECT_LT(maxAbs, kMaxAbsTolerance);
    EXPECT_LT(rms, kRmsTolerance);

    std::filesystem::remove(coeffPath);
}
#else
TEST(VulkanCudaParityTest, SkippedWhenCudaBackendDisabled) {
    GTEST_SKIP() << "CUDA backend is disabled; parity test requires both CUDA and Vulkan.";
}
#endif
