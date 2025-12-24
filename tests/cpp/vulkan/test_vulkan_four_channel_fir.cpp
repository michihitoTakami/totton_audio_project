#include "convolution_engine.h"
#include "vulkan/vulkan_four_channel_fir.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

using ConvolutionEngine::HeadSize;
using ConvolutionEngine::headSizeToString;
using ConvolutionEngine::RateFamily;
using ConvolutionEngine::StreamFloatVector;
using vulkan_backend::VulkanFourChannelFIR;

namespace {

bool hasVulkanDevice() {
    VkInstance instance = VK_NULL_HANDLE;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "vk_four_channel_fir_test";
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

std::vector<float> makeChannelMajorTaps(int taps, float scaleBase) {
    std::vector<float> data(static_cast<size_t>(taps) * 4, 0.0f);
    for (int c = 0; c < 4; ++c) {
        for (int t = 0; t < taps; ++t) {
            data[static_cast<size_t>(c) * taps + t] =
                scaleBase * static_cast<float>(c + 1) + 0.001f * static_cast<float>(t);
        }
    }
    return data;
}

void writeHrtfFiles(const std::filesystem::path& dir, const std::string& headSize,
                    const std::string& rateLabel, int taps, float scaleBase) {
    std::filesystem::create_directories(dir);
    std::filesystem::path binPath = dir / ("hrtf_" + headSize + "_" + rateLabel + ".bin");
    std::filesystem::path jsonPath = dir / ("hrtf_" + headSize + "_" + rateLabel + ".json");

    const auto coeffs = makeChannelMajorTaps(taps, scaleBase);
    std::ofstream bin(binPath, std::ios::binary);
    ASSERT_TRUE(bin.good());
    bin.write(reinterpret_cast<const char*>(coeffs.data()),
              static_cast<std::streamsize>(coeffs.size() * sizeof(float)));
    bin.close();

    std::ofstream json(jsonPath);
    ASSERT_TRUE(json.good());
    json << R"({"n_channels":4, "n_taps":)" << taps << R"(, "storage_format":"channel_major_v1"})";
    json.close();
}

}  // namespace

TEST(VulkanFourChannelFirTest, MatchesCudaOutputForBasicBlock) {
    if (!hasVulkanDevice()) {
        GTEST_SKIP() << "No Vulkan device available";
    }

    const int taps = 64;
    const int blockSize = 128;
    std::filesystem::path tmpDir =
        std::filesystem::temp_directory_path() /
        ("vk_four_channel_fir_" + std::to_string(std::random_device{}()));
    writeHrtfFiles(tmpDir, headSizeToString(HeadSize::M), "44k", taps, 0.25f);
    writeHrtfFiles(tmpDir, headSizeToString(HeadSize::M), "48k", taps, 0.5f);

    ConvolutionEngine::FourChannelFIR cudaFir;
    ASSERT_TRUE(cudaFir.initialize(tmpDir.string(), blockSize, HeadSize::M, RateFamily::RATE_44K));
    ASSERT_TRUE(cudaFir.initializeStreaming());

    VulkanFourChannelFIR vkFir;
    ASSERT_TRUE(vkFir.initialize(tmpDir.string(), blockSize, HeadSize::M, RateFamily::RATE_44K));
    ASSERT_TRUE(vkFir.initializeStreaming());

    const size_t streamBlock = cudaFir.getStreamValidInputPerBlock();
    ASSERT_EQ(streamBlock, vkFir.getStreamValidInputPerBlock());

    std::vector<float> inputL(streamBlock, 0.0f);
    std::vector<float> inputR(streamBlock, 0.0f);
    for (size_t i = 0; i < streamBlock; ++i) {
        inputL[i] = 0.01f * static_cast<float>(i % 13);
        inputR[i] = 0.02f * static_cast<float>(i % 17);
    }

    StreamFloatVector cudaStreamL(streamBlock * 2, 0.0f);
    StreamFloatVector cudaStreamR(streamBlock * 2, 0.0f);
    StreamFloatVector vkStreamL(streamBlock * 2, 0.0f);
    StreamFloatVector vkStreamR(streamBlock * 2, 0.0f);
    StreamFloatVector cudaOutL;
    StreamFloatVector cudaOutR;
    StreamFloatVector vkOutL;
    StreamFloatVector vkOutR;
    size_t cudaAccumL = 0;
    size_t cudaAccumR = 0;
    size_t vkAccumL = 0;
    size_t vkAccumR = 0;

    ASSERT_TRUE(cudaFir.processStreamBlock(inputL.data(), inputR.data(), streamBlock, cudaOutL,
                                           cudaOutR, nullptr, cudaStreamL, cudaStreamR, cudaAccumL,
                                           cudaAccumR));
    ASSERT_TRUE(vkFir.processStreamBlock(inputL.data(), inputR.data(), streamBlock, vkOutL, vkOutR,
                                         nullptr, vkStreamL, vkStreamR, vkAccumL, vkAccumR));

    ASSERT_EQ(cudaOutL.size(), vkOutL.size());
    ASSERT_EQ(cudaOutR.size(), vkOutR.size());
    for (size_t i = 0; i < cudaOutL.size(); ++i) {
        EXPECT_NEAR(cudaOutL[i], vkOutL[i], 1e-3f) << "Mismatch at sample " << i << " (L)";
        EXPECT_NEAR(cudaOutR[i], vkOutR[i], 1e-3f) << "Mismatch at sample " << i << " (R)";
    }

    ASSERT_TRUE(cudaFir.switchRateFamily(RateFamily::RATE_48K));
    ASSERT_TRUE(vkFir.switchRateFamily(RateFamily::RATE_48K));
    cudaFir.resetStreaming();
    vkFir.resetStreaming();
    cudaAccumL = cudaAccumR = vkAccumL = vkAccumR = 0;

    ASSERT_TRUE(cudaFir.processStreamBlock(inputL.data(), inputR.data(), streamBlock, cudaOutL,
                                           cudaOutR, nullptr, cudaStreamL, cudaStreamR, cudaAccumL,
                                           cudaAccumR));
    ASSERT_TRUE(vkFir.processStreamBlock(inputL.data(), inputR.data(), streamBlock, vkOutL, vkOutR,
                                         nullptr, vkStreamL, vkStreamR, vkAccumL, vkAccumR));

    ASSERT_EQ(cudaOutL.size(), vkOutL.size());
    ASSERT_EQ(cudaOutR.size(), vkOutR.size());
    for (size_t i = 0; i < cudaOutL.size(); ++i) {
        EXPECT_NEAR(cudaOutL[i], vkOutL[i], 1e-3f) << "Mismatch at sample " << i << " (L,48k)";
        EXPECT_NEAR(cudaOutR[i], vkOutR[i], 1e-3f) << "Mismatch at sample " << i << " (R,48k)";
    }

    std::filesystem::remove_all(tmpDir);
}
