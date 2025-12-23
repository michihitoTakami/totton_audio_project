#include "gpu/backend/gpu_backend.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>
#include <vulkan/vulkan.h>

using ConvolutionEngine::GpuBackend::BackendKind;
using ConvolutionEngine::GpuBackend::CopyKind;
using ConvolutionEngine::GpuBackend::createVulkanBackend;
using ConvolutionEngine::GpuBackend::DeviceBuffer;
using ConvolutionEngine::GpuBackend::FftDirection;
using ConvolutionEngine::GpuBackend::FftDomain;
using ConvolutionEngine::GpuBackend::FftPlan;
using ConvolutionEngine::GpuBackend::IGpuBackend;

namespace {

bool hasVulkanDevice() {
    VkInstance instance = VK_NULL_HANDLE;
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "vk_backend_test";
    appInfo.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) {
        return false;
    }
    uint32_t deviceCount = 0;
    VkResult res = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    vkDestroyInstance(instance, nullptr);
    return res == VK_SUCCESS && deviceCount > 0;
}

}  // namespace

TEST(VulkanBackendTest, R2CAndC2RRoundTrip) {
    if (!hasVulkanDevice()) {
        GTEST_SKIP() << "No Vulkan device available";
    }

    auto backend = createVulkanBackend();
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->kind(), BackendKind::Vulkan);

    constexpr int kFftSize = 1024;
    constexpr size_t kTimeBytes = static_cast<size_t>(kFftSize) * sizeof(float);
    constexpr size_t kFreqBytes = static_cast<size_t>(kFftSize / 2 + 1) * 2 * sizeof(float);

    DeviceBuffer timeBuf{};
    DeviceBuffer freqBuf{};
    ASSERT_EQ(backend->allocateDeviceBuffer(timeBuf, kTimeBytes, "time"),
              AudioEngine::ErrorCode::OK);
    ASSERT_EQ(backend->allocateDeviceBuffer(freqBuf, kFreqBytes, "freq"),
              AudioEngine::ErrorCode::OK);

    std::vector<float> input(kFftSize, 0.0f);
    input[0] = 1.0f;
    input[1] = -0.5f;
    ASSERT_EQ(backend->copy(timeBuf.handle.ptr, input.data(), kTimeBytes, CopyKind::HostToDevice,
                            nullptr, "h2d"),
              AudioEngine::ErrorCode::OK);

    FftPlan plan{};
    ASSERT_EQ(backend->createFftPlan1d(plan, kFftSize, 1, FftDomain::RealToComplex, "plan"),
              AudioEngine::ErrorCode::OK);

    ASSERT_EQ(backend->executeFft(plan, timeBuf, freqBuf, FftDirection::Forward, nullptr, "fwd"),
              AudioEngine::ErrorCode::OK);
    ASSERT_EQ(backend->executeFft(plan, freqBuf, timeBuf, FftDirection::Inverse, nullptr, "inv"),
              AudioEngine::ErrorCode::OK);

    std::vector<float> output(kFftSize, 0.0f);
    ASSERT_EQ(backend->copy(output.data(), timeBuf.handle.ptr, kTimeBytes, CopyKind::DeviceToHost,
                            nullptr, "d2h"),
              AudioEngine::ErrorCode::OK);

    // VkFFT normalize=1 なので元信号と一致するはず
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(output[i], input[i], 1e-3f);
    }

    backend->destroyFftPlan(plan, "destroy");
    backend->freeDeviceBuffer(freqBuf, "free freq");
    backend->freeDeviceBuffer(timeBuf, "free time");
}

TEST(VulkanBackendTest, ComplexMulScale) {
    if (!hasVulkanDevice()) {
        GTEST_SKIP() << "No Vulkan device available";
    }
    auto backend = createVulkanBackend();
    ASSERT_NE(backend, nullptr);

    constexpr size_t kComplex = 8;
    constexpr size_t kBytes = kComplex * 2 * sizeof(float);
    DeviceBuffer out{};
    DeviceBuffer a{};
    DeviceBuffer b{};
    ASSERT_EQ(backend->allocateDeviceBuffer(out, kBytes, "out"), AudioEngine::ErrorCode::OK);
    ASSERT_EQ(backend->allocateDeviceBuffer(a, kBytes, "a"), AudioEngine::ErrorCode::OK);
    ASSERT_EQ(backend->allocateDeviceBuffer(b, kBytes, "b"), AudioEngine::ErrorCode::OK);

    std::vector<float> hostA(kComplex * 2, 0.0f);
    std::vector<float> hostB(kComplex * 2, 0.0f);
    for (size_t i = 0; i < kComplex; ++i) {
        hostA[i * 2] = 1.0f;
        hostA[i * 2 + 1] = 0.5f;
        hostB[i * 2] = 0.25f;
        hostB[i * 2 + 1] = -0.75f;
    }
    ASSERT_EQ(
        backend->copy(a.handle.ptr, hostA.data(), kBytes, CopyKind::HostToDevice, nullptr, "h2d a"),
        AudioEngine::ErrorCode::OK);
    ASSERT_EQ(
        backend->copy(b.handle.ptr, hostB.data(), kBytes, CopyKind::HostToDevice, nullptr, "h2d b"),
        AudioEngine::ErrorCode::OK);

    ASSERT_EQ(backend->complexPointwiseMulScale(out, a, b, kComplex, 2.0f, nullptr, "mul"),
              AudioEngine::ErrorCode::OK);

    std::vector<float> hostOut(kComplex * 2, 0.0f);
    ASSERT_EQ(backend->copy(hostOut.data(), out.handle.ptr, kBytes, CopyKind::DeviceToHost, nullptr,
                            "d2h out"),
              AudioEngine::ErrorCode::OK);

    for (size_t i = 0; i < kComplex; ++i) {
        float ar = hostA[i * 2];
        float ai = hostA[i * 2 + 1];
        float br = hostB[i * 2];
        float bi = hostB[i * 2 + 1];
        float expectedRe = (ar * br - ai * bi) * 2.0f;
        float expectedIm = (ar * bi + ai * br) * 2.0f;
        EXPECT_NEAR(hostOut[i * 2], expectedRe, 1e-4f);
        EXPECT_NEAR(hostOut[i * 2 + 1], expectedIm, 1e-4f);
    }

    backend->freeDeviceBuffer(out, "free out");
    backend->freeDeviceBuffer(a, "free a");
    backend->freeDeviceBuffer(b, "free b");
}
