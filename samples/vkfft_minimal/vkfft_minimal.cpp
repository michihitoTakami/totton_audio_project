#include "utils_VkFFT.h"
#include "vkFFT.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

VkFFTResult setupVulkan(VkGPU& gpu) {
    gpu.device_id = 0;
    gpu.enableValidationLayers = 0;

    VkResult res = createInstance(&gpu, 0);
    if (res != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE;
    }

    res = findPhysicalDevice(&gpu);
    if (res != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE;
    }

    vkGetPhysicalDeviceProperties(gpu.physicalDevice, &gpu.physicalDeviceProperties);
    vkGetPhysicalDeviceMemoryProperties(gpu.physicalDevice, &gpu.physicalDeviceMemoryProperties);

    res = createDevice(&gpu, 0);
    if (res != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_CREATE_DEVICE;
    }

    res = createFence(&gpu);
    if (res != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_CREATE_FENCE;
    }

    res = createCommandPool(&gpu);
    if (res != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL;
    }

    return VKFFT_SUCCESS;
}

void cleanupVulkan(VkGPU& gpu, VkBuffer buffer, VkDeviceMemory memory) {
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(gpu.device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(gpu.device, memory, nullptr);
    }
    if (gpu.commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(gpu.device, gpu.commandPool, nullptr);
    }
    if (gpu.fence != VK_NULL_HANDLE) {
        vkDestroyFence(gpu.device, gpu.fence, nullptr);
    }
    if (gpu.device != VK_NULL_HANDLE) {
        vkDestroyDevice(gpu.device, nullptr);
    }
    if (gpu.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(gpu.instance, nullptr);
    }
}

VkFFTResult runR2CSample() {
    constexpr uint64_t kFftSize = 1024;
    constexpr double kTolerance = 1e-3;
    constexpr double twoPi = 6.28318530717958647692;

    VkGPU gpu = {};
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFFTApplication app = {};
    bool appInitialized = false;

    VkFFTResult resFFT = setupVulkan(gpu);
    if (resFFT != VKFFT_SUCCESS) {
        cleanupVulkan(gpu, buffer, memory);
        return resFFT;
    }

    const uint64_t complexSamples = 2 * (kFftSize / 2 + 1);
    uint64_t bufferSize = sizeof(float) * complexSamples;

    std::vector<float> hostBuffer(complexSamples, 0.0f);
    for (uint64_t i = 0; i < kFftSize; ++i) {
        hostBuffer[i] = static_cast<float>(std::sin(twoPi * static_cast<double>(i) / kFftSize));
    }

    resFFT = allocateBuffer(&gpu, &buffer, &memory,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, bufferSize);
    if (resFFT != VKFFT_SUCCESS) {
        cleanupVulkan(gpu, buffer, memory);
        return resFFT;
    }

    resFFT = transferDataFromCPU(&gpu, hostBuffer.data(), &buffer, bufferSize);
    if (resFFT != VKFFT_SUCCESS) {
        cleanupVulkan(gpu, buffer, memory);
        return resFFT;
    }

    VkFFTConfiguration configuration = {};
    configuration.FFTdim = 1;
    configuration.size[0] = kFftSize;
    configuration.size[1] = 1;
    configuration.size[2] = 1;
    configuration.performR2C = 1;
    configuration.normalize = 1;
    configuration.buffer = &buffer;
    configuration.bufferSize = &bufferSize;
    configuration.device = &gpu.device;
    configuration.queue = &gpu.queue;
    configuration.commandPool = &gpu.commandPool;
    configuration.fence = &gpu.fence;
    configuration.physicalDevice = &gpu.physicalDevice;

    resFFT = initializeVkFFT(&app, configuration);
    if (resFFT != VKFFT_SUCCESS) {
        cleanupVulkan(gpu, buffer, memory);
        return resFFT;
    }
    appInitialized = true;

    VkFFTLaunchParams launchParams = {};
    double elapsedMs = 0.0;
    resFFT = performVulkanFFTiFFT(&gpu, &app, &launchParams, 1, &elapsedMs);
    if (resFFT != VKFFT_SUCCESS) {
        if (appInitialized) {
            deleteVkFFT(&app);
        }
        cleanupVulkan(gpu, buffer, memory);
        return resFFT;
    }

    std::vector<float> output(hostBuffer.size(), 0.0f);
    resFFT = transferDataToCPU(&gpu, output.data(), &buffer, bufferSize);
    if (resFFT != VKFFT_SUCCESS) {
        if (appInitialized) {
            deleteVkFFT(&app);
        }
        cleanupVulkan(gpu, buffer, memory);
        return resFFT;
    }

    double maxAbsError = 0.0;
    double mse = 0.0;
    for (uint64_t i = 0; i < kFftSize; ++i) {
        double diff = static_cast<double>(output[i]) - static_cast<double>(hostBuffer[i]);
        maxAbsError = std::max(maxAbsError, std::abs(diff));
        mse += diff * diff;
    }
    mse /= static_cast<double>(kFftSize);
    const double rmse = std::sqrt(mse);

    std::cout << "VkFFT 1D R2C/C2R sample\n";
    std::cout << "  device : " << gpu.physicalDeviceProperties.deviceName << "\n";
    std::cout << "  size   : " << kFftSize << "\n";
    std::cout << "  time   : " << elapsedMs << " ms (forward+inverse)\n";
    std::cout << "  maxerr : " << maxAbsError << "\n";
    std::cout << "  rmse   : " << rmse << std::endl;

    if (appInitialized) {
        deleteVkFFT(&app);
    }
    cleanupVulkan(gpu, buffer, memory);

    if (maxAbsError > kTolerance) {
        std::cerr << "Validation failed: max error exceeds tolerance (" << maxAbsError << " > "
                  << kTolerance << ")" << std::endl;
        return VKFFT_ERROR_MATH_FAILED;
    }

    return VKFFT_SUCCESS;
}

}  // namespace

int main() {
    const VkFFTResult result = runR2CSample();
    if (result != VKFFT_SUCCESS) {
        std::cerr << "VkFFT sample failed with code: " << result << std::endl;
        return 1;
    }
    return 0;
}
