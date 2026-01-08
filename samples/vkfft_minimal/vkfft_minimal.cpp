#include "vkFFT.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties physicalDeviceProperties{};
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
};

uint32_t findComputeQueueFamilyIndex(VkPhysicalDevice physicalDevice) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, props.data());
    for (uint32_t i = 0; i < count; ++i) {
        if ((props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && props[i].queueCount > 0) {
            return i;
        }
    }
    return UINT32_MAX;
}

VkFFTResult createVulkanContext(VulkanContext& ctx) {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "totton_audio VkFFT minimal";
    appInfo.applicationVersion = 1;
    appInfo.pEngineName = "totton_audio";
    appInfo.engineVersion = 1;
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instanceCreateInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceCreateInfo.pApplicationInfo = &appInfo;
    VkResult vr = vkCreateInstance(&instanceCreateInfo, nullptr, &ctx.instance);
    if (vr != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE;
    }

    uint32_t deviceCount = 0;
    vr = vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr);
    if (vr != VK_SUCCESS || deviceCount == 0) {
        return VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vr = vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data());
    if (vr != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE;
    }
    ctx.physicalDevice = devices[0];
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &ctx.physicalDeviceProperties);

    const uint32_t qfi = findComputeQueueFamilyIndex(ctx.physicalDevice);
    if (qfi == UINT32_MAX) {
        return VKFFT_ERROR_FAILED_TO_CREATE_DEVICE;
    }
    ctx.queueFamilyIndex = qfi;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueCreateInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceCreateInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    vr = vkCreateDevice(ctx.physicalDevice, &deviceCreateInfo, nullptr, &ctx.device);
    if (vr != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_CREATE_DEVICE;
    }
    vkGetDeviceQueue(ctx.device, ctx.queueFamilyIndex, 0, &ctx.queue);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vr = vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.commandPool);
    if (vr != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vr = vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.fence);
    if (vr != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_CREATE_FENCE;
    }

    return VKFFT_SUCCESS;
}

void destroyVulkanContext(VulkanContext& ctx) {
    if (ctx.fence != VK_NULL_HANDLE) {
        vkDestroyFence(ctx.device, ctx.fence, nullptr);
        ctx.fence = VK_NULL_HANDLE;
    }
    if (ctx.commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
        ctx.commandPool = VK_NULL_HANDLE;
    }
    if (ctx.device != VK_NULL_HANDLE) {
        vkDestroyDevice(ctx.device, nullptr);
        ctx.device = VK_NULL_HANDLE;
    }
    if (ctx.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(ctx.instance, nullptr);
        ctx.instance = VK_NULL_HANDLE;
    }
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) &&
            ((memProps.memoryTypes[i].propertyFlags & properties) == properties)) {
            return i;
        }
    }
    return UINT32_MAX;
}

VkFFTResult createHostVisibleBuffer(VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                                    VkBuffer& outBuffer, VkDeviceMemory& outMemory) {
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult vr = vkCreateBuffer(ctx.device, &bufferInfo, nullptr, &outBuffer);
    if (vr != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_CREATE_BUFFER;
    }

    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(ctx.device, outBuffer, &memReq);
    const uint32_t memoryTypeIndex =
        findMemoryType(ctx.physicalDevice, memReq.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memoryTypeIndex == UINT32_MAX) {
        return VKFFT_ERROR_FAILED_TO_FIND_MEMORY;
    }

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    vr = vkAllocateMemory(ctx.device, &allocInfo, nullptr, &outMemory);
    if (vr != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY;
    }
    vr = vkBindBufferMemory(ctx.device, outBuffer, outMemory, 0);
    if (vr != VK_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY;
    }
    return VKFFT_SUCCESS;
}

VkFFTResult runR2CSample() {
    constexpr uint64_t kFftSize = 1024;
    constexpr double kTolerance = 1e-3;
    constexpr double twoPi = 6.28318530717958647692;

    VulkanContext ctx{};
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFFTApplication app = {};
    bool appInitialized = false;

    VkFFTResult resFFT = createVulkanContext(ctx);
    if (resFFT != VKFFT_SUCCESS) {
        destroyVulkanContext(ctx);
        return resFFT;
    }

    const uint64_t complexSamples = 2 * (kFftSize / 2 + 1);  // in-place R2C layout
    uint64_t bufferSize = sizeof(float) * complexSamples;

    std::vector<float> hostBuffer(complexSamples, 0.0f);
    for (uint64_t i = 0; i < kFftSize; ++i) {
        hostBuffer[i] = static_cast<float>(std::sin(twoPi * static_cast<double>(i) / kFftSize));
    }

    resFFT = createHostVisibleBuffer(ctx, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer,
                                     memory);
    if (resFFT != VKFFT_SUCCESS) {
        destroyVulkanContext(ctx);
        return resFFT;
    }

    void* mapped = nullptr;
    VkResult vr = vkMapMemory(ctx.device, memory, 0, bufferSize, 0, &mapped);
    if (vr != VK_SUCCESS) {
        destroyVulkanContext(ctx);
        return VKFFT_ERROR_FAILED_TO_MAP_MEMORY;
    }
    std::memcpy(mapped, hostBuffer.data(), static_cast<size_t>(bufferSize));
    vkUnmapMemory(ctx.device, memory);

    VkFFTConfiguration configuration = {};
    configuration.FFTdim = 1;
    configuration.size[0] = kFftSize;
    configuration.size[1] = 1;
    configuration.size[2] = 1;
    configuration.performR2C = 1;
    configuration.normalize = 1;
    configuration.buffer = &buffer;
    configuration.bufferSize = &bufferSize;
    configuration.device = &ctx.device;
    configuration.queue = &ctx.queue;
    configuration.commandPool = &ctx.commandPool;
    configuration.fence = &ctx.fence;
    configuration.physicalDevice = &ctx.physicalDevice;

    resFFT = initializeVkFFT(&app, configuration);
    if (resFFT != VKFFT_SUCCESS) {
        destroyVulkanContext(ctx);
        return resFFT;
    }
    appInitialized = true;

    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    vr = vkAllocateCommandBuffers(ctx.device, &allocInfo, &commandBuffer);
    if (vr != VK_SUCCESS) {
        if (appInitialized) {
            deleteVkFFT(&app);
        }
        destroyVulkanContext(ctx);
        return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
    }

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vr = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    if (vr != VK_SUCCESS) {
        if (appInitialized) {
            deleteVkFFT(&app);
        }
        destroyVulkanContext(ctx);
        return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
    }

    VkFFTLaunchParams launchParams = {};
    launchParams.commandBuffer = &commandBuffer;

    resFFT = VkFFTAppend(&app, -1, &launchParams);
    if (resFFT == VKFFT_SUCCESS) {
        resFFT = VkFFTAppend(&app, 1, &launchParams);
    }
    if (resFFT != VKFFT_SUCCESS) {
        vkEndCommandBuffer(commandBuffer);
        if (appInitialized) {
            deleteVkFFT(&app);
        }
        destroyVulkanContext(ctx);
        return resFFT;
    }

    vr = vkEndCommandBuffer(commandBuffer);
    if (vr != VK_SUCCESS) {
        if (appInitialized) {
            deleteVkFFT(&app);
        }
        destroyVulkanContext(ctx);
        return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
    }

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vr = vkQueueSubmit(ctx.queue, 1, &submitInfo, ctx.fence);
    if (vr != VK_SUCCESS) {
        if (appInitialized) {
            deleteVkFFT(&app);
        }
        destroyVulkanContext(ctx);
        return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
    }
    vr = vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
    if (vr != VK_SUCCESS) {
        if (appInitialized) {
            deleteVkFFT(&app);
        }
        destroyVulkanContext(ctx);
        return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
    }
    vkResetFences(ctx.device, 1, &ctx.fence);

    mapped = nullptr;
    vr = vkMapMemory(ctx.device, memory, 0, bufferSize, 0, &mapped);
    if (vr != VK_SUCCESS) {
        if (appInitialized) {
            deleteVkFFT(&app);
        }
        destroyVulkanContext(ctx);
        return VKFFT_ERROR_FAILED_TO_MAP_MEMORY;
    }
    std::vector<float> output(complexSamples, 0.0f);
    std::memcpy(output.data(), mapped, static_cast<size_t>(bufferSize));
    vkUnmapMemory(ctx.device, memory);

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
    std::cout << "  device : " << ctx.physicalDeviceProperties.deviceName << "\n";
    std::cout << "  size   : " << kFftSize << "\n";
    std::cout << "  maxerr : " << maxAbsError << "\n";
    std::cout << "  rmse   : " << rmse << std::endl;

    if (appInitialized) {
        deleteVkFFT(&app);
    }
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx.device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx.device, memory, nullptr);
    }
    destroyVulkanContext(ctx);

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
