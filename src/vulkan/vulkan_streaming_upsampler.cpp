#include "vulkan/vulkan_streaming_upsampler.h"

#include "logging/logger.h"
#include "vkFFT.h"

#include <SPIRV/GlslangToSpv.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

using ConvolutionEngine::detectRateFamily;
using ConvolutionEngine::getBaseSampleRate;
using ConvolutionEngine::RateFamily;

namespace {

uint32_t nextPow2(uint32_t v) {
    if (v == 0) {
        return 1;
    }
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties physicalProps{};
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
};

struct BufferResource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct MultiplyPipeline {
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
};

bool checkVk(VkResult vr, const char* context) {
    if (vr != VK_SUCCESS) {
        LOG_ERROR("[Vulkan] {} failed: {}", context, static_cast<int>(vr));
        return false;
    }
    return true;
}

uint32_t findComputeQueueFamily(VkPhysicalDevice device) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());
    for (uint32_t i = 0; i < count; ++i) {
        if ((props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && props[i].queueCount > 0) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool initContext(VulkanContext& ctx) {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "vulkan_streaming_upsampler";
    appInfo.applicationVersion = 1;
    appInfo.pEngineName = "gpu_os";
    appInfo.engineVersion = 1;
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &appInfo;
    if (!checkVk(vkCreateInstance(&instanceInfo, nullptr, &ctx.instance), "vkCreateInstance")) {
        return false;
    }

    uint32_t deviceCount = 0;
    if (!checkVk(vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr),
                 "vkEnumeratePhysicalDevices") ||
        deviceCount == 0) {
        LOG_ERROR("No Vulkan physical devices found");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data());
    ctx.physicalDevice = devices[0];
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &ctx.physicalProps);

    const uint32_t qfi = findComputeQueueFamily(ctx.physicalDevice);
    if (qfi == UINT32_MAX) {
        LOG_ERROR("No compute queue family found");
        return false;
    }
    ctx.queueFamily = qfi;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = ctx.queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    if (!checkVk(vkCreateDevice(ctx.physicalDevice, &deviceInfo, nullptr, &ctx.device),
                 "vkCreateDevice")) {
        return false;
    }
    vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = ctx.queueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (!checkVk(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.commandPool),
                 "vkCreateCommandPool")) {
        return false;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!checkVk(vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.fence), "vkCreateFence")) {
        return false;
    }
    return true;
}

void destroyContext(VulkanContext& ctx) {
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

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeBits,
                        VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool createHostBuffer(VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                      BufferResource& out) {
    out.size = size;
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!checkVk(vkCreateBuffer(ctx.device, &bufferInfo, nullptr, &out.buffer), "vkCreateBuffer")) {
        return false;
    }
    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(ctx.device, out.buffer, &memReq);
    uint32_t memoryType =
        findMemoryType(ctx.physicalDevice, memReq.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memoryType == UINT32_MAX) {
        LOG_ERROR("No suitable host-visible memory type found");
        return false;
    }
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memoryType;
    if (!checkVk(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &out.memory),
                 "vkAllocateMemory")) {
        return false;
    }
    if (!checkVk(vkBindBufferMemory(ctx.device, out.buffer, out.memory, 0), "vkBindBufferMemory")) {
        return false;
    }
    return true;
}

void destroyBuffer(VulkanContext& ctx, BufferResource& buf) {
    if (buf.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx.device, buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
    }
    if (buf.memory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx.device, buf.memory, nullptr);
        buf.memory = VK_NULL_HANDLE;
    }
}

VkFFTResult createFftApp(VulkanContext& ctx, VkBuffer buffer, uint64_t bufferSize, uint32_t fftSize,
                         VkFFTApplication& outApp) {
    VkFFTConfiguration conf{};
    conf.FFTdim = 1;
    conf.size[0] = fftSize;
    conf.performR2C = 1;
    conf.normalize = 1;
    conf.buffer = &buffer;
    conf.bufferSize = &bufferSize;
    conf.device = &ctx.device;
    conf.queue = &ctx.queue;
    conf.commandPool = &ctx.commandPool;
    conf.fence = &ctx.fence;
    conf.physicalDevice = &ctx.physicalDevice;
    return initializeVkFFT(&outApp, conf);
}

VkResult submitAndWait(VulkanContext& ctx, VkCommandBuffer cmd) {
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VkResult vr = vkQueueSubmit(ctx.queue, 1, &submit, ctx.fence);
    if (vr != VK_SUCCESS) {
        return vr;
    }
    vr = vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
    if (vr == VK_SUCCESS) {
        vkResetFences(ctx.device, 1, &ctx.fence);
    }
    return vr;
}

std::vector<uint32_t> compileComputeShader(const char* source) {
    glslang::InitializeProcess();
    glslang::TShader shader(EShLangCompute);
    shader.setStrings(&source, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, EShLangCompute, glslang::EShClientVulkan, 110);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);

    EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
    const TBuiltInResource* resources = GetDefaultResources();
    if (!shader.parse(resources, 110, false, messages)) {
        LOG_ERROR("glslang parse error: {}", shader.getInfoLog());
        glslang::FinalizeProcess();
        return {};
    }

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) {
        LOG_ERROR("glslang link error: {}", program.getInfoLog());
        glslang::FinalizeProcess();
        return {};
    }
    std::vector<uint32_t> spirv;
    glslang::GlslangToSpv(*program.getIntermediate(EShLangCompute), spirv);
    glslang::FinalizeProcess();
    return spirv;
}

bool createMultiplyPipeline(VulkanContext& ctx, const BufferResource& inputFreq,
                            const BufferResource& filterFreq, MultiplyPipeline& out) {
    const char* shaderSrc = R"(
        #version 450
        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
        layout(set = 0, binding = 0) buffer InputFreq { float data[]; } inputFreq;
        layout(set = 0, binding = 1) buffer FilterFreq { float data[]; } filterFreq;
        layout(push_constant) uniform Params { uint complexCount; } params;
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= params.complexCount) {
                return;
            }
            uint base = idx * 2;
            float aRe = inputFreq.data[base];
            float aIm = inputFreq.data[base + 1];
            float bRe = filterFreq.data[base];
            float bIm = filterFreq.data[base + 1];
            inputFreq.data[base] = aRe * bRe - aIm * bIm;
            inputFreq.data[base + 1] = aRe * bIm + aIm * bRe;
        }
    )";

    auto spirv = compileComputeShader(shaderSrc);
    if (spirv.empty()) {
        return false;
    }

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = spirv.size() * sizeof(uint32_t);
    moduleInfo.pCode = spirv.data();
    if (!checkVk(vkCreateShaderModule(ctx.device, &moduleInfo, nullptr, &out.shaderModule),
                 "vkCreateShaderModule")) {
        return false;
    }

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (!checkVk(vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &out.layout),
                 "vkCreateDescriptorSetLayout")) {
        return false;
    }

    VkPushConstantRange push{};
    push.offset = 0;
    push.size = sizeof(uint32_t);
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &out.layout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &push;
    if (!checkVk(
            vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr, &out.pipelineLayout),
            "vkCreatePipelineLayout")) {
        return false;
    }

    VkComputePipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = out.shaderModule;
    pipeInfo.stage.pName = "main";
    pipeInfo.layout = out.pipelineLayout;
    if (!checkVk(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
                                          &out.pipeline),
                 "vkCreateComputePipelines")) {
        return false;
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (!checkVk(vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &out.descriptorPool),
                 "vkCreateDescriptorPool")) {
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = out.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &out.layout;
    if (!checkVk(vkAllocateDescriptorSets(ctx.device, &allocInfo, &out.descriptorSet),
                 "vkAllocateDescriptorSets")) {
        return false;
    }

    VkDescriptorBufferInfo inputInfo{};
    inputInfo.buffer = inputFreq.buffer;
    inputInfo.offset = 0;
    inputInfo.range = inputFreq.size;

    VkDescriptorBufferInfo filterInfo{};
    filterInfo.buffer = filterFreq.buffer;
    filterInfo.offset = 0;
    filterInfo.range = filterFreq.size;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = out.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &inputInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = out.descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &filterInfo;
    vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);

    return true;
}

void destroyPipeline(VulkanContext& ctx, MultiplyPipeline& pipe) {
    if (pipe.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device, pipe.pipeline, nullptr);
        pipe.pipeline = VK_NULL_HANDLE;
    }
    if (pipe.pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, pipe.pipelineLayout, nullptr);
        pipe.pipelineLayout = VK_NULL_HANDLE;
    }
    if (pipe.layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device, pipe.layout, nullptr);
        pipe.layout = VK_NULL_HANDLE;
    }
    if (pipe.descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx.device, pipe.descriptorPool, nullptr);
        pipe.descriptorPool = VK_NULL_HANDLE;
    }
    if (pipe.shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(ctx.device, pipe.shaderModule, nullptr);
        pipe.shaderModule = VK_NULL_HANDLE;
    }
}

bool runFftOnBuffer(VulkanContext& ctx, VkFFTApplication& app, int direction) {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (!checkVk(vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd),
                 "vkAllocateCommandBuffers")) {
        return false;
    }

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!checkVk(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer")) {
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
        return false;
    }
    VkFFTLaunchParams launch{};
    launch.commandBuffer = &cmd;
    VkFFTResult res = VkFFTAppend(&app, direction, &launch);
    if (res != VKFFT_SUCCESS) {
        LOG_ERROR("VkFFTAppend failed: {}", static_cast<int>(res));
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
        return false;
    }
    if (!checkVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer")) {
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
        return false;
    }
    if (!checkVk(submitAndWait(ctx, cmd), "vkQueueSubmit (FFT)")) {
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
        return false;
    }
    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
    return true;
}

bool runBlock(VulkanContext& ctx, VkFFTApplication& fftApp, const MultiplyPipeline& pipe,
              uint32_t complexCount) {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (!checkVk(vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd),
                 "vkAllocateCommandBuffers block")) {
        return false;
    }

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!checkVk(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer block")) {
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
        return false;
    }

    VkFFTLaunchParams launch{};
    launch.commandBuffer = &cmd;
    VkFFTResult res = VkFFTAppend(&fftApp, -1, &launch);
    if (res != VKFFT_SUCCESS) {
        LOG_ERROR("VkFFT forward failed: {}", static_cast<int>(res));
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
        return false;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipelineLayout, 0, 1,
                            &pipe.descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t),
                       &complexCount);
    const uint32_t groupCount = (complexCount + 255u) / 256u;
    vkCmdDispatch(cmd, groupCount, 1, 1);

    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                         nullptr);

    res = VkFFTAppend(&fftApp, 1, &launch);
    if (res != VKFFT_SUCCESS) {
        LOG_ERROR("VkFFT inverse failed: {}", static_cast<int>(res));
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
        return false;
    }

    if (!checkVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer block")) {
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
        return false;
    }
    bool ok = checkVk(submitAndWait(ctx, cmd), "vkQueueSubmit block");
    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
    return ok;
}

std::size_t readFilterTaps(const std::string& path, std::vector<float>& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        LOG_ERROR("Failed to open filter coefficients: {}", path);
        return 0;
    }
    ifs.seekg(0, std::ios::end);
    const auto bytes = ifs.tellg();
    if (bytes <= 0) {
        LOG_ERROR("Filter file is empty: {}", path);
        return 0;
    }
    const auto taps = static_cast<std::size_t>(bytes) / sizeof(float);
    ifs.seekg(0, std::ios::beg);
    out.resize(taps);
    ifs.read(reinterpret_cast<char*>(out.data()),
             static_cast<std::streamsize>(taps * sizeof(float)));
    if (ifs.gcount() != static_cast<std::streamsize>(taps * sizeof(float))) {
        LOG_ERROR("Failed to read all filter taps from {}", path);
        return 0;
    }
    return taps;
}

std::string derivePhasePath(const std::string& basePath, bool wantLinear) {
    // Best-effort path derivation:
    // - "..._min_phase.bin" <-> "..._linear_phase.bin"
    std::string p = basePath;
    const std::string kMin = "_min_phase";
    const std::string kLinear = "_linear_phase";
    if (wantLinear) {
        auto pos = p.find(kMin);
        if (pos != std::string::npos) {
            p.replace(pos, kMin.size(), kLinear);
            return p;
        }
        return basePath;
    }
    auto pos = p.find(kLinear);
    if (pos != std::string::npos) {
        p.replace(pos, kLinear.size(), kMin);
        return p;
    }
    return basePath;
}

inline std::size_t familyToIndex(RateFamily family) {
    return family == RateFamily::RATE_48K ? 1u : 0u;
}

inline RateFamily indexToFamily(std::size_t idx) {
    return idx == 1u ? RateFamily::RATE_48K : RateFamily::RATE_44K;
}

bool uploadTimeDomainAndFft(VulkanContext& ctx, BufferResource& filterBuf,
                            VkFFTApplication& filterApp, uint64_t bufferBytes,
                            const std::vector<float>& taps) {
    void* mapped = nullptr;
    if (!checkVk(vkMapMemory(ctx.device, filterBuf.memory, 0, bufferBytes, 0, &mapped),
                 "vkMapMemory filter")) {
        return false;
    }
    std::memset(mapped, 0, static_cast<std::size_t>(bufferBytes));
    std::memcpy(mapped, taps.data(), taps.size() * sizeof(float));
    vkUnmapMemory(ctx.device, filterBuf.memory);

    if (!runFftOnBuffer(ctx, filterApp, -1)) {
        return false;
    }
    return true;
}

bool readFilterSpectrum(VulkanContext& ctx, BufferResource& filterBuf, uint64_t bufferBytes,
                        uint32_t complexCount, std::vector<float>& outSpectrum) {
    const uint64_t complexElements = 2ull * complexCount;
    void* mapped = nullptr;
    if (!checkVk(vkMapMemory(ctx.device, filterBuf.memory, 0, bufferBytes, 0, &mapped),
                 "vkMapMemory filter (read spectrum)")) {
        return false;
    }
    outSpectrum.resize(static_cast<std::size_t>(complexElements));
    std::memcpy(outSpectrum.data(), mapped, static_cast<std::size_t>(bufferBytes));
    vkUnmapMemory(ctx.device, filterBuf.memory);
    return true;
}

bool writeFilterSpectrum(VulkanContext& ctx, BufferResource& filterBuf, uint64_t bufferBytes,
                         uint32_t complexCount, const std::vector<float>& spectrum) {
    if (spectrum.size() != static_cast<std::size_t>(2ull * complexCount)) {
        LOG_ERROR("[Vulkan] Invalid spectrum size: expected {}, got {}",
                  static_cast<std::size_t>(2ull * complexCount), spectrum.size());
        return false;
    }
    void* mapped = nullptr;
    if (!checkVk(vkMapMemory(ctx.device, filterBuf.memory, 0, bufferBytes, 0, &mapped),
                 "vkMapMemory filter (write spectrum)")) {
        return false;
    }
    std::memcpy(mapped, spectrum.data(), static_cast<std::size_t>(bufferBytes));
    vkUnmapMemory(ctx.device, filterBuf.memory);
    return true;
}

}  // namespace

namespace vulkan_backend {

struct VulkanStreamingUpsampler::Impl {
    VulkanContext ctx{};
    BufferResource inputBuf{};
    BufferResource filterBuf{};
    VkFFTApplication inputApp{};
    VkFFTApplication filterApp{};
    MultiplyPipeline pipeline{};

    struct PhaseSpectrum {
        std::vector<float> data;
        std::string path;
        bool available = false;
    };

    struct FamilyCache {
        PhaseSpectrum minimum;
        PhaseSpectrum linear;
        RateFamily family = RateFamily::RATE_UNKNOWN;
        uint32_t inputRate = 0;
    };

    uint32_t upsampleRatio = 0;
    uint32_t inputRate = 0;
    uint32_t outputRate = 0;
    uint32_t fftSize = 0;
    uint32_t hopInput = 0;
    uint32_t hopOutput = 0;
    uint32_t overlap = 0;
    uint32_t complexCount = 0;
    uint64_t bufferBytes = 0;
    std::vector<float> overlapBuffer;
    std::vector<float> originalFilterFreq;
    std::array<FamilyCache, 2> families{};
    RateFamily activeFamily = RateFamily::RATE_UNKNOWN;
    PhaseType phaseType = PhaseType::Minimum;
    bool multiRateEnabled = false;
    bool eqApplied = false;

    bool initialized = false;
    std::mutex mutex;

    void cleanup() {
        destroyPipeline(ctx, pipeline);
        deleteVkFFT(&filterApp);
        deleteVkFFT(&inputApp);
        destroyBuffer(ctx, filterBuf);
        destroyBuffer(ctx, inputBuf);
        destroyContext(ctx);
        overlapBuffer.clear();
        originalFilterFreq.clear();
        for (auto& family : families) {
            family.minimum.data.clear();
            family.minimum.path.clear();
            family.minimum.available = false;
            family.linear.data.clear();
            family.linear.path.clear();
            family.linear.available = false;
            family.family = RateFamily::RATE_UNKNOWN;
            family.inputRate = 0;
        }
        upsampleRatio = 0;
        inputRate = 0;
        outputRate = 0;
        fftSize = 0;
        hopInput = 0;
        hopOutput = 0;
        overlap = 0;
        complexCount = 0;
        bufferBytes = 0;
        activeFamily = RateFamily::RATE_UNKNOWN;
        phaseType = PhaseType::Minimum;
        multiRateEnabled = false;
        eqApplied = false;
        initialized = false;
    }
};

VulkanStreamingUpsampler::VulkanStreamingUpsampler() : impl_(std::make_unique<Impl>()) {}

VulkanStreamingUpsampler::~VulkanStreamingUpsampler() {
    if (impl_) {
        impl_->cleanup();
    }
}

bool VulkanStreamingUpsampler::initialize(const InitParams& params) {
    if (!impl_) {
        return false;
    }
    impl_->cleanup();

    if (params.upsampleRatio == 0 || params.blockSize == 0 || params.inputRate == 0) {
        LOG_ERROR("Vulkan upsampler: invalid params (zero ratio/block/input)");
        return false;
    }

    // Fallback paths (legacy single-rate)
    std::string fallbackMinimum =
        !params.filterPathMinimum.empty() ? params.filterPathMinimum : params.filterPath;
    std::string fallbackLinear = params.filterPathLinear;
    if (fallbackLinear.empty() && !fallbackMinimum.empty()) {
        fallbackLinear = derivePhasePath(fallbackMinimum, true);
    }

    struct FamilyPaths {
        std::string min;
        std::string linear;
    };

    RateFamily initialFamily = detectRateFamily(static_cast<int>(params.inputRate));
    if (initialFamily == RateFamily::RATE_UNKNOWN) {
        initialFamily = RateFamily::RATE_44K;
    }
    const std::size_t initialIdx = familyToIndex(initialFamily);

    std::array<FamilyPaths, 2> familyPaths{};
    familyPaths[familyToIndex(RateFamily::RATE_44K)].min =
        !params.filterPathMinimum44k.empty() ? params.filterPathMinimum44k : fallbackMinimum;
    familyPaths[familyToIndex(RateFamily::RATE_44K)].linear =
        !params.filterPathLinear44k.empty() ? params.filterPathLinear44k : fallbackLinear;

    familyPaths[familyToIndex(RateFamily::RATE_48K)].min =
        !params.filterPathMinimum48k.empty()
            ? params.filterPathMinimum48k
            : (initialFamily == RateFamily::RATE_48K ? fallbackMinimum : std::string());
    familyPaths[familyToIndex(RateFamily::RATE_48K)].linear =
        !params.filterPathLinear48k.empty()
            ? params.filterPathLinear48k
            : (initialFamily == RateFamily::RATE_48K ? fallbackLinear : std::string());

    struct FamilyTaps {
        std::vector<float> min;
        std::vector<float> linear;
        bool hasMin = false;
        bool hasLinear = false;
    };

    std::array<FamilyTaps, 2> taps{};
    std::size_t globalMaxTaps = 0;

    for (std::size_t idx = 0; idx < familyPaths.size(); ++idx) {
        if (!familyPaths[idx].min.empty()) {
            const std::size_t count = readFilterTaps(familyPaths[idx].min, taps[idx].min);
            taps[idx].hasMin = (count > 0);
            globalMaxTaps = std::max(globalMaxTaps, count);
        }
        if (!familyPaths[idx].linear.empty()) {
            const std::size_t count = readFilterTaps(familyPaths[idx].linear, taps[idx].linear);
            taps[idx].hasLinear = (count > 0);
            globalMaxTaps = std::max(globalMaxTaps, count);
        }
    }

    if (!taps[initialIdx].hasMin) {
        LOG_ERROR("[Vulkan] Initial family filter missing (family={}, path='{}')",
                  initialFamily == RateFamily::RATE_48K ? "48k" : "44k",
                  familyPaths[initialIdx].min);
        return false;
    }
    if (globalMaxTaps == 0) {
        LOG_ERROR("[Vulkan] No filter taps loaded (min path(s) empty?)");
        return false;
    }

    // Pad all filters to the same length so hop/overlap remain stable across switches.
    for (std::size_t idx = 0; idx < taps.size(); ++idx) {
        if (taps[idx].hasMin && taps[idx].min.size() < globalMaxTaps) {
            taps[idx].min.resize(globalMaxTaps, 0.0f);
        }
        if (taps[idx].hasLinear && taps[idx].linear.size() < globalMaxTaps) {
            taps[idx].linear.resize(globalMaxTaps, 0.0f);
        }
    }

    impl_->upsampleRatio = params.upsampleRatio;
    impl_->inputRate = params.inputRate;
    impl_->outputRate = params.inputRate * params.upsampleRatio;
    impl_->overlap = static_cast<uint32_t>(globalMaxTaps - 1);
    impl_->phaseType = params.initialPhase;
    impl_->multiRateEnabled = taps[0].hasMin && taps[1].hasMin;
    impl_->activeFamily = initialFamily;

    uint32_t fftSize = params.fftSizeOverride;
    if (fftSize == 0) {
        const auto desired = static_cast<std::size_t>(params.upsampleRatio) *
                                 static_cast<std::size_t>(params.blockSize) +
                             globalMaxTaps - 1;
        fftSize = nextPow2(static_cast<uint32_t>(desired));
    }
    if (fftSize <= impl_->overlap) {
        LOG_ERROR("Vulkan upsampler: fftSize {} too small for overlap {}", fftSize, impl_->overlap);
        return false;
    }

    impl_->fftSize = fftSize;
    impl_->complexCount = fftSize / 2 + 1;
    const uint64_t complexElements = 2ull * impl_->complexCount;
    impl_->bufferBytes = sizeof(float) * complexElements;

    for (std::size_t idx = 0; idx < impl_->families.size(); ++idx) {
        impl_->families[idx].family = indexToFamily(idx);
        impl_->families[idx].inputRate =
            static_cast<uint32_t>(getBaseSampleRate(indexToFamily(idx)));
        impl_->families[idx].minimum.path = familyPaths[idx].min;
        impl_->families[idx].linear.path = familyPaths[idx].linear;
        impl_->families[idx].minimum.available = taps[idx].hasMin;
        impl_->families[idx].linear.available = taps[idx].hasLinear;
    }

    if (!initContext(impl_->ctx)) {
        impl_->cleanup();
        return false;
    }

    if (!createHostBuffer(impl_->ctx, impl_->bufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          impl_->inputBuf) ||
        !createHostBuffer(impl_->ctx, impl_->bufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          impl_->filterBuf)) {
        impl_->cleanup();
        return false;
    }

    if (createFftApp(impl_->ctx, impl_->inputBuf.buffer, impl_->bufferBytes, impl_->fftSize,
                     impl_->inputApp) != VKFFT_SUCCESS ||
        createFftApp(impl_->ctx, impl_->filterBuf.buffer, impl_->bufferBytes, impl_->fftSize,
                     impl_->filterApp) != VKFFT_SUCCESS) {
        LOG_ERROR("Vulkan upsampler: Failed to initialize VkFFT");
        impl_->cleanup();
        return false;
    }

    // Pre-compute frequency spectra for every available family/phase.
    for (std::size_t idx = 0; idx < taps.size(); ++idx) {
        if (taps[idx].hasMin) {
            if (!uploadTimeDomainAndFft(impl_->ctx, impl_->filterBuf, impl_->filterApp,
                                        impl_->bufferBytes, taps[idx].min) ||
                !readFilterSpectrum(impl_->ctx, impl_->filterBuf, impl_->bufferBytes,
                                    impl_->complexCount, impl_->families[idx].minimum.data)) {
                impl_->cleanup();
                return false;
            }
        }
        if (taps[idx].hasLinear) {
            if (!uploadTimeDomainAndFft(impl_->ctx, impl_->filterBuf, impl_->filterApp,
                                        impl_->bufferBytes, taps[idx].linear) ||
                !readFilterSpectrum(impl_->ctx, impl_->filterBuf, impl_->bufferBytes,
                                    impl_->complexCount, impl_->families[idx].linear.data)) {
                impl_->cleanup();
                return false;
            }
        }
    }

    if (impl_->phaseType == PhaseType::Linear && !impl_->families[initialIdx].linear.available) {
        LOG_WARN(
            "[Vulkan] Requested initial phase 'linear' but linear filter is unavailable; "
            "falling back to 'minimum'");
        impl_->phaseType = PhaseType::Minimum;
    }

    const auto& activeFamilyCache = impl_->families[initialIdx];
    const bool useLinear = (impl_->phaseType == PhaseType::Linear);
    const std::vector<float>* activeBase = (useLinear && activeFamilyCache.linear.available)
                                               ? &activeFamilyCache.linear.data
                                               : &activeFamilyCache.minimum.data;

    if (!activeBase || activeBase->empty()) {
        LOG_ERROR("[Vulkan] Active filter spectrum is missing for initial family");
        impl_->cleanup();
        return false;
    }

    if (!writeFilterSpectrum(impl_->ctx, impl_->filterBuf, impl_->bufferBytes, impl_->complexCount,
                             *activeBase)) {
        impl_->cleanup();
        return false;
    }
    impl_->originalFilterFreq = *activeBase;
    impl_->eqApplied = false;

    if (!createMultiplyPipeline(impl_->ctx, impl_->inputBuf, impl_->filterBuf, impl_->pipeline)) {
        impl_->cleanup();
        return false;
    }

    // Compute streaming hop sizes (align output to upsampleRatio)
    const uint32_t hopOutputMax = impl_->fftSize - impl_->overlap;
    const uint32_t hopOutputCandidate = hopOutputMax - (hopOutputMax % impl_->upsampleRatio);
    impl_->hopInput = hopOutputCandidate / impl_->upsampleRatio;
    impl_->hopOutput = impl_->hopInput * impl_->upsampleRatio;
    if (impl_->hopInput == 0) {
        LOG_ERROR("Vulkan upsampler: hopInput computed as 0 (fftSize={}, overlap={})",
                  impl_->fftSize, impl_->overlap);
        impl_->cleanup();
        return false;
    }

    impl_->overlapBuffer.assign(impl_->overlap, 0.0f);
    impl_->initialized = true;
    LOG_INFO(
        "[Vulkan] upsampler ready (ratio={}x, block={} frames, fftSize={}, phase={}, families "
        "loaded: {}44k / {}48k)",
        impl_->upsampleRatio, params.blockSize, impl_->fftSize,
        (impl_->phaseType == PhaseType::Minimum ? "minimum" : "linear"),
        taps[familyToIndex(RateFamily::RATE_44K)].hasMin ? "✓" : "×",
        taps[familyToIndex(RateFamily::RATE_48K)].hasMin ? "✓" : "×");
    return true;
}

bool VulkanStreamingUpsampler::initializeStreaming() {
    return impl_ && impl_->initialized;
}

void VulkanStreamingUpsampler::resetStreaming() {
    if (!impl_ || !impl_->initialized) {
        return;
    }
    std::fill(impl_->overlapBuffer.begin(), impl_->overlapBuffer.end(), 0.0f);
}

size_t VulkanStreamingUpsampler::getStreamValidInputPerBlock() const {
    if (!impl_) {
        return 0;
    }
    return impl_->hopInput;
}

int VulkanStreamingUpsampler::getUpsampleRatio() const {
    return impl_ ? static_cast<int>(impl_->upsampleRatio) : 0;
}

int VulkanStreamingUpsampler::getOutputSampleRate() const {
    return impl_ ? static_cast<int>(impl_->outputRate) : 0;
}

int VulkanStreamingUpsampler::getInputSampleRate() const {
    return impl_ ? static_cast<int>(impl_->inputRate) : 0;
}

bool VulkanStreamingUpsampler::isMultiRateEnabled() const {
    return impl_ && impl_->multiRateEnabled;
}

int VulkanStreamingUpsampler::getCurrentInputRate() const {
    return getInputSampleRate();
}

bool VulkanStreamingUpsampler::switchToInputRate(int inputSampleRate) {
    if (!impl_ || !impl_->initialized) {
        LOG_ERROR("[Vulkan] Rate switch requested but upsampler not initialized");
        return false;
    }

    RateFamily targetFamily = detectRateFamily(inputSampleRate);
    if (targetFamily == RateFamily::RATE_UNKNOWN) {
        LOG_ERROR("[Vulkan] Unsupported input sample rate for Vulkan backend: {}", inputSampleRate);
        return false;
    }

    const std::size_t targetIdx = familyToIndex(targetFamily);
    const auto& targetFamilyCache = impl_->families[targetIdx];
    if (!targetFamilyCache.minimum.available) {
        LOG_ERROR("[Vulkan] No filter loaded for target family {}",
                  targetFamily == RateFamily::RATE_48K ? "48k" : "44k");
        return false;
    }

    if (targetFamily == impl_->activeFamily &&
        inputSampleRate == static_cast<int>(impl_->inputRate)) {
        return true;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (targetFamily == impl_->activeFamily &&
        inputSampleRate == static_cast<int>(impl_->inputRate)) {
        return true;
    }

    const bool wantLinear = (impl_->phaseType == PhaseType::Linear);
    if (wantLinear && !targetFamilyCache.linear.available) {
        LOG_WARN("[Vulkan] Linear phase requested but linear filter not loaded for target family");
        return false;
    }

    const std::vector<float>* spectrum = (wantLinear && targetFamilyCache.linear.available)
                                             ? &targetFamilyCache.linear.data
                                             : &targetFamilyCache.minimum.data;
    if (!spectrum || spectrum->empty()) {
        LOG_ERROR("[Vulkan] Target filter spectrum missing for rate switch");
        return false;
    }

    if (!writeFilterSpectrum(impl_->ctx, impl_->filterBuf, impl_->bufferBytes, impl_->complexCount,
                             *spectrum)) {
        return false;
    }

    impl_->originalFilterFreq = *spectrum;
    impl_->activeFamily = targetFamily;
    impl_->inputRate = static_cast<uint32_t>(inputSampleRate);
    impl_->outputRate = impl_->inputRate * impl_->upsampleRatio;
    impl_->eqApplied = false;  // Control plane should re-apply EQ after a family switch
    std::fill(impl_->overlapBuffer.begin(), impl_->overlapBuffer.end(), 0.0f);

    LOG_INFO("[Vulkan] Switched input rate to {} Hz (family={}, phase={}) — please re-apply EQ",
             inputSampleRate, impl_->activeFamily == RateFamily::RATE_48K ? "48k" : "44k",
             (impl_->phaseType == PhaseType::Minimum ? "minimum" : "linear"));
    return true;
}

PhaseType VulkanStreamingUpsampler::getPhaseType() const {
    if (!impl_) {
        return PhaseType::Minimum;
    }
    return impl_->phaseType;
}

bool VulkanStreamingUpsampler::switchPhaseType(PhaseType targetPhase) {
    if (!impl_ || !impl_->initialized) {
        LOG_ERROR("[Vulkan] Phase switch requested but upsampler not initialized");
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (targetPhase == impl_->phaseType) {
        return true;
    }

    RateFamily active = impl_->activeFamily;
    if (active == RateFamily::RATE_UNKNOWN) {
        active = detectRateFamily(static_cast<int>(impl_->inputRate));
    }
    const std::size_t idx = familyToIndex(active);
    const auto& cache = impl_->families[idx];

    if (targetPhase == PhaseType::Linear && !cache.linear.available) {
        LOG_WARN(
            "[Vulkan] Phase switch to 'linear' requested but linear filter is unavailable for "
            "active family");
        return false;
    }

    const std::vector<float>* activeBase =
        (targetPhase == PhaseType::Linear) ? &cache.linear.data : &cache.minimum.data;
    if (!activeBase || activeBase->empty()) {
        LOG_ERROR("[Vulkan] Active family spectrum missing for phase switch");
        return false;
    }
    if (!writeFilterSpectrum(impl_->ctx, impl_->filterBuf, impl_->bufferBytes, impl_->complexCount,
                             *activeBase)) {
        return false;
    }
    impl_->originalFilterFreq =
        *activeBase;  // Reset EQ base; ControlPlane will re-apply EQ if enabled.
    impl_->phaseType = targetPhase;
    impl_->eqApplied = false;
    std::fill(impl_->overlapBuffer.begin(), impl_->overlapBuffer.end(), 0.0f);

    LOG_INFO("[Vulkan] Phase type switched to {} (EQ should be re-applied)",
             (impl_->phaseType == PhaseType::Minimum ? "minimum" : "linear"));
    return true;
}

size_t VulkanStreamingUpsampler::getFilterFftSize() const {
    return impl_ ? static_cast<size_t>(impl_->complexCount) : 0;
}

size_t VulkanStreamingUpsampler::getFullFftSize() const {
    return impl_ ? static_cast<size_t>(impl_->fftSize) : 0;
}

bool VulkanStreamingUpsampler::applyEqMagnitude(const std::vector<double>& eqMagnitude) {
    if (!impl_ || !impl_->initialized) {
        LOG_ERROR("[Vulkan] EQ: Upsampler not initialized");
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (eqMagnitude.size() != impl_->complexCount) {
        LOG_ERROR("[Vulkan] EQ: Magnitude size mismatch: expected {}, got {}", impl_->complexCount,
                  eqMagnitude.size());
        return false;
    }
    if (impl_->originalFilterFreq.size() != static_cast<std::size_t>(impl_->complexCount) * 2u) {
        LOG_ERROR("[Vulkan] EQ: Original filter spectrum not cached");
        return false;
    }

    // Auto-normalize EQ gain to avoid clipping
    std::vector<double> normalizedMagnitude = eqMagnitude;
    double maxMag = *std::max_element(eqMagnitude.begin(), eqMagnitude.end());
    double normalizationFactor = 1.0;
    if (maxMag > 1.0) {
        normalizationFactor = 1.0 / maxMag;
        for (double& v : normalizedMagnitude) {
            v *= normalizationFactor;
        }
        double normDb = 20.0 * std::log10(std::max(normalizationFactor, 1e-30));
        double maxDb = 20.0 * std::log10(std::max(maxMag, 1e-30));
        LOG_INFO("[Vulkan] EQ: Auto-normalization applied: {} dB (max boost was +{} dB)", normDb,
                 maxDb);
    }

    std::vector<float> newSpectrum(static_cast<std::size_t>(impl_->complexCount) * 2u);
    for (uint32_t i = 0; i < impl_->complexCount; ++i) {
        float scale = static_cast<float>(normalizedMagnitude[i]);
        size_t base = static_cast<size_t>(i) * 2;
        newSpectrum[base] = impl_->originalFilterFreq[base] * scale;
        newSpectrum[base + 1] = impl_->originalFilterFreq[base + 1] * scale;
    }

    void* mapped = nullptr;
    if (!checkVk(vkMapMemory(impl_->ctx.device, impl_->filterBuf.memory, 0, impl_->bufferBytes, 0,
                             &mapped),
                 "vkMapMemory filter (apply EQ)")) {
        return false;
    }
    std::memcpy(mapped, newSpectrum.data(), static_cast<std::size_t>(impl_->bufferBytes));
    vkUnmapMemory(impl_->ctx.device, impl_->filterBuf.memory);

    LOG_INFO("[Vulkan] EQ magnitude applied ({} bins)", impl_->complexCount);
    impl_->eqApplied = true;
    return true;
}

bool VulkanStreamingUpsampler::processStreamBlock(
    const float* inputData, size_t inputFrames, ConvolutionEngine::StreamFloatVector& outputData,
    ConvolutionEngine::DeviceStream /*stream*/,
    ConvolutionEngine::StreamFloatVector& streamInputBuffer, size_t& streamInputAccumulated) {
    if (!impl_ || !impl_->initialized) {
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!inputData || inputFrames == 0) {
        return false;
    }

    // Ensure input buffer has enough space
    size_t required =
        streamInputAccumulated + std::max(inputFrames, static_cast<size_t>(impl_->hopInput));
    if (streamInputBuffer.size() < required) {
        streamInputBuffer.resize(required, 0.0f);
    }

    std::memcpy(streamInputBuffer.data() + static_cast<std::ptrdiff_t>(streamInputAccumulated),
                inputData, inputFrames * sizeof(float));
    streamInputAccumulated += inputFrames;

    outputData.clear();
    const uint64_t complexElements = 2ull * impl_->complexCount;
    bool produced = false;

    while (streamInputAccumulated >= impl_->hopInput) {
        std::vector<float> padded(complexElements, 0.0f);
        std::copy(impl_->overlapBuffer.begin(), impl_->overlapBuffer.end(), padded.begin());
        for (uint32_t i = 0; i < impl_->hopInput; ++i) {
            padded[impl_->overlap + static_cast<std::size_t>(i) * impl_->upsampleRatio] =
                streamInputBuffer[i];
        }

        void* mapped = nullptr;
        if (!checkVk(vkMapMemory(impl_->ctx.device, impl_->inputBuf.memory, 0, impl_->bufferBytes,
                                 0, &mapped),
                     "vkMapMemory input")) {
            return false;
        }
        std::memcpy(mapped, padded.data(), static_cast<std::size_t>(impl_->bufferBytes));
        vkUnmapMemory(impl_->ctx.device, impl_->inputBuf.memory);

        if (!runBlock(impl_->ctx, impl_->inputApp, impl_->pipeline, impl_->complexCount)) {
            return false;
        }

        mapped = nullptr;
        if (!checkVk(vkMapMemory(impl_->ctx.device, impl_->inputBuf.memory, 0, impl_->bufferBytes,
                                 0, &mapped),
                     "vkMapMemory output")) {
            return false;
        }
        const auto* data = static_cast<const float*>(mapped);
        const uint32_t validOutput = impl_->hopInput * impl_->upsampleRatio;
        const uint32_t copyCount = std::min(impl_->hopOutput, validOutput);
        size_t current = outputData.size();
        outputData.resize(current + copyCount);
        std::memcpy(outputData.data() + static_cast<std::ptrdiff_t>(current), data + impl_->overlap,
                    copyCount * sizeof(float));
        vkUnmapMemory(impl_->ctx.device, impl_->inputBuf.memory);

        // Update overlap
        std::copy(padded.end() - impl_->overlap, padded.end(), impl_->overlapBuffer.begin());

        // Shift remaining input samples to the start
        if (streamInputAccumulated > impl_->hopInput) {
            std::memmove(streamInputBuffer.data(),
                         streamInputBuffer.data() + static_cast<std::ptrdiff_t>(impl_->hopInput),
                         (streamInputAccumulated - impl_->hopInput) * sizeof(float));
        }
        streamInputAccumulated -= impl_->hopInput;
        produced = true;
    }

    return produced;
}

}  // namespace vulkan_backend
