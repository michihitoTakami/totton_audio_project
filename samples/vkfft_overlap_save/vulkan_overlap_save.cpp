#include "vulkan_overlap_save.h"

#include "audio/audio_io.h"
#include "logging/logger.h"
#include "vkFFT.h"

#include <SPIRV/GlslangToSpv.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

using AudioIO::AudioFile;
using AudioIO::WavReader;
using AudioIO::WavWriter;

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
    appInfo.pApplicationName = "vulkan_overlap_save";
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

bool readFileFloats(const std::string& path, std::size_t expected, std::vector<float>& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        LOG_ERROR("Failed to open filter file: {}", path);
        return false;
    }
    ifs.seekg(0, std::ios::end);
    const std::streamsize bytes = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (bytes < 0 || static_cast<std::size_t>(bytes) < expected * sizeof(float)) {
        LOG_ERROR("Filter file too small: {} bytes, expected at least {}", bytes,
                  expected * sizeof(float));
        return false;
    }
    out.resize(expected);
    ifs.read(reinterpret_cast<char*>(out.data()), expected * sizeof(float));
    return static_cast<std::size_t>(ifs.gcount()) == expected * sizeof(float);
}

}  // namespace

bool loadFilterMetadata(const std::string& jsonPath, FilterMetadata& out) {
    std::ifstream ifs(jsonPath);
    if (!ifs) {
        LOG_ERROR("Failed to open filter metadata: {}", jsonPath);
        return false;
    }
    nlohmann::json j;
    try {
        ifs >> j;
        out.taps = j.value("n_taps_actual", 0);
        out.inputRate = j.value("sample_rate_input", 0);
        out.outputRate = j.value("sample_rate_output", 0);
        out.upsampleRatio = j.value("upsample_ratio", 0);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse filter metadata {}: {}", jsonPath, e.what());
        return false;
    }
    if (out.taps == 0 || out.upsampleRatio == 0) {
        LOG_ERROR("Invalid metadata in {}", jsonPath);
        return false;
    }
    return true;
}

bool loadFilterCoefficients(const std::string& binPath, std::size_t taps, std::vector<float>& out) {
    return readFileFloats(binPath, taps, out);
}

bool processOverlapSaveBuffer(const std::vector<float>& inputMono,
                              const std::vector<float>& filterTaps, uint32_t upsampleRatio,
                              uint32_t fftSize, uint32_t chunkFrames, std::vector<float>& output) {
    if (filterTaps.empty() || upsampleRatio == 0) {
        LOG_ERROR("Invalid parameters: empty filter or zero upsample ratio");
        return false;
    }
    const std::size_t overlap = filterTaps.size() - 1;
    if (fftSize <= overlap) {
        LOG_ERROR("fftSize={} is too small for overlap {}", fftSize, overlap);
        return false;
    }
    const uint64_t complexElements = 2ull * (fftSize / 2 + 1);
    const uint64_t bufferBytes = sizeof(float) * complexElements;
    const uint32_t complexCount = fftSize / 2 + 1;

    VulkanContext ctx{};
    if (!initContext(ctx)) {
        return false;
    }

    BufferResource inputBuf{};
    BufferResource filterBuf{};
    VkFFTApplication inputApp{};
    VkFFTApplication filterApp{};
    MultiplyPipeline pipeline{};

    auto cleanup = [&]() {
        destroyPipeline(ctx, pipeline);
        deleteVkFFT(&filterApp);
        deleteVkFFT(&inputApp);
        destroyBuffer(ctx, filterBuf);
        destroyBuffer(ctx, inputBuf);
        destroyContext(ctx);
    };

    if (!createHostBuffer(ctx, bufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, inputBuf) ||
        !createHostBuffer(ctx, bufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, filterBuf)) {
        cleanup();
        return false;
    }

    if (createFftApp(ctx, inputBuf.buffer, bufferBytes, fftSize, inputApp) != VKFFT_SUCCESS ||
        createFftApp(ctx, filterBuf.buffer, bufferBytes, fftSize, filterApp) != VKFFT_SUCCESS) {
        LOG_ERROR("Failed to initialize VkFFT");
        cleanup();
        return false;
    }

    // Upload filter taps (time domain)
    {
        void* mapped = nullptr;
        if (!checkVk(vkMapMemory(ctx.device, filterBuf.memory, 0, bufferBytes, 0, &mapped),
                     "vkMapMemory filter")) {
            cleanup();
            return false;
        }
        std::memset(mapped, 0, static_cast<std::size_t>(bufferBytes));
        std::memcpy(mapped, filterTaps.data(), filterTaps.size() * sizeof(float));
        vkUnmapMemory(ctx.device, filterBuf.memory);
    }

    if (!runFftOnBuffer(ctx, filterApp, -1)) {
        cleanup();
        return false;
    }

    if (!createMultiplyPipeline(ctx, inputBuf, filterBuf, pipeline)) {
        cleanup();
        return false;
    }

    std::vector<float> overlapBuffer(overlap, 0.0f);
    std::vector<float> padded(complexElements, 0.0f);
    output.clear();
    output.reserve(static_cast<std::size_t>(inputMono.size()) * upsampleRatio);

    const uint32_t hopOutputMax = fftSize - static_cast<uint32_t>(overlap);
    const uint32_t hopOutputCandidate = hopOutputMax - (hopOutputMax % upsampleRatio);
    const uint32_t hopInputMax = hopOutputCandidate / upsampleRatio;
    const uint32_t hopInput = std::min<uint32_t>(hopInputMax, std::max<uint32_t>(1, chunkFrames));
    const uint32_t hopOutput = hopInput * upsampleRatio;
    if (hopInput == 0) {
        LOG_ERROR("fftSize too small for given filter/ratio");
        cleanup();
        return false;
    }

    std::size_t processed = 0;
    while (processed < inputMono.size()) {
        const auto framesThis =
            static_cast<uint32_t>(std::min<std::size_t>(hopInput, inputMono.size() - processed));
        std::fill(padded.begin(), padded.end(), 0.0f);
        std::copy(overlapBuffer.begin(), overlapBuffer.end(), padded.begin());
        for (uint32_t i = 0; i < framesThis; ++i) {
            padded[overlap + static_cast<std::size_t>(i) * upsampleRatio] =
                inputMono[processed + i];
        }

        // Write to GPU buffer
        void* mapped = nullptr;
        if (!checkVk(vkMapMemory(ctx.device, inputBuf.memory, 0, bufferBytes, 0, &mapped),
                     "vkMapMemory input")) {
            cleanup();
            return false;
        }
        std::memcpy(mapped, padded.data(), static_cast<std::size_t>(bufferBytes));
        vkUnmapMemory(ctx.device, inputBuf.memory);

        if (!runBlock(ctx, inputApp, pipeline, complexCount)) {
            cleanup();
            return false;
        }

        // Read back
        mapped = nullptr;
        if (!checkVk(vkMapMemory(ctx.device, inputBuf.memory, 0, bufferBytes, 0, &mapped),
                     "vkMapMemory output")) {
            cleanup();
            return false;
        }
        std::vector<float> blockOutput(hopOutput);
        const auto* data = static_cast<const float*>(mapped);
        const uint32_t validOutput = framesThis * upsampleRatio;
        const uint32_t copyCount = std::min(hopOutput, validOutput);
        std::memcpy(blockOutput.data(), data + overlap, copyCount * sizeof(float));
        vkUnmapMemory(ctx.device, inputBuf.memory);

        output.insert(output.end(), blockOutput.begin(), blockOutput.begin() + copyCount);

        // Update overlap buffer using padded input tail
        std::copy(padded.end() - overlap, padded.end(), overlapBuffer.begin());
        processed += framesThis;
    }

    cleanup();
    return true;
}

int runVulkanOverlapSave(const VulkanOverlapSaveOptions& opts) {
    FilterMetadata meta{};
    if (!loadFilterMetadata(opts.filterMetadataPath, meta)) {
        return 1;
    }
    std::vector<float> filterTaps;
    if (!loadFilterCoefficients(opts.filterPath, meta.taps, filterTaps)) {
        return 1;
    }
    if (filterTaps.size() < 120000) {
        LOG_WARN("Filter taps {} is below acceptance threshold (120k)", filterTaps.size());
    }

    WavReader reader;
    if (!reader.open(opts.inputPath)) {
        return 1;
    }
    const int inputRate = reader.getSampleRate();
    AudioFile inputFile;
    if (!reader.readAll(inputFile)) {
        reader.close();
        return 1;
    }
    reader.close();

    std::vector<float> mono(inputFile.frames);
    if (inputFile.channels == 1) {
        mono = std::move(inputFile.data);
    } else {
        AudioIO::Utils::stereoToMono(inputFile.data.data(), mono.data(), inputFile.frames);
    }

    if (meta.inputRate != 0 && static_cast<uint32_t>(inputRate) != meta.inputRate) {
        LOG_WARN(
            "Input sample rate ({}) does not match filter metadata input rate ({}). "
            "Proceeding with the file rate.",
            inputRate, meta.inputRate);
    }

    uint32_t fftSize = opts.fftSizeOverride;
    if (fftSize == 0) {
        const auto desired = static_cast<std::size_t>(meta.upsampleRatio) *
                                 static_cast<std::size_t>(opts.chunkFrames) +
                             meta.taps - 1;
        fftSize = nextPow2(static_cast<uint32_t>(desired));
    }

    std::cout << "[vulkan_overlap_save] fftSize=" << fftSize << " overlap=" << (meta.taps - 1)
              << " chunkFrames=" << opts.chunkFrames << "\n";

    std::vector<float> output;
    if (!processOverlapSaveBuffer(mono, filterTaps, meta.upsampleRatio, fftSize, opts.chunkFrames,
                                  output)) {
        return 1;
    }

    const uint32_t outputRate = (meta.outputRate != 0)
                                    ? meta.outputRate
                                    : static_cast<uint32_t>(inputRate) * meta.upsampleRatio;

    WavWriter writer;
    if (!writer.open(opts.outputPath, static_cast<int>(outputRate), 1)) {
        return 1;
    }
    AudioFile outFile;
    outFile.channels = 1;
    outFile.sampleRate = static_cast<int>(outputRate);
    outFile.frames = static_cast<int>(output.size());
    outFile.data = std::move(output);
    if (!writer.writeAll(outFile)) {
        writer.close();
        return 1;
    }
    writer.close();
    return 0;
}
