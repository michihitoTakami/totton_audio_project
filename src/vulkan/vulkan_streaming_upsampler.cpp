#include "vulkan/vulkan_streaming_upsampler.h"

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
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

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

}  // namespace

namespace vulkan_backend {

struct VulkanStreamingUpsampler::Impl {
    VulkanContext ctx{};
    BufferResource inputBuf{};
    BufferResource filterBuf{};
    VkFFTApplication inputApp{};
    VkFFTApplication filterApp{};
    MultiplyPipeline pipeline{};

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

    bool initialized = false;

    void cleanup() {
        destroyPipeline(ctx, pipeline);
        deleteVkFFT(&filterApp);
        deleteVkFFT(&inputApp);
        destroyBuffer(ctx, filterBuf);
        destroyBuffer(ctx, inputBuf);
        destroyContext(ctx);
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

    if (params.filterPath.empty() || params.upsampleRatio == 0 || params.blockSize == 0 ||
        params.inputRate == 0) {
        LOG_ERROR("Vulkan upsampler: invalid params (filterPath empty or zero ratio/block/input)");
        return false;
    }

    std::vector<float> filterTaps;
    const std::size_t taps = readFilterTaps(params.filterPath, filterTaps);
    if (taps == 0) {
        return false;
    }

    impl_->upsampleRatio = params.upsampleRatio;
    impl_->inputRate = params.inputRate;
    impl_->outputRate = params.inputRate * params.upsampleRatio;
    impl_->overlap = static_cast<uint32_t>(taps - 1);

    uint32_t fftSize = params.fftSizeOverride;
    if (fftSize == 0) {
        const auto desired = static_cast<std::size_t>(params.upsampleRatio) *
                                 static_cast<std::size_t>(params.blockSize) +
                             taps - 1;
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

    // Upload filter taps (time domain) and run forward FFT
    {
        void* mapped = nullptr;
        if (!checkVk(vkMapMemory(impl_->ctx.device, impl_->filterBuf.memory, 0, impl_->bufferBytes,
                                 0, &mapped),
                     "vkMapMemory filter")) {
            impl_->cleanup();
            return false;
        }
        std::memset(mapped, 0, static_cast<std::size_t>(impl_->bufferBytes));
        std::memcpy(mapped, filterTaps.data(), filterTaps.size() * sizeof(float));
        vkUnmapMemory(impl_->ctx.device, impl_->filterBuf.memory);
    }

    if (!runFftOnBuffer(impl_->ctx, impl_->filterApp, -1)) {
        impl_->cleanup();
        return false;
    }

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
    LOG_INFO("[Vulkan] upsampler ready (ratio={}x, block={} frames, fftSize={})",
             impl_->upsampleRatio, params.blockSize, impl_->fftSize);
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

int VulkanStreamingUpsampler::getCurrentInputRate() const {
    return getInputSampleRate();
}

bool VulkanStreamingUpsampler::switchToInputRate(int inputSampleRate) {
    if (!impl_) {
        return false;
    }
    return inputSampleRate == static_cast<int>(impl_->inputRate);
}

bool VulkanStreamingUpsampler::switchPhaseType(PhaseType targetPhase) {
    return targetPhase == PhaseType::Minimum;
}

size_t VulkanStreamingUpsampler::getFilterFftSize() const {
    return impl_ ? static_cast<size_t>(impl_->fftSize) : 0;
}

size_t VulkanStreamingUpsampler::getFullFftSize() const {
    return impl_ ? static_cast<size_t>(impl_->fftSize) : 0;
}

bool VulkanStreamingUpsampler::applyEqMagnitude(const std::vector<double>& eqMagnitude) {
    (void)eqMagnitude;
    LOG_WARN("[Vulkan] applyEqMagnitude is not supported for streaming backend");
    return false;
}

bool VulkanStreamingUpsampler::processStreamBlock(
    const float* inputData, size_t inputFrames, ConvolutionEngine::StreamFloatVector& outputData,
    cudaStream_t /*stream*/, ConvolutionEngine::StreamFloatVector& streamInputBuffer,
    size_t& streamInputAccumulated) {
    if (!impl_ || !impl_->initialized) {
        return false;
    }
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
