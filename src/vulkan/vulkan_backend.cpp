#include "gpu/backend/gpu_backend.h"
#include "logging/logger.h"
#include "vkFFT.h"

#include <SPIRV/GlslangToSpv.h>
#include <algorithm>
#include <cstring>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace ConvolutionEngine {
namespace GpuBackend {

namespace {

constexpr uint32_t kVkApiVersion = VK_API_VERSION_1_1;

struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct VulkanStreamHandle {
    int placeholder = 0;  // 将来の非同期対応に備えたダミー
};

struct FftPlanData {
    VkFFTApplication app{};
    VkFFTConfiguration conf{};
    FftDomain domain = FftDomain::RealToComplex;
    int fftSize = 0;
    int batch = 1;
    bool initialized = false;
    VulkanBuffer workBuf{};
};

struct MultiplyPipeline {
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    bool ready = false;
};

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

bool checkVk(VkResult vr, const char* context) {
    if (vr != VK_SUCCESS) {
        LOG_ERROR("[Vulkan backend] {} failed: {}", context, static_cast<int>(vr));
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

bool initContext(VulkanContext& ctx) {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "gpu_backend_vulkan";
    appInfo.applicationVersion = 1;
    appInfo.pEngineName = "totton_audio";
    appInfo.engineVersion = 1;
    appInfo.apiVersion = kVkApiVersion;

    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &appInfo;
    if (!checkVk(vkCreateInstance(&instanceInfo, nullptr, &ctx.instance), "vkCreateInstance")) {
        return false;
    }

    uint32_t deviceCount = 0;
    if (!checkVk(vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr),
                 "vkEnumeratePhysicalDevices") ||
        deviceCount == 0) {
        LOG_ERROR("[Vulkan backend] No Vulkan physical devices found");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data());
    ctx.physicalDevice = devices[0];
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &ctx.physicalProps);

    const uint32_t qfi = findComputeQueueFamily(ctx.physicalDevice);
    if (qfi == UINT32_MAX) {
        LOG_ERROR("[Vulkan backend] No compute queue family found");
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

bool createBuffer(VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags props, VulkanBuffer& out) {
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
    uint32_t memoryType = findMemoryType(ctx.physicalDevice, memReq.memoryTypeBits, props);
    if (memoryType == UINT32_MAX) {
        LOG_ERROR("[Vulkan backend] No suitable memory type found");
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

void destroyBuffer(VulkanContext& ctx, VulkanBuffer& buf) {
    if (buf.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx.device, buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
    }
    if (buf.memory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx.device, buf.memory, nullptr);
        buf.memory = VK_NULL_HANDLE;
    }
    buf.size = 0;
}

VkCommandBuffer allocateCmd(VulkanContext& ctx) {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (!checkVk(vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd),
                 "vkAllocateCommandBuffers")) {
        return VK_NULL_HANDLE;
    }
    return cmd;
}

bool beginCmd(VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return checkVk(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
}

bool submitAndWait(VulkanContext& ctx, VkCommandBuffer cmd) {
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (!checkVk(vkQueueSubmit(ctx.queue, 1, &submit, ctx.fence), "vkQueueSubmit")) {
        return false;
    }
    if (!checkVk(vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX),
                 "vkWaitForFences")) {
        return false;
    }
    vkResetFences(ctx.device, 1, &ctx.fence);
    return true;
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

bool ensureMultiplyPipeline(VulkanContext& ctx, MultiplyPipeline& pipe, const VulkanBuffer& outBuf,
                            const VulkanBuffer& aBuf, const VulkanBuffer& bBuf) {
    if (pipe.ready) {
        return true;
    }
    const char* shaderSrc = R"(
        #version 450
        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
        layout(set = 0, binding = 0) buffer OutBuf { float data[]; } outBuf;
        layout(set = 0, binding = 1) buffer ABuf { float data[]; } aBuf;
        layout(set = 0, binding = 2) buffer BBuf { float data[]; } bBuf;
        layout(push_constant) uniform Params { uint complexCount; float scale; } params;
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= params.complexCount) {
                return;
            }
            uint base = idx * 2;
            float aRe = aBuf.data[base];
            float aIm = aBuf.data[base + 1];
            float bRe = bBuf.data[base];
            float bIm = bBuf.data[base + 1];
            outBuf.data[base]     = (aRe * bRe - aIm * bIm) * params.scale;
            outBuf.data[base + 1] = (aRe * bIm + aIm * bRe) * params.scale;
        }
    )";

    auto spirv = compileComputeShader(shaderSrc);
    if (spirv.empty()) {
        return false;
    }

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = spirv.size() * sizeof(uint32_t);
    moduleInfo.pCode = spirv.data();
    if (!checkVk(vkCreateShaderModule(ctx.device, &moduleInfo, nullptr, &pipe.shaderModule),
                 "vkCreateShaderModule")) {
        return false;
    }

    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    if (!checkVk(vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &pipe.layout),
                 "vkCreateDescriptorSetLayout")) {
        return false;
    }

    VkPushConstantRange push{};
    push.offset = 0;
    push.size = sizeof(uint32_t) + sizeof(float);
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &pipe.layout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &push;
    if (!checkVk(
            vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr, &pipe.pipelineLayout),
            "vkCreatePipelineLayout")) {
        return false;
    }

    VkComputePipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = pipe.shaderModule;
    pipeInfo.stage.pName = "main";
    pipeInfo.layout = pipe.pipelineLayout;
    if (!checkVk(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
                                          &pipe.pipeline),
                 "vkCreateComputePipelines")) {
        return false;
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (!checkVk(vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &pipe.descriptorPool),
                 "vkCreateDescriptorPool")) {
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = pipe.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &pipe.layout;
    if (!checkVk(vkAllocateDescriptorSets(ctx.device, &allocInfo, &pipe.descriptorSet),
                 "vkAllocateDescriptorSets")) {
        return false;
    }

    pipe.ready = true;
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
    pipe.ready = false;
}

class VulkanBackend final : public IGpuBackend {
   public:
    VulkanBackend() {
        initialized_ = initContext(ctx_);
        if (!initialized_) {
            destroyContext(ctx_);
        }
    }

    ~VulkanBackend() override {
        destroyPipeline(ctx_, pipeline_);
        destroyContext(ctx_);
    }

    BackendKind kind() const override {
        return BackendKind::Vulkan;
    }
    const char* name() const override {
        return "vulkan";
    }

    AudioEngine::ErrorCode allocateDeviceBuffer(DeviceBuffer& out, size_t bytes,
                                                const char* /*context*/) override {
        if (!initialized_) {
            return AudioEngine::ErrorCode::GPU_INIT_FAILED;
        }
        auto buf = std::make_unique<VulkanBuffer>();
        if (!createBuffer(
                ctx_, static_cast<VkDeviceSize>(bytes),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, *buf)) {
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        out.handle.ptr = buf.release();
        out.bytes = bytes;
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode freeDeviceBuffer(DeviceBuffer& buffer,
                                            const char* /*context*/) override {
        if (!buffer.handle.ptr) {
            return AudioEngine::ErrorCode::OK;
        }
        auto* buf = static_cast<VulkanBuffer*>(buffer.handle.ptr);
        destroyBuffer(ctx_, *buf);
        delete buf;
        buffer.handle.ptr = nullptr;
        buffer.bytes = 0;
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode createStream(Stream& out, const char* /*context*/) override {
        out.handle.ptr = new VulkanStreamHandle();
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode destroyStream(Stream& stream, const char* /*context*/) override {
        if (stream.handle.ptr) {
            delete static_cast<VulkanStreamHandle*>(stream.handle.ptr);
            stream.handle.ptr = nullptr;
        }
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode streamSynchronize(const Stream* /*stream*/,
                                             const char* /*context*/) override {
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode copy(void* dst, const void* src, size_t bytes, CopyKind kind,
                                const Stream* /*stream*/, const char* /*context*/) override {
        if (!dst || !src || bytes == 0) {
            return AudioEngine::ErrorCode::VALIDATION_INVALID_CONFIG;
        }
        if (!initialized_) {
            return AudioEngine::ErrorCode::GPU_INIT_FAILED;
        }

        auto copyHostToBuffer = [&](const void* host, VulkanBuffer* buf) -> bool {
            void* mapped = nullptr;
            if (!checkVk(vkMapMemory(ctx_.device, buf->memory, 0, buf->size, 0, &mapped),
                         "vkMapMemory H2D")) {
                return false;
            }
            std::memcpy(mapped, host, std::min<uint64_t>(buf->size, bytes));
            vkUnmapMemory(ctx_.device, buf->memory);
            return true;
        };

        auto copyBufferToHost = [&](const VulkanBuffer* buf, void* host) -> bool {
            void* mapped = nullptr;
            if (!checkVk(vkMapMemory(ctx_.device, buf->memory, 0, buf->size, 0, &mapped),
                         "vkMapMemory D2H")) {
                return false;
            }
            std::memcpy(host, mapped, std::min<uint64_t>(buf->size, bytes));
            vkUnmapMemory(ctx_.device, buf->memory);
            return true;
        };

        switch (kind) {
        case CopyKind::HostToDevice: {
            auto* dstBuf = static_cast<VulkanBuffer*>(dst);
            return copyHostToBuffer(src, dstBuf) ? AudioEngine::ErrorCode::OK
                                                 : AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        case CopyKind::DeviceToHost: {
            auto* srcBuf = static_cast<const VulkanBuffer*>(src);
            return copyBufferToHost(srcBuf, dst) ? AudioEngine::ErrorCode::OK
                                                 : AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        case CopyKind::DeviceToDevice: {
            const auto* srcBuf = static_cast<const VulkanBuffer*>(src);
            auto* dstBuf = static_cast<VulkanBuffer*>(dst);
            VkCommandBuffer cmd = allocateCmd(ctx_);
            if (cmd == VK_NULL_HANDLE) {
                return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
            }
            if (!beginCmd(cmd)) {
                vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmd);
                return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
            }
            VkBufferCopy region{};
            region.srcOffset = 0;
            region.dstOffset = 0;
            region.size = std::min<VkDeviceSize>(srcBuf->size, dstBuf->size);
            vkCmdCopyBuffer(cmd, srcBuf->buffer, dstBuf->buffer, 1, &region);
            if (!checkVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer copy")) {
                vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmd);
                return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
            }
            bool ok = submitAndWait(ctx_, cmd);
            vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmd);
            return ok ? AudioEngine::ErrorCode::OK : AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        }
        return AudioEngine::ErrorCode::INTERNAL_UNKNOWN;
    }

    AudioEngine::ErrorCode createFftPlan1d(FftPlan& out, int fftSize, int batch, FftDomain domain,
                                           const char* /*context*/) override {
        if (!initialized_) {
            return AudioEngine::ErrorCode::GPU_INIT_FAILED;
        }
        if (fftSize <= 0 || batch <= 0) {
            return AudioEngine::ErrorCode::VALIDATION_INVALID_CONFIG;
        }
        if (batch != 1) {
            return AudioEngine::ErrorCode::VALIDATION_INVALID_CONFIG;
        }

        auto plan = std::make_unique<FftPlanData>();
        plan->domain = domain;
        plan->fftSize = fftSize;
        plan->batch = batch;
        plan->conf.FFTdim = 1;
        plan->conf.size[0] = static_cast<uint64_t>(fftSize);
        // 実数⇔複素の往復を許容するため、ComplexToReal でも performR2C を有効化する
        plan->conf.performR2C = (domain != FftDomain::ComplexToComplex) ? 1 : 0;
        plan->conf.normalize = 1;
        plan->conf.device = &ctx_.device;
        plan->conf.queue = &ctx_.queue;
        plan->conf.commandPool = &ctx_.commandPool;
        plan->conf.fence = &ctx_.fence;
        plan->conf.physicalDevice = &ctx_.physicalDevice;
        plan->conf.inverseReturnToInputBuffer = 1;
        plan->conf.isCompilerInitialized = 0;
        plan->conf.saveApplicationToString = 0;
        plan->initialized = false;
        plan->workBuf = {};
        out.handle.ptr = plan.release();
        out.fftSize = fftSize;
        out.batch = batch;
        out.domain = domain;
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode destroyFftPlan(FftPlan& plan, const char* /*context*/) override {
        if (!plan.handle.ptr) {
            return AudioEngine::ErrorCode::OK;
        }
        auto* data = static_cast<FftPlanData*>(plan.handle.ptr);
        deleteVkFFT(&data->app);
        if (data->workBuf.buffer) {
            destroyBuffer(ctx_, data->workBuf);
        }
        delete data;
        plan.handle.ptr = nullptr;
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode executeFft(const FftPlan& plan, const DeviceBuffer& in,
                                      DeviceBuffer& out, FftDirection direction,
                                      const Stream* /*stream*/, const char* /*context*/) override {
        if (!initialized_ || !plan.handle.ptr || !in.handle.ptr || !out.handle.ptr) {
            return AudioEngine::ErrorCode::VALIDATION_INVALID_CONFIG;
        }
        auto* data = static_cast<FftPlanData*>(plan.handle.ptr);
        auto* inBuf = static_cast<VulkanBuffer*>(in.handle.ptr);
        auto* outBuf = static_cast<VulkanBuffer*>(out.handle.ptr);

        const size_t realBytes = static_cast<size_t>(plan.fftSize) * sizeof(float);
        const size_t complexBytes = static_cast<size_t>(plan.fftSize / 2 + 1) * 2 * sizeof(float);
        const size_t complexFullBytes = static_cast<size_t>(plan.fftSize) * 2 * sizeof(float);

        size_t expectedInBytes = 0;
        size_t expectedOutBytes = 0;
        switch (plan.domain) {
        case FftDomain::RealToComplex:
            if (direction == FftDirection::Forward) {
                expectedInBytes = realBytes;
                expectedOutBytes = complexBytes;
            } else {
                expectedInBytes = complexBytes;
                expectedOutBytes = realBytes;
            }
            break;
        case FftDomain::ComplexToReal:
            if (direction == FftDirection::Forward) {
                expectedInBytes = complexBytes;
                expectedOutBytes = realBytes;
            } else {
                expectedInBytes = realBytes;
                expectedOutBytes = complexBytes;
            }
            break;
        case FftDomain::ComplexToComplex:
            expectedInBytes = complexFullBytes;
            expectedOutBytes = complexFullBytes;
            break;
        }

        if (inBuf->size < expectedInBytes || outBuf->size < expectedOutBytes) {
            LOG_ERROR(
                "[Vulkan backend] FFT buffer size mismatch domain={} dir={} in={}/{} out={}/{}",
                static_cast<int>(plan.domain), static_cast<int>(direction), inBuf->size,
                expectedInBytes, outBuf->size, expectedOutBytes);
            return AudioEngine::ErrorCode::VALIDATION_INVALID_CONFIG;
        }

        size_t workBytes = std::max(expectedInBytes, expectedOutBytes);
        if (!data->workBuf.buffer || data->workBuf.size < workBytes) {
            if (data->workBuf.buffer) {
                destroyBuffer(ctx_, data->workBuf);
            }
            if (!createBuffer(
                    ctx_, static_cast<VkDeviceSize>(workBytes),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    data->workBuf)) {
                return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
            }
            data->initialized = false;
        }

        size_t copyInBytes = expectedInBytes;
        VkCommandBuffer cmdCopy = allocateCmd(ctx_);
        if (cmdCopy == VK_NULL_HANDLE) {
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        if (!beginCmd(cmdCopy)) {
            vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmdCopy);
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        VkBufferCopy regionIn{};
        regionIn.size = static_cast<VkDeviceSize>(copyInBytes);
        vkCmdCopyBuffer(cmdCopy, inBuf->buffer, data->workBuf.buffer, 1, &regionIn);
        if (!checkVk(vkEndCommandBuffer(cmdCopy), "vkEndCommandBuffer fft prep copy")) {
            vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmdCopy);
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        bool okCopy = submitAndWait(ctx_, cmdCopy);
        vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmdCopy);
        if (!okCopy) {
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }

        data->conf.buffer = &data->workBuf.buffer;
        data->conf.bufferSize = &data->workBuf.size;
        data->conf.outputBuffer = &data->workBuf.buffer;
        data->conf.outputBufferSize = &data->workBuf.size;
        data->conf.inputBuffer = &data->workBuf.buffer;
        data->conf.inputBufferSize = &data->workBuf.size;

        if (!data->initialized) {
            deleteVkFFT(&data->app);
            data->app = {};
            VkFFTResult initRes = initializeVkFFT(&data->app, data->conf);
            if (initRes != VKFFT_SUCCESS) {
                LOG_ERROR("[Vulkan backend] Failed to initialize VkFFT: {}",
                          static_cast<int>(initRes));
                return AudioEngine::ErrorCode::GPU_INIT_FAILED;
            }
            data->initialized = true;
        }

        VkCommandBuffer cmd = allocateCmd(ctx_);
        if (cmd == VK_NULL_HANDLE) {
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        if (!beginCmd(cmd)) {
            vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmd);
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        VkFFTLaunchParams launch{};
        launch.commandBuffer = &cmd;
        int dir = (direction == FftDirection::Forward) ? -1 : 1;
        VkFFTResult res = VkFFTAppend(&data->app, dir, &launch);
        if (res != VKFFT_SUCCESS) {
            LOG_ERROR("[Vulkan backend] VkFFTAppend failed: {}", static_cast<int>(res));
            vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmd);
            return AudioEngine::ErrorCode::GPU_CUFFT_ERROR;
        }
        if (!checkVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer fft")) {
            vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmd);
            return AudioEngine::ErrorCode::GPU_CUFFT_ERROR;
        }
        bool ok = submitAndWait(ctx_, cmd);
        vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmd);
        if (!ok) {
            return AudioEngine::ErrorCode::GPU_CUFFT_ERROR;
        }

        size_t copyOutBytes = expectedOutBytes;
        VkCommandBuffer cmdOut = allocateCmd(ctx_);
        if (cmdOut == VK_NULL_HANDLE) {
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        if (!beginCmd(cmdOut)) {
            vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmdOut);
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        VkBufferCopy regionOut{};
        regionOut.size = static_cast<VkDeviceSize>(copyOutBytes);
        vkCmdCopyBuffer(cmdOut, data->workBuf.buffer, outBuf->buffer, 1, &regionOut);
        if (!checkVk(vkEndCommandBuffer(cmdOut), "vkEndCommandBuffer fft copy out")) {
            vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmdOut);
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        bool okOut = submitAndWait(ctx_, cmdOut);
        vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmdOut);
        return okOut ? AudioEngine::ErrorCode::OK : AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
    }

    AudioEngine::ErrorCode complexPointwiseMulScale(DeviceBuffer& out, const DeviceBuffer& a,
                                                    const DeviceBuffer& b, size_t complexCount,
                                                    float scale, const Stream* /*stream*/,
                                                    const char* /*context*/) override {
        if (!initialized_) {
            return AudioEngine::ErrorCode::GPU_INIT_FAILED;
        }
        if (!out.handle.ptr || !a.handle.ptr || !b.handle.ptr) {
            return AudioEngine::ErrorCode::VALIDATION_INVALID_CONFIG;
        }
        auto* outBuf = static_cast<VulkanBuffer*>(out.handle.ptr);
        auto* aBuf = static_cast<VulkanBuffer*>(a.handle.ptr);
        auto* bBuf = static_cast<VulkanBuffer*>(b.handle.ptr);
        size_t requiredBytes = complexCount * 2 * sizeof(float);
        if (outBuf->size < requiredBytes || aBuf->size < requiredBytes ||
            bBuf->size < requiredBytes) {
            return AudioEngine::ErrorCode::VALIDATION_INVALID_CONFIG;
        }

        if (!ensureMultiplyPipeline(ctx_, pipeline_, *outBuf, *aBuf, *bBuf)) {
            return AudioEngine::ErrorCode::GPU_INIT_FAILED;
        }

        // デスクリプタは呼び出し毎に最新のバッファへ更新する
        VkDescriptorBufferInfo outInfo{outBuf->buffer, 0, outBuf->size};
        VkDescriptorBufferInfo aInfo{aBuf->buffer, 0, aBuf->size};
        VkDescriptorBufferInfo bInfo{bBuf->buffer, 0, bBuf->size};
        VkWriteDescriptorSet writes[3]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = pipeline_.descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &outInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = pipeline_.descriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &aInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = pipeline_.descriptorSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &bInfo;
        vkUpdateDescriptorSets(ctx_.device, 3, writes, 0, nullptr);

        VkCommandBuffer cmd = allocateCmd(ctx_);
        if (cmd == VK_NULL_HANDLE) {
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        if (!beginCmd(cmd)) {
            vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmd);
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.pipelineLayout, 0, 1,
                                &pipeline_.descriptorSet, 0, nullptr);
        struct Push {
            uint32_t complexCount;
            float scale;
        } push{static_cast<uint32_t>(complexCount), scale};
        vkCmdPushConstants(cmd, pipeline_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(Push), &push);
        const uint32_t groupCount = (static_cast<uint32_t>(complexCount) + 255u) / 256u;
        vkCmdDispatch(cmd, groupCount, 1, 1);

        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                             nullptr);

        if (!checkVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer mul")) {
            vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmd);
            return AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
        }
        bool ok = submitAndWait(ctx_, cmd);
        vkFreeCommandBuffers(ctx_.device, ctx_.commandPool, 1, &cmd);
        return ok ? AudioEngine::ErrorCode::OK : AudioEngine::ErrorCode::GPU_MEMORY_ERROR;
    }

   private:
    VulkanContext ctx_{};
    bool initialized_ = false;
    MultiplyPipeline pipeline_{};
};

}  // namespace

std::unique_ptr<IGpuBackend> createVulkanBackend() {
    auto backend = std::make_unique<VulkanBackend>();
    return backend;
}

}  // namespace GpuBackend
}  // namespace ConvolutionEngine
