#include "gpu/backend/gpu_backend.h"
#include "logging/logger.h"

#include <memory>

namespace ConvolutionEngine {
namespace GpuBackend {

namespace {

class VulkanBackendStub final : public IGpuBackend {
   public:
    BackendKind kind() const override {
        return BackendKind::Vulkan;
    }
    const char* name() const override {
        return "vulkan_stub";
    }

    AudioEngine::ErrorCode allocateDeviceBuffer(DeviceBuffer& /*out*/, size_t /*bytes*/,
                                                const char* context) override {
        LOG_ERROR("[Vulkan backend] {}: not implemented yet", context);
        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }
    AudioEngine::ErrorCode freeDeviceBuffer(DeviceBuffer& /*buffer*/,
                                            const char* /*context*/) override {
        return AudioEngine::ErrorCode::OK;
    }

    AudioEngine::ErrorCode createStream(Stream& /*out*/, const char* context) override {
        LOG_ERROR("[Vulkan backend] {}: not implemented yet", context);
        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }
    AudioEngine::ErrorCode destroyStream(Stream& /*stream*/, const char* /*context*/) override {
        return AudioEngine::ErrorCode::OK;
    }
    AudioEngine::ErrorCode streamSynchronize(const Stream* /*stream*/,
                                             const char* context) override {
        LOG_ERROR("[Vulkan backend] {}: not implemented yet", context);
        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }

    AudioEngine::ErrorCode copy(void* /*dst*/, const void* /*src*/, size_t /*bytes*/,
                                CopyKind /*kind*/, const Stream* /*stream*/,
                                const char* context) override {
        LOG_ERROR("[Vulkan backend] {}: not implemented yet", context);
        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }

    AudioEngine::ErrorCode createFftPlan1d(FftPlan& /*out*/, int /*fftSize*/, int /*batch*/,
                                           FftDomain /*domain*/, const char* context) override {
        LOG_ERROR("[Vulkan backend] {}: not implemented yet", context);
        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }
    AudioEngine::ErrorCode destroyFftPlan(FftPlan& /*plan*/, const char* /*context*/) override {
        return AudioEngine::ErrorCode::OK;
    }
    AudioEngine::ErrorCode executeFft(const FftPlan& /*plan*/, const DeviceBuffer& /*in*/,
                                      DeviceBuffer& /*out*/, FftDirection /*direction*/,
                                      const Stream* /*stream*/, const char* context) override {
        LOG_ERROR("[Vulkan backend] {}: not implemented yet", context);
        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }

    AudioEngine::ErrorCode complexPointwiseMulScale(DeviceBuffer& /*out*/,
                                                    const DeviceBuffer& /*a*/,
                                                    const DeviceBuffer& /*b*/,
                                                    size_t /*complexCount*/, float /*scale*/,
                                                    const Stream* /*stream*/,
                                                    const char* context) override {
        LOG_ERROR("[Vulkan backend] {}: not implemented yet", context);
        return AudioEngine::ErrorCode::NOT_IMPLEMENTED;
    }
};

}  // namespace

std::unique_ptr<IGpuBackend> createVulkanBackend() {
    LOG_WARN("[Vulkan backend] Stub implementation active (NOT_IMPLEMENTED)");
    return std::make_unique<VulkanBackendStub>();
}

}  // namespace GpuBackend
}  // namespace ConvolutionEngine
