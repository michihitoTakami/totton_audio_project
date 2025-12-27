#pragma once

#include "gpu/backend/four_channel_fir_backend.h"

namespace vulkan_backend {

// Vulkan backend implementation of FourChannelFIR using the generic GPU backend abstraction.
class VulkanFourChannelFIR : public ConvolutionEngine::GpuBackend::BackendFourChannelFIR {
   public:
    VulkanFourChannelFIR()
        : BackendFourChannelFIR(ConvolutionEngine::GpuBackend::createVulkanBackend,
                                "VulkanFourChannelFIR") {}
    ~VulkanFourChannelFIR() override;
};

}  // namespace vulkan_backend
