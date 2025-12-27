#include "vulkan/vulkan_four_channel_fir.h"

namespace vulkan_backend {

VulkanFourChannelFIR::VulkanFourChannelFIR()
    : ConvolutionEngine::GpuBackend::BackendFourChannelFIR(
          ConvolutionEngine::GpuBackend::createVulkanBackend, "VulkanFourChannelFIR") {}

VulkanFourChannelFIR::~VulkanFourChannelFIR() = default;

}  // namespace vulkan_backend
