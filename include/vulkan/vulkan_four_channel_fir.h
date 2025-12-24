#pragma once

#include "convolution_engine.h"
#include "gpu/backend/gpu_backend.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace vulkan_backend {

// Vulkan backend implementation of FourChannelFIR using the generic GPU backend abstraction.
class VulkanFourChannelFIR : public ConvolutionEngine::IFourChannelFIR {
   public:
    VulkanFourChannelFIR();
    ~VulkanFourChannelFIR() override;

    bool initialize(const std::string& hrtfDir, int blockSize = 8192,
                    ConvolutionEngine::HeadSize initialSize = ConvolutionEngine::HeadSize::M,
                    ConvolutionEngine::RateFamily initialFamily =
                        ConvolutionEngine::RateFamily::RATE_44K) override;
    bool initializeStreaming() override;
    void resetStreaming() override;

    bool switchHeadSize(ConvolutionEngine::HeadSize targetSize) override;
    bool switchRateFamily(ConvolutionEngine::RateFamily targetFamily) override;

    bool processStreamBlock(const float* inputL, const float* inputR, size_t inputFrames,
                            ConvolutionEngine::StreamFloatVector& outputL,
                            ConvolutionEngine::StreamFloatVector& outputR, cudaStream_t stream,
                            ConvolutionEngine::StreamFloatVector& streamInputBufferL,
                            ConvolutionEngine::StreamFloatVector& streamInputBufferR,
                            size_t& streamInputAccumulatedL,
                            size_t& streamInputAccumulatedR) override;

    size_t getStreamValidInputPerBlock() const override;
    size_t getValidOutputPerBlock() const override;
    int getFilterTaps() const override;
    ConvolutionEngine::HeadSize getCurrentHeadSize() const override;
    ConvolutionEngine::RateFamily getCurrentRateFamily() const override;
    bool isInitialized() const override;
    void setEnabled(bool enabled) override;
    bool isEnabled() const override;

   private:
    using DeviceBuffer = ConvolutionEngine::GpuBackend::DeviceBuffer;
    using FftPlan = ConvolutionEngine::GpuBackend::FftPlan;

    int getFamilyIndex(ConvolutionEngine::RateFamily family) const;
    int getConfigIndex(ConvolutionEngine::HeadSize size,
                       ConvolutionEngine::RateFamily family) const;
    bool loadHrtfConfig(const std::string& binPath, const std::string& jsonPath,
                        ConvolutionEngine::HeadSize size, ConvolutionEngine::RateFamily family,
                        int expectedTaps);
    bool setupBackendResources();
    void cleanup();

    std::unique_ptr<ConvolutionEngine::GpuBackend::IGpuBackend> backend_;

    // Metadata / configuration
    std::string hrtfDir_;
    int blockSize_ = 0;
    int filterTaps_ = 0;
    int fftSize_ = 0;
    int overlapSize_ = 0;
    int validOutputPerBlock_ = 0;
    size_t streamValidInputPerBlock_ = 0;
    size_t complexCount_ = 0;
    bool initialized_ = false;
    bool streamInitialized_ = false;
    bool enabled_ = true;
    ConvolutionEngine::HeadSize currentHeadSize_ = ConvolutionEngine::HeadSize::M;
    ConvolutionEngine::RateFamily currentRateFamily_ = ConvolutionEngine::RateFamily::RATE_44K;

    static constexpr int kHeadSizeCount = static_cast<int>(ConvolutionEngine::HeadSize::COUNT);
    static constexpr int kChannelCount = 4;
    static constexpr int kRateFamilyCount = 2;
    static constexpr int kConfigCount = kHeadSizeCount * kRateFamilyCount;

    // Host-side coefficients and state
    std::array<std::array<std::vector<float>, kChannelCount>, kConfigCount> h_filterCoeffs_{};
    std::array<bool, kConfigCount> configLoaded_{};
    std::vector<float> overlapInputL_;
    std::vector<float> overlapInputR_;
    std::vector<float> hostPaddedL_;
    std::vector<float> hostPaddedR_;
    std::vector<float> hostTimeWork_;
    ConvolutionEngine::StreamFloatVector stagedOutputL_;
    ConvolutionEngine::StreamFloatVector stagedOutputR_;

    // GPU resources
    DeviceBuffer d_paddedInputL_{};
    DeviceBuffer d_paddedInputR_{};
    DeviceBuffer d_fftInputL_{};
    DeviceBuffer d_fftInputR_{};
    DeviceBuffer d_convLL_{};
    DeviceBuffer d_convLR_{};
    DeviceBuffer d_convRL_{};
    DeviceBuffer d_convRR_{};
    DeviceBuffer d_timeDomain_{};
    DeviceBuffer d_filterPadded_{};
    DeviceBuffer d_filterWork_{};
    std::array<std::array<DeviceBuffer, kChannelCount>, kConfigCount> d_filterFFTs_{};
    std::array<const DeviceBuffer*, kChannelCount> activeFilterFFT_{};
    FftPlan fftPlanForward_{};
    FftPlan fftPlanInverse_{};
};

}  // namespace vulkan_backend
