#pragma once

#include "convolution_engine.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vulkan_backend {

// ストリーミング向けの最小 Vulkan アップサンプラ
class VulkanStreamingUpsampler : public ConvolutionEngine::IAudioUpsampler {
   public:
    struct InitParams {
        // Legacy: 旧実装との互換のため残す（Minimum 用として扱う）
        std::string filterPath;
        // 新実装: phase 切替のため Minimum/Linear を両方指定できる
        std::string filterPathMinimum;
        std::string filterPathLinear;
        PhaseType initialPhase = PhaseType::Minimum;
        uint32_t upsampleRatio = 0;
        uint32_t blockSize = 0;
        uint32_t inputRate = 0;
        uint32_t fftSizeOverride = 0;
    };

    VulkanStreamingUpsampler();
    ~VulkanStreamingUpsampler() override;

    bool initialize(const InitParams& params);

    // IAudioUpsampler
    void setPartitionedConvolutionConfig(
        const AppConfig::PartitionedConvolutionConfig& /*config*/) override {}
    bool initializeStreaming() override;
    void resetStreaming() override;

    size_t getStreamValidInputPerBlock() const override;
    int getUpsampleRatio() const override;
    int getOutputSampleRate() const override;
    int getInputSampleRate() const override;

    bool isMultiRateEnabled() const override {
        return false;
    }
    int getCurrentInputRate() const override;
    bool switchToInputRate(int inputSampleRate) override;
    PhaseType getPhaseType() const override;
    bool switchPhaseType(PhaseType targetPhase) override;
    size_t getFilterFftSize() const override;
    size_t getFullFftSize() const override;
    bool applyEqMagnitude(const std::vector<double>& eqMagnitude) override;

    bool processStreamBlock(const float* inputData, size_t inputFrames,
                            ConvolutionEngine::StreamFloatVector& outputData, cudaStream_t stream,
                            ConvolutionEngine::StreamFloatVector& streamInputBuffer,
                            size_t& streamInputAccumulated) override;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vulkan_backend
