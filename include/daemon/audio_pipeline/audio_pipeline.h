#pragma once

#include "audio/soft_mute.h"
#include "convolution_engine.h"
#include "core/config_loader.h"
#include "core/daemon_constants.h"
#include "daemon/audio_pipeline/streaming_cache_manager.h"
#include "daemon/metrics/runtime_stats.h"
#include "daemon/output/playback_buffer_manager.h"
#include "delimiter/safety_controller.h"
#include "io/audio_ring_buffer.h"
#include "logging/logger.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace delimiter {
class InferenceBackend;
class SafetyController;
}  // namespace delimiter

namespace audio_pipeline {

struct BufferResources {
    daemon_output::PlaybackBufferManager* playbackBuffer = nullptr;
};

struct Upsampler {
    using ProcessFn = std::function<bool(
        const float* inputData, size_t inputFrames,
        ConvolutionEngine::StreamFloatVector& outputData, cudaStream_t stream,
        ConvolutionEngine::StreamFloatVector& streamInputBuffer, size_t& streamInputAccumulated)>;

    ProcessFn process;
    cudaStream_t streamLeft = nullptr;
    cudaStream_t streamRight = nullptr;
    bool available = false;
};

struct OutputState {
    std::atomic<float>* outputGain = nullptr;
    std::atomic<float>* limiterGain = nullptr;
    std::atomic<float>* effectiveGain = nullptr;
};

struct Dependencies {
    const AppConfig* config = nullptr;
    Upsampler upsampler;
    OutputState output;
    std::atomic<bool>* running = nullptr;
    std::atomic<bool>* fallbackActive = nullptr;
    std::atomic<bool>* outputReady = nullptr;
    std::atomic<bool>* crossfeedEnabled = nullptr;
    std::atomic<bool>* crossfeedResetRequested = nullptr;
    ConvolutionEngine::FourChannelFIR* crossfeedProcessor = nullptr;
    ConvolutionEngine::StreamFloatVector* cfStreamInputLeft = nullptr;
    ConvolutionEngine::StreamFloatVector* cfStreamInputRight = nullptr;
    size_t* cfStreamAccumulatedLeft = nullptr;
    size_t* cfStreamAccumulatedRight = nullptr;
    ConvolutionEngine::StreamFloatVector* cfOutputLeft = nullptr;
    ConvolutionEngine::StreamFloatVector* cfOutputRight = nullptr;

    ConvolutionEngine::StreamFloatVector* streamInputLeft = nullptr;
    ConvolutionEngine::StreamFloatVector* streamInputRight = nullptr;
    size_t* streamAccumulatedLeft = nullptr;
    size_t* streamAccumulatedRight = nullptr;
    ConvolutionEngine::StreamFloatVector* upsamplerOutputLeft = nullptr;
    ConvolutionEngine::StreamFloatVector* upsamplerOutputRight = nullptr;

    streaming_cache::StreamingCacheManager* streamingCacheManager = nullptr;
    BufferResources buffer;
    std::function<size_t()> maxOutputBufferFrames;
    std::function<int()> currentInputRate;
    std::function<int()> currentOutputRate;
    std::atomic<int>* delimiterMode = nullptr;
    std::atomic<int>* delimiterFallbackReason = nullptr;
    std::atomic<bool>* delimiterBypassLocked = nullptr;
    std::atomic<bool>* delimiterEnabled = nullptr;
    std::atomic<bool>* delimiterWarmup = nullptr;
    std::atomic<std::size_t>* delimiterQueueSamples = nullptr;
    std::atomic<double>* delimiterQueueSeconds = nullptr;
    std::atomic<double>* delimiterLastInferenceMs = nullptr;
    std::atomic<bool>* delimiterBackendAvailable = nullptr;
    std::atomic<bool>* delimiterBackendValid = nullptr;
    std::atomic<int>* delimiterTargetMode = nullptr;

    // Test hook: override delimiter backend creation (defaults to createDelimiterInferenceBackend).
    std::function<std::unique_ptr<delimiter::InferenceBackend>(const AppConfig::DelimiterConfig&)>
        delimiterBackendFactory;
};

struct DelimiterStatusSnapshot {
    delimiter::ProcessingMode mode = delimiter::ProcessingMode::Active;
    delimiter::ProcessingMode targetMode = delimiter::ProcessingMode::Active;
    delimiter::FallbackReason fallbackReason = delimiter::FallbackReason::None;
    bool bypassLocked = false;
    bool enabled = false;
    bool warmup = false;
    bool backendAvailable = false;
    bool backendValid = false;
    double queueSeconds = 0.0;
    std::size_t queueSamples = 0;
    double lastInferenceMs = 0.0;
    std::string detail;
};

struct RenderResult {
    size_t framesRequested = 0;
    size_t framesRendered = 0;
    bool wroteSilence = false;
};

class AudioPipeline {
   public:
    explicit AudioPipeline(Dependencies deps);
    ~AudioPipeline();
    bool process(const float* inputSamples, uint32_t nFrames);
    void requestRtPause();
    void resumeRtPause();
    bool waitForRtPaused(std::chrono::milliseconds timeout) const;
    bool waitForRtQuiescent(std::chrono::milliseconds timeout) const;
    bool requestDelimiterEnable();
    bool requestDelimiterDisable();
    DelimiterStatusSnapshot delimiterStatus() const;
    RenderResult renderOutput(size_t frames, std::vector<int32_t>& interleavedOut,
                              std::vector<float>& floatScratch, SoftMute::Controller* softMute);
    void trimOutputBuffer(size_t minFramesToRemove);
    const BufferResources& bufferResources() const;

   private:
    bool processDirect(const float* inputSamples, uint32_t nFrames);
    bool enqueueInputForWorker(const float* inputSamples, uint32_t nFrames);
    void workerLoop();
    void resetHighLatencyState(const char* reason);
    void resetHighLatencyStateLocked(const char* reason);
    void updateDelimiterStatus(const delimiter::SafetyStatus& status);
    void applyDelimiterCommand();
    void updateDelimiterTelemetry(bool backendEnabled, bool backendValid, double queueSeconds,
                                  std::size_t queueSamples);
    void recordInferenceDurationMs(double durationMs);

    bool hasBufferState() const;
    bool isUpsamplerAvailable() const;
    bool isOutputReady() const;
    void logDroppingInput();
    void logDroppingHighLatencyInput();
    void trimInternal(size_t minFramesToRemove);
    float computeStereoPeak(const float* left, const float* right, size_t frames) const;
    float applyOutputLimiter(float* interleaved, size_t frames);
    bool startDelimiterBackend();

    template <typename Container>
    size_t enqueueOutputFramesLocked(const Container& left, const Container& right);

    Dependencies deps_;
    std::chrono::steady_clock::time_point lastDropWarn_{std::chrono::steady_clock::now() -
                                                        std::chrono::seconds(6)};
    std::atomic<int> pauseRequestCount_{0};
    std::atomic<bool> rtPaused_{false};
    std::atomic<bool> rtInProcess_{false};
    std::atomic<bool> throttleOutput_{false};
    bool lastCrossfeedEnabledApplied_ = false;

    // RT パスで毎回 std::vector を生成しないためのワークバッファ (Issue #894)
    std::vector<float> workLeft_;
    std::vector<float> workRight_;

    // High-latency worker path (Fix #1010 / Epic #1006)
    bool highLatencyEnabled_ = false;
    std::unique_ptr<delimiter::InferenceBackend> delimiterBackend_;
    std::unique_ptr<delimiter::SafetyController> delimiterSafety_;
    std::thread workerThread_;
    std::atomic<bool> workerStop_{false};
    std::atomic<bool> workerFailed_{false};
    std::atomic<bool> inputDropDetected_{false};
    std::chrono::steady_clock::time_point lastInputDropWarn_{std::chrono::steady_clock::now() -
                                                             std::chrono::seconds(6)};

    std::mutex inputCvMutex_;
    std::condition_variable inputCv_;
    AudioRingBuffer inputInterleaved_;

    // Worker-owned state (single consumer thread)
    int workerInputRate_ = 0;
    size_t chunkFrames_ = 0;
    size_t overlapFrames_ = 0;
    size_t hopFrames_ = 0;
    std::vector<float> fadeIn_;
    std::vector<float> prevInputTailLeft_;
    std::vector<float> prevInputTailRight_;
    std::vector<float> prevOutputTailWeightedLeft_;
    std::vector<float> prevOutputTailWeightedRight_;
    std::vector<float> chunkInputLeft_;
    std::vector<float> chunkInputRight_;
    std::vector<float> chunkOutputLeft_;
    std::vector<float> chunkOutputRight_;
    std::vector<float> readInterleaved_;
    std::vector<float> readLeft_;
    std::vector<float> readRight_;
    std::vector<float> segmentLeft_;
    std::vector<float> segmentRight_;
    std::vector<float> downstreamInterleaved_;
    bool hasPrevChunk_ = false;

    enum class DelimiterCommand { None, Enable, Disable };
    std::atomic<int> delimiterCommand_{static_cast<int>(DelimiterCommand::None)};
    std::atomic<bool> delimiterWarmup_{false};
    std::atomic<std::size_t> delimiterQueueSamples_{0};
    std::atomic<double> delimiterQueueSeconds_{0.0};
    std::atomic<double> delimiterLastInferenceMs_{0.0};
    std::atomic<bool> delimiterBackendAvailable_{false};
    std::atomic<bool> delimiterBackendValid_{false};
    std::atomic<int> delimiterTargetMode_{static_cast<int>(delimiter::ProcessingMode::Active)};
    mutable std::mutex delimiterDetailMutex_;
    std::string delimiterDetail_;
};

template <typename Container>
size_t AudioPipeline::enqueueOutputFramesLocked(const Container& left, const Container& right) {
    if (!hasBufferState() || !deps_.buffer.playbackBuffer) {
        return 0;
    }

    size_t framesAvailable = std::min(left.size(), right.size());
    if (framesAvailable == 0) {
        return 0;
    }

    if (throttleOutput_.load(std::memory_order_acquire) && deps_.running &&
        deps_.buffer.playbackBuffer) {
        deps_.buffer.playbackBuffer->throttleProducerIfFull(*deps_.running, deps_.currentOutputRate,
                                                            framesAvailable);
    }

    int outputRate = DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE;
    if (deps_.currentOutputRate) {
        outputRate = deps_.currentOutputRate();
        if (outputRate <= 0) {
            outputRate = DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE;
        }
    }

    size_t stored = 0;
    size_t dropped = 0;
    if (!deps_.buffer.playbackBuffer->enqueue(left.data(), right.data(), framesAvailable,
                                              outputRate, stored, dropped)) {
        LOG_ERROR("Failed to enqueue output frames into playback buffer");
        return 0;
    }
    if (dropped > 0) {
        runtime_stats::addDroppedFrames(dropped);
    }
    return stored;
}

}  // namespace audio_pipeline
