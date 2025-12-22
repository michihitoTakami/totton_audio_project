#include "audio/overlap_add.h"
#include "convolution_engine.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/output/playback_buffer_manager.h"
#include "delimiter/inference_backend.h"
#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

class AlternatingGainBackend final : public delimiter::InferenceBackend {
   public:
    explicit AlternatingGainBackend(uint32_t expectedSampleRate)
        : expectedSampleRate_(expectedSampleRate) {}

    const char* name() const override {
        return "test_alternating_gain";
    }

    uint32_t expectedSampleRate() const override {
        return expectedSampleRate_;
    }

    delimiter::InferenceResult process(const delimiter::StereoPlanarView& input,
                                       std::vector<float>& outLeft,
                                       std::vector<float>& outRight) override {
        if (!input.valid() || input.frames == 0) {
            outLeft.clear();
            outRight.clear();
            return {delimiter::InferenceStatus::InvalidConfig, "invalid input"};
        }

        float gain = (callCount_ == 0) ? 0.0f : 1.0f;
        ++callCount_;

        outLeft.resize(input.frames);
        outRight.resize(input.frames);
        for (std::size_t i = 0; i < input.frames; ++i) {
            outLeft[i] = input.left[i] * gain;
            outRight[i] = input.right[i] * gain;
        }
        return {delimiter::InferenceStatus::Ok, ""};
    }

    void reset() override {
        callCount_ = 0;
    }

   private:
    uint32_t expectedSampleRate_{44100};
    int callCount_ = 0;
};

class ConstantGainBackend final : public delimiter::InferenceBackend {
   public:
    ConstantGainBackend(uint32_t expectedSampleRate, float gain)
        : expectedSampleRate_(expectedSampleRate), gain_(gain) {}

    const char* name() const override {
        return "test_constant_gain";
    }

    uint32_t expectedSampleRate() const override {
        return expectedSampleRate_;
    }

    delimiter::InferenceResult process(const delimiter::StereoPlanarView& input,
                                       std::vector<float>& outLeft,
                                       std::vector<float>& outRight) override {
        if (!input.valid() || input.frames == 0) {
            return {delimiter::InferenceStatus::InvalidConfig, "invalid input"};
        }
        outLeft.resize(input.frames);
        outRight.resize(input.frames);
        for (std::size_t i = 0; i < input.frames; ++i) {
            outLeft[i] = input.left[i] * gain_;
            outRight[i] = input.right[i] * gain_;
        }
        return {delimiter::InferenceStatus::Ok, ""};
    }

    void reset() override {}

   private:
    uint32_t expectedSampleRate_{44100};
    float gain_ = 1.0f;
};

class FailingBackend final : public delimiter::InferenceBackend {
   public:
    explicit FailingBackend(uint32_t expectedSampleRate)
        : expectedSampleRate_(expectedSampleRate) {}

    const char* name() const override {
        return "test_fail";
    }

    uint32_t expectedSampleRate() const override {
        return expectedSampleRate_;
    }

    delimiter::InferenceResult process(const delimiter::StereoPlanarView& input,
                                       std::vector<float>& outLeft,
                                       std::vector<float>& outRight) override {
        (void)input;
        outLeft.clear();
        outRight.clear();
        return {delimiter::InferenceStatus::Error, "intentional failure"};
    }

    void reset() override {}

   private:
    uint32_t expectedSampleRate_{44100};
};

bool waitForQueuedFrames(daemon_output::PlaybackBufferManager& buffer, std::size_t frames,
                         std::chrono::milliseconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    std::unique_lock<std::mutex> lock(buffer.mutex());
    while (buffer.queuedFramesLocked() < frames) {
        if (buffer.cv().wait_until(lock, deadline) == std::cv_status::timeout) {
            break;
        }
    }
    return buffer.queuedFramesLocked() >= frames;
}

}  // namespace

TEST(AudioPipelineHighLatency, OutputsAfterInitialChunkAndCrossfades) {
    AppConfig config;
    config.upsampleRatio = 1;
    config.delimiter.enabled = true;
    config.delimiter.backend = "bypass";
    config.delimiter.expectedSampleRate = 1000;
    config.delimiter.chunkSec = 0.020f;    // 20 frames @ 1000Hz
    config.delimiter.overlapSec = 0.005f;  // 5 frames @ 1000Hz

    std::atomic<bool> running{true};
    std::atomic<bool> fallbackActive{false};
    std::atomic<bool> outputReady{true};
    std::atomic<bool> crossfeedEnabled{false};
    std::atomic<bool> crossfeedResetRequested{false};

    constexpr std::size_t kBufferFrames = 4096;
    daemon_output::PlaybackBufferManager playbackBuffer([]() { return kBufferFrames; });

    ConvolutionEngine::StreamFloatVector streamInputLeft;
    ConvolutionEngine::StreamFloatVector streamInputRight;
    std::size_t streamAccumLeft = 0;
    std::size_t streamAccumRight = 0;
    ConvolutionEngine::StreamFloatVector upsamplerOutputLeft;
    ConvolutionEngine::StreamFloatVector upsamplerOutputRight;

    ConvolutionEngine::StreamFloatVector cfStreamInputLeft;
    ConvolutionEngine::StreamFloatVector cfStreamInputRight;
    std::size_t cfAccumLeft = 0;
    std::size_t cfAccumRight = 0;
    ConvolutionEngine::StreamFloatVector cfOutputLeft;
    ConvolutionEngine::StreamFloatVector cfOutputRight;

    audio_pipeline::Dependencies deps{};
    deps.config = &config;
    deps.running = &running;
    deps.fallbackActive = &fallbackActive;
    deps.outputReady = &outputReady;
    deps.crossfeedEnabled = &crossfeedEnabled;
    deps.crossfeedResetRequested = &crossfeedResetRequested;
    deps.crossfeedProcessor = nullptr;
    deps.cfStreamInputLeft = &cfStreamInputLeft;
    deps.cfStreamInputRight = &cfStreamInputRight;
    deps.cfStreamAccumulatedLeft = &cfAccumLeft;
    deps.cfStreamAccumulatedRight = &cfAccumRight;
    deps.cfOutputLeft = &cfOutputLeft;
    deps.cfOutputRight = &cfOutputRight;
    deps.streamInputLeft = &streamInputLeft;
    deps.streamInputRight = &streamInputRight;
    deps.streamAccumulatedLeft = &streamAccumLeft;
    deps.streamAccumulatedRight = &streamAccumRight;
    deps.upsamplerOutputLeft = &upsamplerOutputLeft;
    deps.upsamplerOutputRight = &upsamplerOutputRight;
    deps.streamingCacheManager = nullptr;
    deps.buffer.playbackBuffer = &playbackBuffer;
    deps.maxOutputBufferFrames = []() { return kBufferFrames; };
    deps.currentInputRate = []() { return 1000; };
    deps.currentOutputRate = []() { return 1000; };
    deps.delimiterBackendFactory = [](const AppConfig::DelimiterConfig& cfg) {
        return std::make_unique<AlternatingGainBackend>(cfg.expectedSampleRate);
    };

    deps.upsampler.available = true;
    deps.upsampler.streamLeft = nullptr;
    deps.upsampler.streamRight = nullptr;
    deps.upsampler.process = [](const float* inputData, std::size_t inputFrames,
                                ConvolutionEngine::StreamFloatVector& outputData,
                                cudaStream_t /*stream*/,
                                ConvolutionEngine::StreamFloatVector& /*streamInput*/,
                                std::size_t& /*streamAccumulated*/) {
        outputData.assign(inputData, inputData + inputFrames);
        return true;
    };

    audio_pipeline::AudioPipeline pipeline(std::move(deps));

    const std::size_t chunkFrames = static_cast<std::size_t>(
        std::lround(config.delimiter.chunkSec * config.delimiter.expectedSampleRate));
    const std::size_t overlapFrames = static_cast<std::size_t>(
        std::lround(config.delimiter.overlapSec * config.delimiter.expectedSampleRate));
    const std::size_t hopFrames = chunkFrames - overlapFrames;

    // Chunk #1 (backend gain=0): output segment should be all zeros.
    std::vector<float> inputChunk(chunkFrames * 2, 0.0f);
    for (std::size_t i = 0; i < chunkFrames; ++i) {
        inputChunk[i * 2] = 1.0f;
        inputChunk[i * 2 + 1] = -1.0f;
    }
    ASSERT_TRUE(pipeline.process(inputChunk.data(), static_cast<uint32_t>(chunkFrames)));

    ASSERT_TRUE(waitForQueuedFrames(playbackBuffer, hopFrames, std::chrono::milliseconds(1000)));
    std::vector<float> outLeft1(hopFrames);
    std::vector<float> outRight1(hopFrames);
    ASSERT_TRUE(playbackBuffer.readPlanar(outLeft1.data(), outRight1.data(), hopFrames));
    for (std::size_t i = 0; i < hopFrames; ++i) {
        EXPECT_FLOAT_EQ(outLeft1[i], 0.0f);
        EXPECT_FLOAT_EQ(outRight1[i], 0.0f);
    }

    // Chunk #2 input provides hop frames; worker reuses previous overlap input internally.
    // Backend gain=1: overlap region should fade-in from 0 to 1.
    std::vector<float> inputHop(hopFrames * 2, 0.0f);
    for (std::size_t i = 0; i < hopFrames; ++i) {
        inputHop[i * 2] = 1.0f;
        inputHop[i * 2 + 1] = -1.0f;
    }
    ASSERT_TRUE(pipeline.process(inputHop.data(), static_cast<uint32_t>(hopFrames)));

    ASSERT_TRUE(waitForQueuedFrames(playbackBuffer, hopFrames, std::chrono::milliseconds(1000)));
    std::vector<float> outLeft2(hopFrames);
    std::vector<float> outRight2(hopFrames);
    ASSERT_TRUE(playbackBuffer.readPlanar(outLeft2.data(), outRight2.data(), hopFrames));

    std::vector<float> fadeIn = AudioUtils::makeRaisedCosineFade(overlapFrames);
    for (std::size_t i = 0; i < overlapFrames; ++i) {
        EXPECT_NEAR(outLeft2[i], fadeIn[i], 1e-5f);
        EXPECT_NEAR(outRight2[i], -fadeIn[i], 1e-5f);
    }
    for (std::size_t i = overlapFrames; i < hopFrames; ++i) {
        EXPECT_FLOAT_EQ(outLeft2[i], 1.0f);
        EXPECT_FLOAT_EQ(outRight2[i], -1.0f);
    }

    running.store(false, std::memory_order_release);
}

TEST(AudioPipelineHighLatency, RunsInferenceWithSampleRateResample) {
    AppConfig config;
    config.upsampleRatio = 1;
    config.delimiter.enabled = true;
    config.delimiter.backend = "ort";
    config.delimiter.expectedSampleRate = 44100;
    config.delimiter.chunkSec = 0.020f;    // 20ms @ 48k -> 960 frames
    config.delimiter.overlapSec = 0.005f;  // 5ms @ 48k -> 240 frames

    std::atomic<bool> running{true};
    std::atomic<bool> fallbackActive{false};
    std::atomic<bool> outputReady{true};
    std::atomic<bool> crossfeedEnabled{false};
    std::atomic<bool> crossfeedResetRequested{false};

    constexpr std::size_t kBufferFrames = 4096;
    daemon_output::PlaybackBufferManager playbackBuffer([]() { return kBufferFrames; });

    ConvolutionEngine::StreamFloatVector streamInputLeft;
    ConvolutionEngine::StreamFloatVector streamInputRight;
    std::size_t streamAccumLeft = 0;
    std::size_t streamAccumRight = 0;
    ConvolutionEngine::StreamFloatVector upsamplerOutputLeft;
    ConvolutionEngine::StreamFloatVector upsamplerOutputRight;

    ConvolutionEngine::StreamFloatVector cfStreamInputLeft;
    ConvolutionEngine::StreamFloatVector cfStreamInputRight;
    std::size_t cfAccumLeft = 0;
    std::size_t cfAccumRight = 0;
    ConvolutionEngine::StreamFloatVector cfOutputLeft;
    ConvolutionEngine::StreamFloatVector cfOutputRight;

    std::atomic<int> delimiterMode(static_cast<int>(delimiter::ProcessingMode::Active));
    std::atomic<int> delimiterReason(static_cast<int>(delimiter::FallbackReason::None));
    std::atomic<bool> delimiterLocked(false);

    audio_pipeline::Dependencies deps{};
    deps.config = &config;
    deps.running = &running;
    deps.fallbackActive = &fallbackActive;
    deps.outputReady = &outputReady;
    deps.crossfeedEnabled = &crossfeedEnabled;
    deps.crossfeedResetRequested = &crossfeedResetRequested;
    deps.crossfeedProcessor = nullptr;
    deps.cfStreamInputLeft = &cfStreamInputLeft;
    deps.cfStreamInputRight = &cfStreamInputRight;
    deps.cfStreamAccumulatedLeft = &cfAccumLeft;
    deps.cfStreamAccumulatedRight = &cfAccumRight;
    deps.cfOutputLeft = &cfOutputLeft;
    deps.cfOutputRight = &cfOutputRight;
    deps.streamInputLeft = &streamInputLeft;
    deps.streamInputRight = &streamInputRight;
    deps.streamAccumulatedLeft = &streamAccumLeft;
    deps.streamAccumulatedRight = &streamAccumRight;
    deps.upsamplerOutputLeft = &upsamplerOutputLeft;
    deps.upsamplerOutputRight = &upsamplerOutputRight;
    deps.streamingCacheManager = nullptr;
    deps.buffer.playbackBuffer = &playbackBuffer;
    deps.maxOutputBufferFrames = []() { return kBufferFrames; };
    deps.currentInputRate = []() { return 48000; };
    deps.currentOutputRate = []() { return 48000; };
    deps.delimiterMode = &delimiterMode;
    deps.delimiterFallbackReason = &delimiterReason;
    deps.delimiterBypassLocked = &delimiterLocked;
    deps.delimiterBackendFactory = [](const AppConfig::DelimiterConfig& cfg) {
        return std::make_unique<ConstantGainBackend>(44100, 2.0f);
    };

    deps.upsampler.available = true;
    deps.upsampler.streamLeft = nullptr;
    deps.upsampler.streamRight = nullptr;
    deps.upsampler.process = [](const float* inputData, std::size_t inputFrames,
                                ConvolutionEngine::StreamFloatVector& outputData,
                                cudaStream_t /*stream*/,
                                ConvolutionEngine::StreamFloatVector& /*streamInput*/,
                                std::size_t& /*streamAccumulated*/) {
        outputData.assign(inputData, inputData + inputFrames);
        return true;
    };

    audio_pipeline::AudioPipeline pipeline(std::move(deps));

    const std::size_t chunkFrames =
        static_cast<std::size_t>(std::lround(config.delimiter.chunkSec * 48000));
    const std::size_t overlapFrames =
        static_cast<std::size_t>(std::lround(config.delimiter.overlapSec * 48000));
    const std::size_t hopFrames = chunkFrames - overlapFrames;

    std::vector<float> input(chunkFrames * 2, 0.0f);
    for (std::size_t i = 0; i < chunkFrames; ++i) {
        input[i * 2] = 1.0f;
        input[i * 2 + 1] = 1.0f;
    }

    ASSERT_TRUE(pipeline.process(input.data(), static_cast<uint32_t>(chunkFrames)));
    ASSERT_TRUE(waitForQueuedFrames(playbackBuffer, hopFrames, std::chrono::milliseconds(1000)));

    std::vector<float> outLeft(hopFrames);
    std::vector<float> outRight(hopFrames);
    ASSERT_TRUE(playbackBuffer.readPlanar(outLeft.data(), outRight.data(), hopFrames));

    for (std::size_t i = 0; i < hopFrames; ++i) {
        EXPECT_NEAR(outLeft[i], 2.0f, 1e-3f);
        EXPECT_NEAR(outRight[i], 2.0f, 1e-3f);
    }
    EXPECT_EQ(static_cast<delimiter::ProcessingMode>(delimiterMode.load()),
              delimiter::ProcessingMode::Active);
    EXPECT_EQ(static_cast<delimiter::FallbackReason>(delimiterReason.load()),
              delimiter::FallbackReason::None);
    EXPECT_FALSE(delimiterLocked.load());

    running.store(false, std::memory_order_release);
}

TEST(AudioPipelineHighLatency, FallbackTriggersBypassLockOnRepeatedFailures) {
    AppConfig config;
    config.upsampleRatio = 1;
    config.delimiter.enabled = true;
    config.delimiter.backend = "ort";
    config.delimiter.expectedSampleRate = 1000;
    config.delimiter.chunkSec = 0.010f;  // shorter to speed up test
    config.delimiter.overlapSec = 0.002f;

    std::atomic<bool> running{true};
    std::atomic<bool> fallbackActive{false};
    std::atomic<bool> outputReady{true};
    std::atomic<bool> crossfeedEnabled{false};
    std::atomic<bool> crossfeedResetRequested{false};

    constexpr std::size_t kBufferFrames = 4096;
    daemon_output::PlaybackBufferManager playbackBuffer([]() { return kBufferFrames; });

    ConvolutionEngine::StreamFloatVector streamInputLeft;
    ConvolutionEngine::StreamFloatVector streamInputRight;
    std::size_t streamAccumLeft = 0;
    std::size_t streamAccumRight = 0;
    ConvolutionEngine::StreamFloatVector upsamplerOutputLeft;
    ConvolutionEngine::StreamFloatVector upsamplerOutputRight;

    ConvolutionEngine::StreamFloatVector cfStreamInputLeft;
    ConvolutionEngine::StreamFloatVector cfStreamInputRight;
    std::size_t cfAccumLeft = 0;
    std::size_t cfAccumRight = 0;
    ConvolutionEngine::StreamFloatVector cfOutputLeft;
    ConvolutionEngine::StreamFloatVector cfOutputRight;

    std::atomic<int> delimiterMode(static_cast<int>(delimiter::ProcessingMode::Active));
    std::atomic<int> delimiterReason(static_cast<int>(delimiter::FallbackReason::None));
    std::atomic<bool> delimiterLocked(false);

    audio_pipeline::Dependencies deps{};
    deps.config = &config;
    deps.running = &running;
    deps.fallbackActive = &fallbackActive;
    deps.outputReady = &outputReady;
    deps.crossfeedEnabled = &crossfeedEnabled;
    deps.crossfeedResetRequested = &crossfeedResetRequested;
    deps.crossfeedProcessor = nullptr;
    deps.cfStreamInputLeft = &cfStreamInputLeft;
    deps.cfStreamInputRight = &cfStreamInputRight;
    deps.cfStreamAccumulatedLeft = &cfAccumLeft;
    deps.cfStreamAccumulatedRight = &cfAccumRight;
    deps.cfOutputLeft = &cfOutputLeft;
    deps.cfOutputRight = &cfOutputRight;
    deps.streamInputLeft = &streamInputLeft;
    deps.streamInputRight = &streamInputRight;
    deps.streamAccumulatedLeft = &streamAccumLeft;
    deps.streamAccumulatedRight = &streamAccumRight;
    deps.upsamplerOutputLeft = &upsamplerOutputLeft;
    deps.upsamplerOutputRight = &upsamplerOutputRight;
    deps.streamingCacheManager = nullptr;
    deps.buffer.playbackBuffer = &playbackBuffer;
    deps.maxOutputBufferFrames = []() { return kBufferFrames; };
    deps.currentInputRate = []() { return 1000; };
    deps.currentOutputRate = []() { return 1000; };
    deps.delimiterMode = &delimiterMode;
    deps.delimiterFallbackReason = &delimiterReason;
    deps.delimiterBypassLocked = &delimiterLocked;
    deps.delimiterBackendFactory = [](const AppConfig::DelimiterConfig& cfg) {
        return std::make_unique<FailingBackend>(cfg.expectedSampleRate);
    };

    deps.upsampler.available = true;
    deps.upsampler.streamLeft = nullptr;
    deps.upsampler.streamRight = nullptr;
    deps.upsampler.process = [](const float* inputData, std::size_t inputFrames,
                                ConvolutionEngine::StreamFloatVector& outputData,
                                cudaStream_t /*stream*/,
                                ConvolutionEngine::StreamFloatVector& /*streamInput*/,
                                std::size_t& /*streamAccumulated*/) {
        outputData.assign(inputData, inputData + inputFrames);
        return true;
    };

    audio_pipeline::AudioPipeline pipeline(std::move(deps));

    const std::size_t chunkFrames =
        static_cast<std::size_t>(std::lround(config.delimiter.chunkSec * 1000));
    const std::size_t overlapFrames =
        static_cast<std::size_t>(std::lround(config.delimiter.overlapSec * 1000));
    const std::size_t hopFrames = chunkFrames - overlapFrames;

    std::vector<float> input(chunkFrames * 2, 0.1f);

    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(pipeline.process(input.data(), static_cast<uint32_t>(chunkFrames)));
        ASSERT_TRUE(waitForQueuedFrames(playbackBuffer, hopFrames, std::chrono::milliseconds(500)));
        std::vector<float> discardL(hopFrames);
        std::vector<float> discardR(hopFrames);
        ASSERT_TRUE(playbackBuffer.readPlanar(discardL.data(), discardR.data(), hopFrames));
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        auto mode =
            static_cast<delimiter::ProcessingMode>(delimiterMode.load(std::memory_order_relaxed));
        bool locked = delimiterLocked.load(std::memory_order_relaxed);
        if (locked && mode == delimiter::ProcessingMode::Bypass) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_TRUE(delimiterLocked.load());
    EXPECT_EQ(static_cast<delimiter::ProcessingMode>(delimiterMode.load()),
              delimiter::ProcessingMode::Bypass);
    EXPECT_EQ(static_cast<delimiter::FallbackReason>(delimiterReason.load()),
              delimiter::FallbackReason::InferenceFailure);

    running.store(false, std::memory_order_release);
}
