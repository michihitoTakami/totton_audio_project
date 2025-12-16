#include "convolution_engine.h"
#include "core/config_loader.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/metrics/runtime_stats.h"
#include "gtest/gtest.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

TEST(AudioPipeline, EnqueuesOutputAndUpdatesStats) {
    runtime_stats::reset();

    AppConfig config;
    config.upsampleRatio = 2;

    std::vector<float> outputBufferLeft;
    std::vector<float> outputBufferRight;
    size_t outputReadPos = 0;
    std::mutex bufferMutex;
    std::condition_variable bufferCv;

    std::atomic<bool> fallbackActive{false};
    std::atomic<bool> outputReady{true};
    std::atomic<bool> crossfeedEnabled{false};

    std::mutex inputMutex;
    std::mutex crossfeedMutex;

    ConvolutionEngine::StreamFloatVector streamInputLeft;
    ConvolutionEngine::StreamFloatVector streamInputRight;
    size_t streamAccumLeft = 0;
    size_t streamAccumRight = 0;
    ConvolutionEngine::StreamFloatVector upsamplerOutputLeft;
    ConvolutionEngine::StreamFloatVector upsamplerOutputRight;

    std::vector<float> cfStreamInputLeft;
    std::vector<float> cfStreamInputRight;
    size_t cfAccumLeft = 0;
    size_t cfAccumRight = 0;
    std::vector<float> cfOutputLeft;
    std::vector<float> cfOutputRight;

    audio_pipeline::Dependencies deps{};
    deps.config = &config;
    deps.fallbackActive = &fallbackActive;
    deps.outputReady = &outputReady;
    deps.crossfeedEnabled = &crossfeedEnabled;
    deps.crossfeedProcessor = nullptr;
    deps.crossfeedMutex = &crossfeedMutex;
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
    deps.inputMutex = &inputMutex;
    deps.buffer.outputBufferLeft = &outputBufferLeft;
    deps.buffer.outputBufferRight = &outputBufferRight;
    deps.buffer.outputReadPos = &outputReadPos;
    deps.buffer.bufferMutex = &bufferMutex;
    deps.buffer.bufferCv = &bufferCv;
    deps.maxOutputBufferFrames = []() { return static_cast<size_t>(48); };
    deps.currentOutputRate = []() { return 48000; };

    deps.upsampler.available = true;
    deps.upsampler.streamLeft = 0;
    deps.upsampler.streamRight = 0;
    deps.upsampler.process =
        [](const float* inputData, size_t inputFrames,
           ConvolutionEngine::StreamFloatVector& outputData, cudaStream_t /*stream*/,
           ConvolutionEngine::StreamFloatVector& /*streamInput*/, size_t& /*streamAccumulated*/) {
            outputData.assign(inputFrames * 2, 0.7f);
            return true;
        };

    audio_pipeline::AudioPipeline pipeline(std::move(deps));

    std::vector<float> input = {0.3f, -0.2f, -0.5f, 0.1f, 0.2f, 0.4f};
    ASSERT_TRUE(pipeline.process(input.data(), 3));

    EXPECT_EQ(outputBufferLeft.size(), static_cast<size_t>(6));
    EXPECT_EQ(outputBufferRight.size(), static_cast<size_t>(6));
    for (float value : outputBufferLeft) {
        EXPECT_FLOAT_EQ(value, 0.7f);
    }
    for (float value : outputBufferRight) {
        EXPECT_FLOAT_EQ(value, 0.7f);
    }

    runtime_stats::Dependencies statsDeps;
    statsDeps.config = &config;
    auto stats = runtime_stats::collect(statsDeps, 16);
    const auto& peaks = stats["peaks"];
    EXPECT_DOUBLE_EQ(peaks["input"]["linear"], 0.5);
    EXPECT_NEAR(peaks["upsampler"]["linear"], 0.7, 1e-6);
    EXPECT_NEAR(peaks["post_mix"]["linear"], 0.7, 1e-6);
    EXPECT_EQ(runtime_stats::clipCount(), 0u);
}

TEST(AudioPipeline, RenderOutputAppliesLimiterAndClipping) {
    runtime_stats::reset();

    AppConfig config;
    config.upsampleRatio = 1;
    config.headroomTarget = 1.0f;

    std::vector<float> outputBufferLeft;
    std::vector<float> outputBufferRight;
    size_t outputReadPos = 0;
    std::mutex bufferMutex;
    std::condition_variable bufferCv;

    std::atomic<bool> fallbackActive{false};
    std::atomic<bool> outputReady{true};
    std::atomic<bool> crossfeedEnabled{false};

    std::mutex inputMutex;
    std::mutex crossfeedMutex;

    ConvolutionEngine::StreamFloatVector streamInputLeft;
    ConvolutionEngine::StreamFloatVector streamInputRight;
    size_t streamAccumLeft = 0;
    size_t streamAccumRight = 0;
    ConvolutionEngine::StreamFloatVector upsamplerOutputLeft;
    ConvolutionEngine::StreamFloatVector upsamplerOutputRight;

    std::vector<float> cfStreamInputLeft;
    std::vector<float> cfStreamInputRight;
    size_t cfAccumLeft = 0;
    size_t cfAccumRight = 0;
    std::vector<float> cfOutputLeft;
    std::vector<float> cfOutputRight;

    std::atomic<float> outputGain{2.0f};
    std::atomic<float> limiterGain{1.0f};
    std::atomic<float> effectiveGain{1.0f};

    audio_pipeline::Dependencies deps{};
    deps.config = &config;
    deps.output.outputGain = &outputGain;
    deps.output.limiterGain = &limiterGain;
    deps.output.effectiveGain = &effectiveGain;
    deps.fallbackActive = &fallbackActive;
    deps.outputReady = &outputReady;
    deps.crossfeedEnabled = &crossfeedEnabled;
    deps.crossfeedProcessor = nullptr;
    deps.crossfeedMutex = &crossfeedMutex;
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
    deps.inputMutex = &inputMutex;
    deps.buffer.outputBufferLeft = &outputBufferLeft;
    deps.buffer.outputBufferRight = &outputBufferRight;
    deps.buffer.outputReadPos = &outputReadPos;
    deps.buffer.bufferMutex = &bufferMutex;
    deps.buffer.bufferCv = &bufferCv;
    deps.maxOutputBufferFrames = []() { return static_cast<size_t>(48); };
    deps.currentOutputRate = []() { return 48000; };

    deps.upsampler.available = true;
    deps.upsampler.streamLeft = 0;
    deps.upsampler.streamRight = 0;
    deps.upsampler.process =
        [](const float* inputData, size_t inputFrames,
           ConvolutionEngine::StreamFloatVector& outputData, cudaStream_t /*stream*/,
           ConvolutionEngine::StreamFloatVector& /*streamInput*/, size_t& /*streamAccumulated*/) {
            outputData.assign(inputFrames, 0.8f);
            return true;
        };

    audio_pipeline::AudioPipeline pipeline(std::move(deps));

    std::vector<float> input = {0.8f, 0.8f, 0.8f, 0.8f};
    ASSERT_TRUE(pipeline.process(input.data(), 2));

    std::vector<int32_t> interleaved;
    std::vector<float> floatScratch;
    auto renderResult = pipeline.renderOutput(2, interleaved, floatScratch, nullptr);
    EXPECT_EQ(renderResult.framesRendered, static_cast<size_t>(2));
    EXPECT_FALSE(renderResult.wroteSilence);
    ASSERT_EQ(interleaved.size(), static_cast<size_t>(4));
    constexpr double kInt32Max = 2147483647.0;
    for (auto sample : interleaved) {
        EXPECT_NEAR(static_cast<double>(sample), kInt32Max, 2048.0);
    }
    EXPECT_NEAR(limiterGain.load(), 0.625f, 1e-6f);
    EXPECT_NEAR(effectiveGain.load(), outputGain.load() * 0.625f, 1e-6f);

    runtime_stats::Dependencies statsDeps;
    statsDeps.config = &config;
    auto stats = runtime_stats::collect(statsDeps, 16);
    const auto& peaks = stats["peaks"];
    EXPECT_DOUBLE_EQ(peaks["post_gain"]["linear"], 1.0);
    EXPECT_EQ(runtime_stats::clipCount(), 0u);
    EXPECT_EQ(runtime_stats::totalSamples(), static_cast<size_t>(4));
}
