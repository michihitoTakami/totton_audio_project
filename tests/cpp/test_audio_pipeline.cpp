#include "config_loader.h"
#include "convolution_engine.h"
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
    EXPECT_DOUBLE_EQ(peaks["upsampler"]["linear"], 0.7);
    EXPECT_DOUBLE_EQ(peaks["post_mix"]["linear"], 0.7);
    EXPECT_EQ(runtime_stats::clipCount(), 0u);
}
