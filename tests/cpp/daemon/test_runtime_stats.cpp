#include "audio/filter_headroom.h"
#include "daemon/metrics/runtime_stats.h"
#include "gtest/gtest.h"

#include <atomic>
#include <filesystem>
#include <fstream>

namespace {

runtime_stats::Dependencies makeDeps(AppConfig& config, FilterHeadroomCache& headroom,
                                     std::atomic<float>& headroomGain,
                                     std::atomic<float>& outputGain,
                                     std::atomic<float>& limiterGain,
                                     std::atomic<float>& effectiveGain,
                                     std::atomic<bool>& fallbackActive, int inputRate) {
    runtime_stats::Dependencies deps;
    deps.config = &config;
    deps.headroomCache = &headroom;
    deps.headroomGain = &headroomGain;
    deps.outputGain = &outputGain;
    deps.limiterGain = &limiterGain;
    deps.effectiveGain = &effectiveGain;
    deps.fallbackActive = &fallbackActive;
    deps.inputSampleRate = &inputRate;
    return deps;
}

}  // namespace

TEST(RuntimeStatsTest, CollectsUpdatedValues) {
    runtime_stats::reset();
    AppConfig config;
    config.eqEnabled = true;
    config.gain = 1.5f;
    config.headroomTarget = 0.85f;

    FilterHeadroomCache headroom;
    std::atomic<float> headroomGain(0.75f);
    std::atomic<float> outputGain(1.1f);
    std::atomic<float> limiterGain(0.95f);
    std::atomic<float> effectiveGain(1.05f);
    std::atomic<bool> fallbackActive(false);
    int inputRate = 48000;

    runtime_stats::recordClip();
    runtime_stats::addSamples(48000);
    runtime_stats::addDroppedFrames(17);
    runtime_stats::updateInputPeak(0.2f);
    runtime_stats::updateUpsamplerPeak(0.4f);
    runtime_stats::updatePostCrossfeedPeak(0.6f);
    runtime_stats::updatePostGainPeak(0.8f);
    runtime_stats::recordRenderedSilenceBlock();
    runtime_stats::addRenderedSilenceFrames(128);
    runtime_stats::recordUpsamplerNeedMoreBlock(true);
    runtime_stats::recordUpsamplerNeedMoreBlock(false);
    runtime_stats::recordUpsamplerErrorBlock(true);
    runtime_stats::recordUpsamplerErrorBlock(false);

    auto deps = makeDeps(config, headroom, headroomGain, outputGain, limiterGain, effectiveGain,
                         fallbackActive, inputRate);
    auto stats = runtime_stats::collect(deps, 2048);

    EXPECT_EQ(stats["clip_count"], 1);
    EXPECT_EQ(stats["total_samples"], 48000);
    EXPECT_TRUE(stats["eq_enabled"].get<bool>());
    EXPECT_EQ(stats["buffer"]["capacity_frames"], 2048u);
    EXPECT_EQ(stats["buffer"]["dropped_frames"], 17u);
    EXPECT_EQ(stats["fallback"]["active"], false);

    EXPECT_FLOAT_EQ(stats["gain"]["headroom"].get<float>(), 0.75f);
    EXPECT_FLOAT_EQ(stats["gain"]["headroom_effective"].get<float>(), 1.1f);
    EXPECT_FLOAT_EQ(stats["gain"]["limiter"].get<float>(), 0.95f);
    EXPECT_FLOAT_EQ(stats["gain"]["effective"].get<float>(), 1.05f);

    EXPECT_FLOAT_EQ(stats["peaks"]["input"]["linear"].get<float>(), 0.2f);
    EXPECT_FLOAT_EQ(stats["peaks"]["upsampler"]["linear"].get<float>(), 0.4f);
    EXPECT_FLOAT_EQ(stats["peaks"]["post_mix"]["linear"].get<float>(), 0.6f);
    EXPECT_FLOAT_EQ(stats["peaks"]["post_gain"]["linear"].get<float>(), 0.8f);
    EXPECT_EQ(stats["buffer"]["rendered_silence_blocks"], 1u);
    EXPECT_EQ(stats["buffer"]["rendered_silence_frames"], 128u);
    EXPECT_EQ(stats["upsampler_streaming"]["need_more_blocks_left"], 1u);
    EXPECT_EQ(stats["upsampler_streaming"]["need_more_blocks_right"], 1u);
    EXPECT_EQ(stats["upsampler_streaming"]["error_blocks_left"], 1u);
    EXPECT_EQ(stats["upsampler_streaming"]["error_blocks_right"], 1u);
}

TEST(RuntimeStatsTest, WriteStatsFileCreatesJson) {
    runtime_stats::reset();
    AppConfig config;
    FilterHeadroomCache headroom;
    std::atomic<float> headroomGain(1.0f);
    std::atomic<float> outputGain(1.0f);
    std::atomic<float> limiterGain(1.0f);
    std::atomic<float> effectiveGain(1.0f);
    std::atomic<bool> fallbackActive(true);
    int inputRate = 44100;

    runtime_stats::recordClip();
    runtime_stats::addSamples(1024);
    runtime_stats::addDroppedFrames(5);

    auto deps = makeDeps(config, headroom, headroomGain, outputGain, limiterGain, effectiveGain,
                         fallbackActive, inputRate);

    auto tmpFile = std::filesystem::temp_directory_path() / "gpu_runtime_stats_test.json";
    std::error_code ec;
    std::filesystem::remove(tmpFile, ec);

    runtime_stats::writeStatsFile(deps, 1024, tmpFile.string());

    std::ifstream ifs(tmpFile);
    ASSERT_TRUE(ifs.is_open());

    nlohmann::json data;
    ifs >> data;
    EXPECT_EQ(data["clip_count"], 1);
    EXPECT_EQ(data["buffer"]["capacity_frames"], 1024u);
    EXPECT_EQ(data["buffer"]["dropped_frames"], 5u);
    EXPECT_EQ(data["fallback"]["active"], true);
    EXPECT_EQ(data["buffer"]["rendered_silence_blocks"], 0u);
    EXPECT_EQ(data["buffer"]["rendered_silence_frames"], 0u);
    EXPECT_EQ(data["upsampler_streaming"]["need_more_blocks_left"], 0u);
    EXPECT_EQ(data["upsampler_streaming"]["need_more_blocks_right"], 0u);
    EXPECT_EQ(data["upsampler_streaming"]["error_blocks_left"], 0u);
    EXPECT_EQ(data["upsampler_streaming"]["error_blocks_right"], 0u);

    std::filesystem::remove(tmpFile, ec);
}
