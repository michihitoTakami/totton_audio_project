#include "daemon/app/runtime_state.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/audio_pipeline/filter_manager.h"
#include "daemon/audio_pipeline/headroom_controller.h"
#include "daemon/audio_pipeline/rate_switcher.h"
#include "daemon/audio_pipeline/soft_mute_runner.h"
#include "daemon/audio_pipeline/streaming_cache_manager.h"
#include "daemon/audio_pipeline/switch_actions.h"
#include "daemon/control/handlers/handler_registry.h"
#include "daemon/output/alsa_output.h"
#include "daemon/output/playback_buffer_access.h"
#include "daemon/pcm/dac_manager.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

TEST(SwitchActionsTest, ResetStreamingCachesClearsPlaybackAndStreamingStateWithoutPipeline) {
    daemon_app::RuntimeState state;

    // Fill playback buffer with some frames.
    {
        auto& pb = daemon_output::playbackBuffer(state);
        std::vector<float> left(128, 0.25f);
        std::vector<float> right(128, -0.25f);
        size_t stored = 0;
        size_t dropped = 0;
        ASSERT_TRUE(pb.enqueue(left.data(), right.data(), left.size(),
                               /*outputRate=*/DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE, stored,
                               dropped));
        ASSERT_GT(stored, 0u);
    }

    // Seed streaming buffers and counters.
    state.streaming.streamInputLeft.assign(16, 1.0f);
    state.streaming.streamInputRight.assign(16, -1.0f);
    state.streaming.streamAccumulatedLeft = 7;
    state.streaming.streamAccumulatedRight = 9;
    state.streaming.upsamplerOutputLeft.assign(32, 0.1f);
    state.streaming.upsamplerOutputRight.assign(32, 0.2f);
    state.crossfeed.resetRequested.store(false, std::memory_order_release);

    // No pipeline; should still succeed and reset state deterministically.
    ASSERT_TRUE(audio_pipeline::resetStreamingCachesForSwitch(state));

    // Playback buffer should be empty.
    {
        auto& pb = daemon_output::playbackBuffer(state);
        std::lock_guard<std::mutex> lock(pb.mutex());
        EXPECT_EQ(pb.queuedFramesLocked(), 0u);
    }

    // Streaming buffers should be zeroed/cleared.
    EXPECT_EQ(state.streaming.streamAccumulatedLeft, 0u);
    EXPECT_EQ(state.streaming.streamAccumulatedRight, 0u);
    EXPECT_TRUE(state.streaming.upsamplerOutputLeft.empty());
    EXPECT_TRUE(state.streaming.upsamplerOutputRight.empty());
    EXPECT_TRUE(std::all_of(state.streaming.streamInputLeft.begin(),
                            state.streaming.streamInputLeft.end(),
                            [](float v) { return v == 0.0f; }));
    EXPECT_TRUE(std::all_of(state.streaming.streamInputRight.begin(),
                            state.streaming.streamInputRight.end(),
                            [](float v) { return v == 0.0f; }));

    // Crossfeed reset should be requested.
    EXPECT_TRUE(state.crossfeed.resetRequested.load(std::memory_order_acquire));
}

TEST(SwitchActionsTest, ReinitializeStreamingForLegacyModeReturnsFalseWithoutUpsampler) {
    daemon_app::RuntimeState state;
    state.upsampler = nullptr;
    EXPECT_FALSE(audio_pipeline::reinitializeStreamingForLegacyMode(state));
}
