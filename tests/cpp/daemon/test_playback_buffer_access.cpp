#include "core/daemon_constants.h"
#include "daemon/app/runtime_state.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/audio_pipeline/filter_manager.h"
#include "daemon/audio_pipeline/headroom_controller.h"
#include "daemon/audio_pipeline/rate_switcher.h"
#include "daemon/audio_pipeline/soft_mute_runner.h"
#include "daemon/audio_pipeline/streaming_cache_manager.h"
#include "daemon/control/handlers/handler_registry.h"
#include "daemon/output/alsa_output.h"
#include "daemon/output/playback_buffer_access.h"
#include "daemon/pcm/dac_manager.h"
#include "gtest/gtest.h"

#include <atomic>
#include <mutex>

TEST(PlaybackBufferAccessTest, ComputesMaxOutputBufferFramesFromRuntimeRate) {
    daemon_app::RuntimeState state;

    state.rates.currentOutputRate.store(48000, std::memory_order_release);
    size_t frames = daemon_output::maxOutputBufferFrames(state);
    EXPECT_EQ(frames,
              static_cast<size_t>(48000.0 *
                                  static_cast<double>(DaemonConstants::MAX_OUTPUT_BUFFER_SECONDS)));

    state.rates.currentOutputRate.store(0, std::memory_order_release);
    frames = daemon_output::maxOutputBufferFrames(state);
    EXPECT_EQ(frames, DaemonConstants::DEFAULT_MAX_OUTPUT_BUFFER_FRAMES);
}

TEST(PlaybackBufferAccessTest, LazilyCreatesPlaybackBufferManagerAndReturnsSameInstance) {
    daemon_app::RuntimeState state;
    auto& a = daemon_output::playbackBuffer(state);
    auto& b = daemon_output::playbackBuffer(state);
    EXPECT_EQ(&a, &b);
    {
        std::lock_guard<std::mutex> lock(a.mutex());
        EXPECT_EQ(a.queuedFramesLocked(), 0u);
    }
}
