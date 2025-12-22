#include "daemon/app/runtime_state.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/audio_pipeline/filter_manager.h"
#include "daemon/audio_pipeline/headroom_controller.h"
#include "daemon/audio_pipeline/rate_switcher.h"
#include "daemon/audio_pipeline/soft_mute_runner.h"
#include "daemon/audio_pipeline/stream_buffer_sizing.h"
#include "daemon/audio_pipeline/streaming_cache_manager.h"
#include "daemon/control/handlers/handler_registry.h"
#include "daemon/output/alsa_output.h"
#include "daemon/pcm/dac_manager.h"
#include "gtest/gtest.h"

TEST(StreamBufferSizingTest, UsesMaxOfRelevantSizesAndAppliesSafetyMargin) {
    daemon_app::RuntimeState state;

    auto expectedCapacity = [&](size_t streamValidInputPerBlock) -> size_t {
        size_t frames = static_cast<size_t>(DaemonConstants::DEFAULT_BLOCK_SIZE);
        if (state.config.blockSize > 0) {
            frames = std::max(frames, static_cast<size_t>(state.config.blockSize));
        }
        if (state.config.periodSize > 0) {
            frames = std::max(frames, static_cast<size_t>(state.config.periodSize));
        }
        if (state.config.loopback.periodFrames > 0) {
            frames = std::max(frames, static_cast<size_t>(state.config.loopback.periodFrames));
        }
        frames = std::max(frames, streamValidInputPerBlock);
        // 3x margin to tolerate overlap of RT path and high-latency worker (Fix #1138)
        return frames * 3;
    };

    size_t cap =
        audio_pipeline::computeStreamBufferCapacity(state, /*streamValidInputPerBlock=*/1024);
    EXPECT_EQ(cap, expectedCapacity(1024));

    // If periodSize is larger, it should dominate.
    state.config.periodSize = 16384;
    cap = audio_pipeline::computeStreamBufferCapacity(state, /*streamValidInputPerBlock=*/1024);
    EXPECT_EQ(cap, expectedCapacity(1024));

    // If loopback.periodFrames is larger, it should dominate.
    state.config.loopback.periodFrames = 20000;
    cap = audio_pipeline::computeStreamBufferCapacity(state, /*streamValidInputPerBlock=*/1024);
    EXPECT_EQ(cap, expectedCapacity(1024));

    // If streamValidInputPerBlock is larger than any config, it should dominate.
    cap = audio_pipeline::computeStreamBufferCapacity(state, /*streamValidInputPerBlock=*/40000);
    EXPECT_EQ(cap, expectedCapacity(40000));
}
