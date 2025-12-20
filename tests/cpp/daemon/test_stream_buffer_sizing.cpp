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

    // Defaults: blockSize/periodSize/loopback.periodFrames are 0 until set.
    // The helper should fall back to DEFAULT_BLOCK_SIZE (4096) and apply 2x margin.
    size_t cap =
        audio_pipeline::computeStreamBufferCapacity(state, /*streamValidInputPerBlock=*/1024);
    EXPECT_EQ(cap, 4096u * 2u);

    // If periodSize is larger, it should dominate.
    state.config.periodSize = 16384;
    cap = audio_pipeline::computeStreamBufferCapacity(state, /*streamValidInputPerBlock=*/1024);
    EXPECT_EQ(cap, 16384u * 2u);

    // If loopback.periodFrames is larger, it should dominate.
    state.config.loopback.periodFrames = 20000;
    cap = audio_pipeline::computeStreamBufferCapacity(state, /*streamValidInputPerBlock=*/1024);
    EXPECT_EQ(cap, 20000u * 2u);

    // If streamValidInputPerBlock is larger than any config, it should dominate.
    cap = audio_pipeline::computeStreamBufferCapacity(state, /*streamValidInputPerBlock=*/40000);
    EXPECT_EQ(cap, 40000u * 2u);
}
