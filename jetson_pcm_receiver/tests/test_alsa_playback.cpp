#include "alsa_playback.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

TEST(AlsaPlayback, RejectsUnsupportedParams) {
    AlsaPlayback playback("null");

    EXPECT_FALSE(playback.open(50000, 2, 1));  // rate unsupported
    EXPECT_FALSE(playback.open(48000, 2, 3));  // unsupported format code
    EXPECT_FALSE(playback.open(48000, 1, 1));  // channels mismatch
}

TEST(AlsaPlayback, RejectsMissingDevice) {
    AlsaPlayback playback("hw:NonexistentDeviceForTest");
    EXPECT_FALSE(playback.open(48000, 2, 1));
}

TEST(AlsaPlayback, OpenWriteCloseWithNullDevice) {
    AlsaPlayback playback("null");
    if (!playback.open(48000, 2, 1)) {
        GTEST_SKIP() << "ALSA null device is not available on this system";
    }

    std::vector<std::uint8_t> silence(32, 0);  // 8 frames of S16_LE stereo
    EXPECT_TRUE(playback.write(silence.data(), 8));
    playback.close();
}
