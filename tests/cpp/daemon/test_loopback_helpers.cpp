#include "core/daemon_constants.h"
#include "daemon/input/loopback_capture.h"
#include "gtest/gtest.h"

TEST(LoopbackHelpersTest, ParseLoopbackFormatHandlesAliases) {
    EXPECT_EQ(daemon_input::parseLoopbackFormat("S16_LE"), SND_PCM_FORMAT_S16_LE);
    EXPECT_EQ(daemon_input::parseLoopbackFormat("s24_3le"), SND_PCM_FORMAT_S24_3LE);
    EXPECT_EQ(daemon_input::parseLoopbackFormat("S32"), SND_PCM_FORMAT_S32_LE);
    EXPECT_EQ(daemon_input::parseLoopbackFormat("unknown"), SND_PCM_FORMAT_UNKNOWN);
}

TEST(LoopbackHelpersTest, ValidateLoopbackConfigAcceptsValidSettings) {
    AppConfig cfg;
    cfg.loopback.enabled = true;
    cfg.loopback.sampleRate = 44100;
    cfg.loopback.channels = DaemonConstants::CHANNELS;
    cfg.loopback.periodFrames = 1024;
    cfg.loopback.format = "S16_LE";

    EXPECT_TRUE(daemon_input::validateLoopbackConfig(cfg));
}

TEST(LoopbackHelpersTest, ValidateLoopbackConfigRejectsInvalidSettings) {
    AppConfig cfg;
    cfg.loopback.enabled = true;
    cfg.loopback.sampleRate = 44100;
    cfg.loopback.channels = DaemonConstants::CHANNELS;
    cfg.loopback.periodFrames = 1024;
    cfg.loopback.format = "S16_LE";

    cfg.loopback.sampleRate = 96000;
    EXPECT_FALSE(daemon_input::validateLoopbackConfig(cfg));

    cfg.loopback.sampleRate = 44100;
    cfg.loopback.channels = 1;
    EXPECT_FALSE(daemon_input::validateLoopbackConfig(cfg));

    cfg.loopback.channels = DaemonConstants::CHANNELS;
    cfg.loopback.periodFrames = 0;
    EXPECT_FALSE(daemon_input::validateLoopbackConfig(cfg));

    cfg.loopback.periodFrames = 1024;
    cfg.loopback.format = "BAD";
    EXPECT_FALSE(daemon_input::validateLoopbackConfig(cfg));
}

TEST(LoopbackHelpersTest, ValidateLoopbackConfigAllowsDisabled) {
    AppConfig cfg;
    cfg.loopback.enabled = false;
    cfg.loopback.sampleRate = 96000;
    cfg.loopback.channels = 1;
    cfg.loopback.periodFrames = 0;
    cfg.loopback.format = "BAD";

    EXPECT_TRUE(daemon_input::validateLoopbackConfig(cfg));
}

TEST(LoopbackHelpersTest, ConvertPcmToFloatHandlesS16) {
    const int16_t samples[] = {0, 32767, -32768};
    std::vector<float> dst;

    ASSERT_TRUE(daemon_input::convertPcmToFloat(samples, SND_PCM_FORMAT_S16_LE, 3, 1, dst));
    ASSERT_EQ(dst.size(), 3u);
    EXPECT_NEAR(dst[0], 0.0f, 1e-6f);
    EXPECT_NEAR(dst[1], 32767.0f / 32768.0f, 1e-6f);
    EXPECT_NEAR(dst[2], -1.0f, 1e-6f);
}

TEST(LoopbackHelpersTest, ConvertPcmToFloatHandlesS32) {
    const int32_t samples[] = {0, 2147483647, static_cast<int32_t>(0x80000000)};
    std::vector<float> dst;

    ASSERT_TRUE(daemon_input::convertPcmToFloat(samples, SND_PCM_FORMAT_S32_LE, 3, 1, dst));
    ASSERT_EQ(dst.size(), 3u);
    EXPECT_NEAR(dst[0], 0.0f, 1e-6f);
    EXPECT_NEAR(dst[1], 2147483647.0f / 2147483648.0f, 1e-6f);
    EXPECT_NEAR(dst[2], -1.0f, 1e-6f);
}

TEST(LoopbackHelpersTest, ConvertPcmToFloatHandlesS24_3LE) {
    std::vector<uint8_t> samples = {
        0x00, 0x00, 0x00,  // 0
        0xFF, 0xFF, 0x7F,  // 0x7FFFFF
        0x00, 0x00, 0x80   // 0x800000
    };
    std::vector<float> dst;

    ASSERT_TRUE(daemon_input::convertPcmToFloat(samples.data(), SND_PCM_FORMAT_S24_3LE, 3, 1, dst));
    ASSERT_EQ(dst.size(), 3u);
    EXPECT_NEAR(dst[0], 0.0f, 1e-6f);
    EXPECT_NEAR(dst[1], 8388607.0f / 8388608.0f, 1e-6f);
    EXPECT_NEAR(dst[2], -1.0f, 1e-6f);
}

TEST(LoopbackHelpersTest, ConvertPcmToFloatRejectsUnknownFormat) {
    const int16_t samples[] = {0, 1};
    std::vector<float> dst;

    EXPECT_FALSE(daemon_input::convertPcmToFloat(samples, SND_PCM_FORMAT_UNKNOWN, 2, 1, dst));
}
