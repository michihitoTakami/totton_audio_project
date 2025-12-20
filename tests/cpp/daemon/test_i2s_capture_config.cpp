#include "daemon/input/i2s_capture.h"
#include "gtest/gtest.h"

TEST(I2sCaptureConfigTest, ParseI2sFormatUsesLoopbackFormatSet) {
    EXPECT_EQ(daemon_input::parseI2sFormat("S16_LE"), SND_PCM_FORMAT_S16_LE);
    EXPECT_EQ(daemon_input::parseI2sFormat("s24_3le"), SND_PCM_FORMAT_S24_3LE);
    EXPECT_EQ(daemon_input::parseI2sFormat("S32"), SND_PCM_FORMAT_S32_LE);
    EXPECT_EQ(daemon_input::parseI2sFormat("unknown"), SND_PCM_FORMAT_UNKNOWN);
}

TEST(I2sCaptureConfigTest, ValidateI2sConfigRejectsInvalidConfigs) {
    AppConfig cfg;
    cfg.i2s.enabled = true;

    // device is required
    cfg.i2s.device = "";
    cfg.i2s.channels = DaemonConstants::CHANNELS;
    cfg.i2s.periodFrames = 1024;
    cfg.i2s.format = "S16_LE";
    cfg.i2s.sampleRate = 0;
    EXPECT_FALSE(daemon_input::validateI2sConfig(cfg));

    // valid baseline
    cfg.i2s.device = "hw:Dummy";
    EXPECT_TRUE(daemon_input::validateI2sConfig(cfg));

    // unsupported sample rate (except 0/44100/48000)
    cfg.i2s.sampleRate = 96000;
    EXPECT_FALSE(daemon_input::validateI2sConfig(cfg));
}
