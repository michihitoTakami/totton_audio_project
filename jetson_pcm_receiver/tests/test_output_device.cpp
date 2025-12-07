#include "output_device.h"

#include <gtest/gtest.h>

TEST(OutputDevice, ParsesLoopbackAlias) {
    auto res = parseOutputDevice("loopback");
    ASSERT_TRUE(res.ok);
    EXPECT_EQ(res.spec.type, OutputDeviceType::Loopback);
    EXPECT_EQ(res.spec.alsaName, "hw:Loopback,0,0");
    EXPECT_EQ(res.spec.userValue, "loopback");
}

TEST(OutputDevice, ParsesLoopbackPlaybackAlias) {
    auto res = parseOutputDevice("loopback-playback");
    ASSERT_TRUE(res.ok);
    EXPECT_EQ(res.spec.type, OutputDeviceType::Loopback);
    EXPECT_EQ(res.spec.alsaName, "hw:Loopback,0,0");
    EXPECT_EQ(res.spec.userValue, "loopback-playback");
}

TEST(OutputDevice, ParsesAlsaPrefix) {
    auto res = parseOutputDevice("alsa:hw:1,0,0");
    ASSERT_TRUE(res.ok);
    EXPECT_EQ(res.spec.type, OutputDeviceType::Alsa);
    EXPECT_EQ(res.spec.alsaName, "hw:1,0,0");
}

TEST(OutputDevice, ParsesRawPcmNameForBackwardCompatibility) {
    auto res = parseOutputDevice("hw:USB,0,0");
    ASSERT_TRUE(res.ok);
    EXPECT_EQ(res.spec.type, OutputDeviceType::Alsa);
    EXPECT_EQ(res.spec.alsaName, "hw:USB,0,0");
}

TEST(OutputDevice, ParsesNullAlias) {
    auto res = parseOutputDevice("null");
    ASSERT_TRUE(res.ok);
    EXPECT_EQ(res.spec.type, OutputDeviceType::NullSink);
    EXPECT_EQ(res.spec.alsaName, "null");
}

TEST(OutputDevice, RejectsEmptyValue) {
    auto res = parseOutputDevice("");
    EXPECT_FALSE(res.ok);
    EXPECT_FALSE(res.error.empty());
}

TEST(OutputDevice, RejectsMissingAlsaName) {
    auto res = parseOutputDevice("alsa:");
    EXPECT_FALSE(res.ok);
    EXPECT_FALSE(res.error.empty());
}
