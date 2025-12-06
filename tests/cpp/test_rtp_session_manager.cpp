#include "daemon/rtp/rtp_session_manager.h"

#include <gtest/gtest.h>

using namespace Network;

TEST(RtpSessionConfigTest, ParseValidJson) {
    nlohmann::json params = {{"session_id", "primary"}, {"port", 7000},
                             {"sample_rate", 96000},    {"bits_per_sample", 32},
                             {"payload_type", 98},      {"enable_rtcp", false}};

    SessionConfig config;
    std::string error;
    EXPECT_TRUE(sessionConfigFromJson(params, config, error));
    EXPECT_EQ(config.sessionId, "primary");
    EXPECT_EQ(config.port, 7000);
    EXPECT_EQ(config.sampleRate, 96000u);
    EXPECT_EQ(config.bitsPerSample, 32);
    EXPECT_EQ(config.payloadType, 98);
    EXPECT_FALSE(config.enableRtcp);
}

TEST(RtpSessionConfigTest, RejectInvalidBitsPerSample) {
    nlohmann::json params = {{"bits_per_sample", 20}};
    SessionConfig config;
    std::string error;
    EXPECT_FALSE(sessionConfigFromJson(params, config, error));
    EXPECT_FALSE(error.empty());
}

TEST(RtpSessionConfigTest, ApplySdpOverrides) {
    nlohmann::json params = {
        {"session_id", "sdp"},
        {"payload_type", 111},
        {"sample_rate", 48000},
        {"channels", 2},
        {"bits_per_sample", 24},
        {"sdp", R"(v=0
o=- 0 0 IN IP4 127.0.0.1
s=Test
t=0 0
m=audio 6000 RTP/AVP 112
a=rtpmap:112 L24/96000/6
a=rtpmap:111 L16/44100/2
)"},
    };

    SessionConfig config;
    std::string error;
    ASSERT_TRUE(sessionConfigFromJson(params, config, error));
    // Prefer the rtpmap line that matches payload_type
    EXPECT_EQ(config.payloadType, 111);
    EXPECT_EQ(config.sampleRate, 44100u);
    EXPECT_EQ(config.channels, 2);
    EXPECT_EQ(config.bitsPerSample, 16);
}

TEST(RtpSessionMetricsTest, MetricsToJsonIncludesCounters) {
    SessionMetrics metrics;
    metrics.sessionId = "test";
    metrics.packetsReceived = 128;
    metrics.packetsDropped = 3;
    metrics.ssrc = 42;
    metrics.sampleRate = 48000;
    auto json = sessionMetricsToJson(metrics);
    EXPECT_EQ(json["session_id"], "test");
    EXPECT_EQ(json["packets_received"], 128);
    EXPECT_EQ(json["packets_dropped"], 3);
    EXPECT_EQ(json["ssrc"], 42);
    EXPECT_EQ(json["sample_rate"], 48000);
}
