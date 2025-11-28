#include "rtp_session_manager.h"

#include <gtest/gtest.h>

using namespace Network;

TEST(RtpSessionConfigTest, ParseValidJson) {
    nlohmann::json params = {
        {"session_id", "primary"},
        {"port", 7000},
        {"sample_rate", 96000},
        {"bits_per_sample", 32},
        {"payload_type", 98},
        {"enable_rtcp", false}};

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

