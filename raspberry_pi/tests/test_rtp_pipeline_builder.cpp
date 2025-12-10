// Tests for RtpPipelineBuilder
#include "RtpPipelineBuilder.h"

#include <gtest/gtest.h>
#include <string>

TEST(RtpPipelineBuilder, BuildsL24Pipeline) {
    RtpPipelineConfig cfg;
    cfg.host = "192.168.0.20";
    cfg.rtpPort = 47000;
    cfg.rtcpSendPort = 47001;
    cfg.rtcpListenPort = 47002;
    cfg.payloadType = 96;
    cfg.device = "hw:1,0";

    CaptureParams params;
    params.sampleRate = 96000;
    params.channels = 2;
    params.format = AlsaCapture::SampleFormat::S24_3LE;

    auto args = RtpPipelineBuilder::build(cfg, params);
    const std::string cmd = RtpPipelineBuilder::toCommandString(args);

    EXPECT_NE(cmd.find("audio/x-raw,rate=96000,channels=2,format=S24LE"), std::string::npos);
    EXPECT_NE(cmd.find("rtpL24pay"), std::string::npos);
    EXPECT_NE(cmd.find("clock-rate=96000"), std::string::npos);
    EXPECT_NE(cmd.find("port=47000"), std::string::npos);
    EXPECT_NE(cmd.find("port=47001"), std::string::npos);
    EXPECT_NE(cmd.find("port=47002"), std::string::npos);
}

TEST(RtpPipelineBuilder, UsesEncodingForFormat) {
    RtpPipelineConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.rtpPort = 46000;
    cfg.rtcpSendPort = 46001;
    cfg.rtcpListenPort = 46002;
    cfg.payloadType = 97;
    cfg.device = "hw:0,0";

    CaptureParams params;
    params.sampleRate = 48000;
    params.channels = 2;
    params.format = AlsaCapture::SampleFormat::S16_LE;

    const std::string cmd =
        RtpPipelineBuilder::toCommandString(RtpPipelineBuilder::build(cfg, params));

    EXPECT_NE(cmd.find("rtpL16pay"), std::string::npos);
    EXPECT_NE(cmd.find("encoding-name=L16"), std::string::npos);
    EXPECT_NE(cmd.find("payload=97"), std::string::npos);
}
