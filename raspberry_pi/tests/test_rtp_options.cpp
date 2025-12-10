// Tests for rpi_rtp_sender options
#include "RtpOptions.h"

#include <functional>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::vector<char *> makeArgv(const std::vector<std::string> &args) {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (const auto &arg : args) {
        argv.push_back(const_cast<char *>(arg.c_str()));
    }
    return argv;
}

using EnvMap = std::unordered_map<std::string, std::string>;

std::function<const char *(const char *)> makeGetEnv(const EnvMap &env) {
    return [&env](const char *name) -> const char * {
        auto it = env.find(name ? std::string{name} : std::string{});
        if (it == env.end()) {
            return nullptr;
        }
        return it->second.c_str();
    };
}

}  // namespace

TEST(RtpOptions, DefaultsApplied) {
    std::vector<std::string> args = {"rpi_rtp_sender"};
    auto argv = makeArgv(args);
    auto parsed = parseRtpOptions(static_cast<int>(argv.size()), argv.data(), "rpi_rtp_sender");

    ASSERT_FALSE(parsed.hasError);
    ASSERT_TRUE(parsed.options.has_value());
    EXPECT_EQ(parsed.options->device, "hw:0,0");
    EXPECT_EQ(parsed.options->host, "127.0.0.1");
    EXPECT_EQ(parsed.options->rtpPort, 46000);
    EXPECT_EQ(parsed.options->rtcpSendPort, 46001);
    EXPECT_EQ(parsed.options->rtcpListenPort, 46002);
    EXPECT_EQ(parsed.options->payloadType, 96);
    EXPECT_EQ(parsed.options->pollIntervalMs, 250);
    EXPECT_EQ(parsed.options->logLevel, LogLevel::Warn);
    EXPECT_FALSE(parsed.options->formatOverride.has_value());
}

TEST(RtpOptions, ParsesCliArguments) {
    std::vector<std::string> args = {"rpi_rtp_sender",
                                     "-d",
                                     "hw:2,0",
                                     "-H",
                                     "192.168.0.9",
                                     "--rtp-port",
                                     "50000",
                                     "--rtcp-port",
                                     "50001",
                                     "--rtcp-listen-port",
                                     "50002",
                                     "--payload-type",
                                     "97",
                                     "--format",
                                     "S24_3LE",
                                     "--poll-ms",
                                     "300",
                                     "--log-level",
                                     "debug",
                                     "--rate-notify-url",
                                     "http://localhost/rate",
                                     "--dry-run"};
    auto argv = makeArgv(args);
    auto parsed = parseRtpOptions(static_cast<int>(argv.size()), argv.data(), "rpi_rtp_sender");

    ASSERT_FALSE(parsed.hasError);
    ASSERT_TRUE(parsed.options.has_value());
    EXPECT_EQ(parsed.options->device, "hw:2,0");
    EXPECT_EQ(parsed.options->host, "192.168.0.9");
    EXPECT_EQ(parsed.options->rtpPort, 50000);
    EXPECT_EQ(parsed.options->rtcpSendPort, 50001);
    EXPECT_EQ(parsed.options->rtcpListenPort, 50002);
    EXPECT_EQ(parsed.options->payloadType, 97);
    ASSERT_TRUE(parsed.options->formatOverride.has_value());
    EXPECT_EQ(parsed.options->formatOverride.value(), AlsaCapture::SampleFormat::S24_3LE);
    EXPECT_EQ(parsed.options->pollIntervalMs, 300);
    EXPECT_EQ(parsed.options->logLevel, LogLevel::Debug);
    EXPECT_EQ(parsed.options->rateNotifyUrl, "http://localhost/rate");
    EXPECT_TRUE(parsed.options->dryRun);
}

TEST(RtpOptions, EnvironmentOverrides) {
    std::vector<std::string> args = {"rpi_rtp_sender"};
    auto argv = makeArgv(args);

    EnvMap env = {{"RTP_SENDER_DEVICE", "hw:9,0"},
                  {"RTP_SENDER_HOST", "10.0.0.10"},
                  {"RTP_SENDER_RTP_PORT", "55000"},
                  {"RTP_SENDER_RTCP_PORT", "55001"},
                  {"RTP_SENDER_RTCP_LISTEN_PORT", "55002"},
                  {"RTP_SENDER_PAYLOAD_TYPE", "98"},
                  {"RTP_SENDER_FORMAT", "S32_LE"},
                  {"RTP_SENDER_POLL_MS", "500"},
                  {"RTP_SENDER_LOG_LEVEL", "info"},
                  {"RTP_SENDER_NOTIFY_URL", "http://example.com"},
                  {"RTP_SENDER_DRY_RUN", "true"}};

    auto parsed = parseRtpOptions(static_cast<int>(argv.size()), argv.data(), "rpi_rtp_sender",
                                  makeGetEnv(env));

    ASSERT_FALSE(parsed.hasError);
    ASSERT_TRUE(parsed.options.has_value());
    EXPECT_EQ(parsed.options->device, "hw:9,0");
    EXPECT_EQ(parsed.options->host, "10.0.0.10");
    EXPECT_EQ(parsed.options->rtpPort, 55000);
    EXPECT_EQ(parsed.options->rtcpSendPort, 55001);
    EXPECT_EQ(parsed.options->rtcpListenPort, 55002);
    EXPECT_EQ(parsed.options->payloadType, 98);
    ASSERT_TRUE(parsed.options->formatOverride.has_value());
    EXPECT_EQ(parsed.options->formatOverride.value(), AlsaCapture::SampleFormat::S32_LE);
    EXPECT_EQ(parsed.options->pollIntervalMs, 500);
    EXPECT_EQ(parsed.options->logLevel, LogLevel::Info);
    EXPECT_EQ(parsed.options->rateNotifyUrl, "http://example.com");
    EXPECT_TRUE(parsed.options->dryRun);
}

TEST(RtpOptions, RejectsInvalidPort) {
    std::vector<std::string> args = {"rpi_rtp_sender", "--rtp-port", "70000"};
    auto argv = makeArgv(args);
    auto parsed = parseRtpOptions(static_cast<int>(argv.size()), argv.data(), "rpi_rtp_sender");

    EXPECT_TRUE(parsed.hasError);
    EXPECT_FALSE(parsed.options.has_value());
}
