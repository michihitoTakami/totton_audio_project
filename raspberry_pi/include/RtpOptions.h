// RTP sender CLI options for Raspberry Pi GStreamer pipeline
#pragma once

#include "AlsaCapture.h"
#include "logging.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

struct RtpOptions {
    std::string device{"hw:0,0"};
    std::string host{"127.0.0.1"};
    std::uint16_t rtpPort{46000};
    std::uint16_t rtcpSendPort{46001};
    std::uint16_t rtcpListenPort{46002};
    int payloadType{96};
    std::optional<AlsaCapture::SampleFormat> formatOverride{};
    LogLevel logLevel{LogLevel::Warn};
    int pollIntervalMs{250};
    int latencyMs{100};
    std::string rateNotifyUrl{};
    bool dryRun{false};
};

struct ParseRtpOptionsResult {
    std::optional<RtpOptions> options;
    bool showHelp{false};
    bool showVersion{false};
    bool hasError{false};
    std::string errorMessage;
};

ParseRtpOptionsResult parseRtpOptions(
    int argc, char **argv, std::string_view programName,
    const std::function<const char *(const char *)> &getenvFn = ::getenv);
void printRtpHelp(std::string_view programName);
void printRtpVersion(std::string_view programName);
