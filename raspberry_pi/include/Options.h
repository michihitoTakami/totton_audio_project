#pragma once

#include "AlsaCapture.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

struct Options {
    std::string device{"hw:0,0"};
    std::string host{"127.0.0.1"};
    std::uint16_t port{46001};
    unsigned int rate{48000};
    AlsaCapture::SampleFormat format{AlsaCapture::SampleFormat::S16_LE};
    snd_pcm_uframes_t frames{4096};
    int iterations{3};
    enum class LogLevel {
        Trace,
        Debug,
        Info,
        Warn,
        Error,
    } logLevel{LogLevel::Info};
};

struct ParseOptionsResult {
    std::optional<Options> options;
    bool showHelp{false};
    bool showVersion{false};
    bool hasError{false};
    std::string errorMessage;
};

std::optional<AlsaCapture::SampleFormat> parseFormat(std::string_view value);
ParseOptionsResult parseOptions(int argc, char **argv, std::string_view programName);
void printHelp(std::string_view programName);
void printVersion();
std::string_view toString(Options::LogLevel level);
std::optional<Options::LogLevel> parseLogLevel(std::string_view value);
constexpr std::string_view kAppVersion = "0.1.0";
