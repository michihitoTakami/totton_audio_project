#pragma once

#include "AlsaCapture.h"

#include <cstdint>
#include <functional>
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
    std::string logLevel{"info"};
    int iterations{-1};  // 負の値は無限ループで送信を続ける。
};

struct ParseOptionsResult {
    std::optional<Options> options;
    bool showHelp{false};
    bool showVersion{false};
    bool hasError{false};
    std::string errorMessage;
};

std::optional<AlsaCapture::SampleFormat> parseFormat(std::string_view value);
ParseOptionsResult parseOptions(
    int argc, char **argv, std::string_view programName,
    const std::function<const char *(const char *)> &getenvFn = ::getenv);
void printHelp(std::string_view programName);
void printVersion(std::string_view programName);
