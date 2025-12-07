#pragma once

#include "AlsaCapture.h"

#include <optional>
#include <string>
#include <string_view>

struct Options {
    std::string device{"hw:0,0"};
    unsigned int rate{48000};
    AlsaCapture::SampleFormat format{AlsaCapture::SampleFormat::S16_LE};
    snd_pcm_uframes_t frames{4096};
    int iterations{3};
};

std::optional<AlsaCapture::SampleFormat> parseFormat(std::string_view value);
std::optional<Options> parseOptions(int argc, char **argv, std::string_view programName);
void printHelp(std::string_view programName);
