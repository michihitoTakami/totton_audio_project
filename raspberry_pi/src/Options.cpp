#include "Options.h"

#include <cstdlib>
#include <iostream>

std::optional<AlsaCapture::SampleFormat> parseFormat(std::string_view value) {
    if (value == "S16_LE") {
        return AlsaCapture::SampleFormat::S16_LE;
    }
    if (value == "S24_3LE") {
        return AlsaCapture::SampleFormat::S24_3LE;
    }
    if (value == "S32_LE") {
        return AlsaCapture::SampleFormat::S32_LE;
    }
    return std::nullopt;
}

void printHelp(std::string_view programName) {
    std::cout << "Usage: " << programName << " [--device hw:0,0] [--rate 48000]"
              << " [--format S16_LE|S24_3LE|S32_LE] [--frames 4096]"
              << " [--iterations 3] [--help]" << std::endl
              << std::endl
              << "Prototype ALSA capture test entrypoint." << std::endl
              << "Opens the given ALSA device, reads a few periods, and exits." << std::endl;
}

std::optional<Options> parseOptions(int argc, char **argv, std::string_view programName) {
    Options opt{};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "-h" || arg == "--help") {
            printHelp(programName);
            return std::nullopt;
        } else if (arg == "--device" && i + 1 < argc) {
            opt.device = argv[++i];
        } else if (arg == "--rate" && i + 1 < argc) {
            opt.rate = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--format" && i + 1 < argc) {
            auto fmt = parseFormat(argv[++i]);
            if (!fmt) {
                std::cerr << "Unsupported format. Use one of: "
                          << "S16_LE | S24_3LE | S32_LE" << std::endl;
                return std::nullopt;
            }
            opt.format = *fmt;
        } else if (arg == "--frames" && i + 1 < argc) {
            opt.frames = static_cast<snd_pcm_uframes_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--iterations" && i + 1 < argc) {
            opt.iterations = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return std::nullopt;
        }
    }
    return opt;
}
