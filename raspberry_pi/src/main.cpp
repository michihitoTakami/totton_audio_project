#include "AlsaCapture.h"
#include "TcpClient.h"

#include <alsa/asoundlib.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string device{"hw:0,0"};
    unsigned int rate{48000};
    AlsaCapture::SampleFormat format{AlsaCapture::SampleFormat::S16_LE};
    snd_pcm_uframes_t frames{4096};
    int iterations{3};
};

std::optional<AlsaCapture::SampleFormat> parseFormat(std::string_view value)
{
    if(value == "S16_LE") {
        return AlsaCapture::SampleFormat::S16_LE;
    }
    if(value == "S24_3LE") {
        return AlsaCapture::SampleFormat::S24_3LE;
    }
    if(value == "S32_LE") {
        return AlsaCapture::SampleFormat::S32_LE;
    }
    return std::nullopt;
}

void printHelp(std::string_view programName)
{
    std::cout << "Usage: " << programName
              << " [--device hw:0,0] [--rate 48000]"
              << " [--format S16_LE|S24_3LE|S32_LE] [--frames 4096]"
              << " [--iterations 3] [--help]" << std::endl
              << std::endl
              << "Prototype ALSA capture test entrypoint." << std::endl
              << "Opens the given ALSA device, reads a few periods, and exits."
              << std::endl;
}

std::optional<Options> parseOptions(int argc, char **argv,
    std::string_view programName)
{
    Options opt{};
    for(int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if(arg == "-h" || arg == "--help") {
            printHelp(programName);
            return std::nullopt;
        } else if(arg == "--device" && i + 1 < argc) {
            opt.device = argv[++i];
        } else if(arg == "--rate" && i + 1 < argc) {
            opt.rate = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
        } else if(arg == "--format" && i + 1 < argc) {
            auto fmt = parseFormat(argv[++i]);
            if(!fmt) {
                std::cerr << "Unsupported format. Use one of: "
                          << "S16_LE | S24_3LE | S32_LE" << std::endl;
                return std::nullopt;
            }
            opt.format = *fmt;
        } else if(arg == "--frames" && i + 1 < argc) {
            opt.frames = static_cast<snd_pcm_uframes_t>(
                std::strtoul(argv[++i], nullptr, 10));
        } else if(arg == "--iterations" && i + 1 < argc) {
            opt.iterations = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return std::nullopt;
        }
    }
    return opt;
}

} // namespace

int main(int argc, char **argv)
{
    const std::string_view programName = (argc > 0 && argv[0] != nullptr)
        ? std::string_view{argv[0]}
        : "rpi_pcm_bridge";

    auto parsed = parseOptions(argc, argv, programName);
    if(!parsed) {
        return EXIT_SUCCESS;
    }
    const Options opt = *parsed;

    AlsaCapture capture;
    AlsaCapture::Config cfg;
    cfg.deviceName = opt.device;
    cfg.sampleRate = opt.rate;
    cfg.channels = 2;
    cfg.format = opt.format;
    cfg.periodFrames = opt.frames;

    if(!capture.open(cfg)) {
        std::cerr << "[rpi_pcm_bridge] Failed to open device" << std::endl;
        return EXIT_FAILURE;
    }
    if(!capture.start()) {
        std::cerr << "[rpi_pcm_bridge] Failed to start capture" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::uint8_t> buffer;
    for(int i = 0; i < opt.iterations; ++i) {
        int bytes = capture.read(buffer);
        if(bytes == -EPIPE) {
            std::clog << "[rpi_pcm_bridge] XRUN recovered, continuing"
                      << std::endl;
            continue;
        }
        if(bytes < 0) {
            std::clog << "[rpi_pcm_bridge] Read failed: " << bytes << std::endl;
            break;
        }
        std::clog << "[rpi_pcm_bridge] Read " << bytes << " bytes" << std::endl;
    }

    capture.stop();
    capture.close();

    return EXIT_SUCCESS;
}

