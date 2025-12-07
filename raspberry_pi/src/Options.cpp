#include "Options.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>

namespace {

const std::array<unsigned int, 10> kAllowedRates = {44100,  48000,  88200,  96000,  176400,
                                                    192000, 352800, 384000, 705600, 768000};

std::string toLower(std::string_view s) {
    std::string out{s};
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

bool isAllowedRate(unsigned int rate) {
    return std::find(kAllowedRates.begin(), kAllowedRates.end(), rate) != kAllowedRates.end();
}

}  // namespace

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
    std::cout << "Usage: " << programName << " [--device hw:0,0] [--host 127.0.0.1] [--port 46001]"
              << " [--rate 48000] [--format S16_LE|S24_3LE|S32_LE]"
              << " [--frames 4096] [--log-level info]"
              << " [--iterations 3] [--help] [--version]" << std::endl
              << std::endl
              << "PCM bridge CLI options:" << std::endl
              << "  -d, --device     ALSA device name (e.g., hw:0,0)" << std::endl
              << "  -H, --host       Destination host/IP for TCP server" << std::endl
              << "  -p, --port       Destination TCP port (1-65535)" << std::endl
              << "  -r, --rate       Sample rate "
                 "(44100|48000|88200|96000|176400|192000|352800|384000|705600|768000)"
              << std::endl
              << "  -f, --format     Sample format: S16_LE | S24_3LE | S32_LE" << std::endl
              << "  --frames         ALSA period frames (capture chunk size)" << std::endl
              << "  --log-level      Log level: debug | info | warn | error" << std::endl
              << "  --iterations     (Test only) loop count before exit" << std::endl
              << "  -h, --help       Show this help and exit" << std::endl
              << "  -V, --version    Show version and exit" << std::endl;
}

ParseOptionsResult parseOptions(int argc, char **argv, std::string_view programName) {
    Options opt{};
    ParseOptionsResult result{};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "-h" || arg == "--help") {
            printHelp(programName);
            result.showHelp = true;
            return result;
        } else if (arg == "-V" || arg == "--version") {
            printVersion(programName);
            result.showVersion = true;
            return result;
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            opt.device = argv[++i];
        } else if ((arg == "-H" || arg == "--host") && i + 1 < argc) {
            opt.host = argv[++i];
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            auto port = std::strtoul(argv[++i], nullptr, 10);
            if (port == 0 || port > 65535) {
                result.hasError = true;
                result.errorMessage = "Port must be in 1-65535";
                return result;
            }
            opt.port = static_cast<std::uint16_t>(port);
        } else if ((arg == "-r" || arg == "--rate") && i + 1 < argc) {
            auto rate = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
            if (!isAllowedRate(rate)) {
                result.hasError = true;
                result.errorMessage =
                    "Unsupported rate. Allowed: "
                    "44100|48000|88200|96000|176400|192000|352800|384000|705600|768000";
                return result;
            }
            opt.rate = rate;
        } else if ((arg == "-f" || arg == "--format") && i + 1 < argc) {
            auto fmt = parseFormat(argv[++i]);
            if (!fmt) {
                result.hasError = true;
                result.errorMessage = "Unsupported format. Use one of: S16_LE | S24_3LE | S32_LE";
                return result;
            }
            opt.format = *fmt;
        } else if (arg == "--frames" && i + 1 < argc) {
            opt.frames = static_cast<snd_pcm_uframes_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--iterations" && i + 1 < argc) {
            opt.iterations = std::atoi(argv[++i]);
        } else if (arg == "--log-level" && i + 1 < argc) {
            auto lvl = toLower(argv[++i]);
            if (lvl != "debug" && lvl != "info" && lvl != "warn" && lvl != "error") {
                result.hasError = true;
                result.errorMessage = "Unsupported log level. Use one of: debug|info|warn|error";
                return result;
            }
            opt.logLevel = lvl;
        } else {
            result.hasError = true;
            result.errorMessage = std::string("Unknown argument: ") + std::string(arg);
            return result;
        }
    }
    result.options = opt;
    return result;
}

void printVersion(std::string_view programName) {
    std::cout << programName << " version 0.1.0" << std::endl;
}
