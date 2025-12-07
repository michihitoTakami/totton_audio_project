#include "Options.h"

#include <array>
#include <cstdlib>
#include <iostream>

namespace {

bool isSupportedRate(unsigned int rate) {
    static constexpr std::array<unsigned int, 10> kSupported = {
        44100, 88200, 176400, 352800, 705600, 48000, 96000, 192000, 384000, 768000};
    for (const auto v : kSupported) {
        if (rate == v) {
            return true;
        }
    }
    return false;
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

std::optional<Options::LogLevel> parseLogLevel(std::string_view value) {
    if (value == "trace") {
        return Options::LogLevel::Trace;
    }
    if (value == "debug") {
        return Options::LogLevel::Debug;
    }
    if (value == "info") {
        return Options::LogLevel::Info;
    }
    if (value == "warn" || value == "warning") {
        return Options::LogLevel::Warn;
    }
    if (value == "error") {
        return Options::LogLevel::Error;
    }
    return std::nullopt;
}

std::string_view toString(Options::LogLevel level) {
    switch (level) {
    case Options::LogLevel::Trace:
        return "trace";
    case Options::LogLevel::Debug:
        return "debug";
    case Options::LogLevel::Info:
        return "info";
    case Options::LogLevel::Warn:
        return "warn";
    case Options::LogLevel::Error:
        return "error";
    }
    return "info";
}

void printHelp(std::string_view programName) {
    std::cout << "Usage: " << programName << " [--device hw:0,0] [--host 127.0.0.1] [--port 46001]"
              << " [--rate 48000] [--format S16_LE|S24_3LE|S32_LE]"
              << " [--frames 4096] [--iterations 3]"
              << " [--log-level trace|debug|info|warn|error]"
              << " [--help] [--version]" << std::endl
              << std::endl
              << "Prototype ALSA capture test entrypoint." << std::endl
              << "Opens the given ALSA device, reads a few periods, and exits." << std::endl;
}

void printVersion() {
    std::cout << "rpi_pcm_bridge version " << kAppVersion << std::endl;
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
            printVersion();
            result.showVersion = true;
            return result;
        } else if (arg == "-d" || arg == "--device") {
            if (i + 1 >= argc) {
                result.hasError = true;
                result.errorMessage = "--device requires a value";
                return result;
            }
            opt.device = argv[++i];
        } else if (arg == "-H" || arg == "--host") {
            if (i + 1 >= argc) {
                result.hasError = true;
                result.errorMessage = "--host requires a value";
                return result;
            }
            opt.host = argv[++i];
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 >= argc) {
                result.hasError = true;
                result.errorMessage = "--port requires a value";
                return result;
            }
            const unsigned long portVal = std::strtoul(argv[++i], nullptr, 10);
            if (portVal == 0 || portVal > 65535) {
                result.hasError = true;
                result.errorMessage = "Port must be in range 1-65535";
                return result;
            }
            opt.port = static_cast<std::uint16_t>(portVal);
        } else if (arg == "-r" || arg == "--rate") {
            if (i + 1 >= argc) {
                result.hasError = true;
                result.errorMessage = "--rate requires a value";
                return result;
            }
            opt.rate = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
            if (!isSupportedRate(opt.rate)) {
                result.hasError = true;
                result.errorMessage = "Unsupported rate. Use 44.1/48k and 2/4/8/16x multiples.";
                return result;
            }
        } else if (arg == "-f" || arg == "--format") {
            if (i + 1 >= argc) {
                result.hasError = true;
                result.errorMessage = "--format requires a value";
                return result;
            }
            auto fmt = parseFormat(argv[++i]);
            if (!fmt) {
                result.hasError = true;
                result.errorMessage = "Unsupported format. Use one of: S16_LE | S24_3LE | S32_LE";
                return result;
            }
            opt.format = *fmt;
        } else if (arg == "--frames") {
            if (i + 1 >= argc) {
                result.hasError = true;
                result.errorMessage = "--frames requires a value";
                return result;
            }
            opt.frames = static_cast<snd_pcm_uframes_t>(std::strtoul(argv[++i], nullptr, 10));
            if (opt.frames == 0) {
                result.hasError = true;
                result.errorMessage = "Frames must be greater than zero";
                return result;
            }
        } else if (arg == "--iterations") {
            if (i + 1 >= argc) {
                result.hasError = true;
                result.errorMessage = "--iterations requires a value";
                return result;
            }
            opt.iterations = std::atoi(argv[++i]);
            if (opt.iterations <= 0) {
                result.hasError = true;
                result.errorMessage = "Iterations must be greater than zero";
                return result;
            }
        } else if (arg == "-l" || arg == "--log-level") {
            if (i + 1 >= argc) {
                result.hasError = true;
                result.errorMessage = "--log-level requires a value";
                return result;
            }
            auto level = parseLogLevel(argv[++i]);
            if (!level) {
                result.hasError = true;
                result.errorMessage = "Unsupported log level. Use trace|debug|info|warn|error";
                return result;
            }
            opt.logLevel = *level;
        } else {
            result.hasError = true;
            result.errorMessage = std::string("Unknown argument: ") + std::string(arg);
            return result;
        }
    }
    if (opt.host.empty()) {
        result.hasError = true;
        result.errorMessage = "Host must not be empty";
        return result;
    }
    if (opt.device.empty()) {
        result.hasError = true;
        result.errorMessage = "Device must not be empty";
        return result;
    }
    result.options = opt;
    return result;
}
