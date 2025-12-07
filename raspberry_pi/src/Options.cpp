#include "Options.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

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

std::optional<std::uint16_t> parsePortValue(std::string_view value) {
    const std::string buffer{value};
    char *end = nullptr;
    unsigned long parsed = std::strtoul(buffer.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return std::nullopt;
    }
    if (parsed == 0 || parsed > 65535) {
        return std::nullopt;
    }
    return static_cast<std::uint16_t>(parsed);
}

std::optional<unsigned int> parseRateValue(std::string_view value) {
    const std::string buffer{value};
    char *end = nullptr;
    unsigned long parsed = std::strtoul(buffer.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return std::nullopt;
    }
    const auto rate = static_cast<unsigned int>(parsed);
    if (!isAllowedRate(rate)) {
        return std::nullopt;
    }
    return rate;
}

bool isValidLogLevel(std::string_view value) {
    const std::string lower = toLower(value);
    return lower == "debug" || lower == "d" || lower == "info" || lower == "i" || lower == "warn" ||
           lower == "warning" || lower == "w" || lower == "error" || lower == "err" || lower == "e";
}

bool applyEnvOverrides(Options &opt, ParseOptionsResult &result,
                       const std::function<const char *(const char *)> &getenvFn) {
    auto fail = [&](const std::string &message) {
        result.hasError = true;
        result.errorMessage = message;
        return false;
    };

    if (const char *device = getenvFn("PCM_BRIDGE_DEVICE")) {
        opt.device = device;
    }
    if (const char *host = getenvFn("PCM_BRIDGE_HOST")) {
        opt.host = host;
    }
    if (const char *port = getenvFn("PCM_BRIDGE_PORT")) {
        auto parsed = parsePortValue(port);
        if (!parsed) {
            return fail("PCM_BRIDGE_PORT must be in 1-65535");
        }
        opt.port = *parsed;
    }
    if (const char *rate = getenvFn("PCM_BRIDGE_RATE")) {
        auto parsed = parseRateValue(rate);
        if (!parsed) {
            return fail(
                "Unsupported PCM_BRIDGE_RATE. Allowed: "
                "44100|48000|88200|96000|176400|192000|352800|384000|705600|768000");
        }
        opt.rate = *parsed;
    }
    if (const char *fmt = getenvFn("PCM_BRIDGE_FORMAT")) {
        auto parsed = parseFormat(fmt);
        if (!parsed) {
            return fail("Unsupported PCM_BRIDGE_FORMAT. Use one of: S16_LE | S24_3LE | S32_LE");
        }
        opt.format = *parsed;
    }
    if (const char *frames = getenvFn("PCM_BRIDGE_FRAMES")) {
        const std::string buffer{frames};
        char *end = nullptr;
        unsigned long parsed = std::strtoul(buffer.c_str(), &end, 10);
        if (!end || *end != '\0' || parsed == 0) {
            return fail("PCM_BRIDGE_FRAMES must be a positive integer");
        }
        opt.frames = static_cast<snd_pcm_uframes_t>(parsed);
    }
    if (const char *logLevel = getenvFn("PCM_BRIDGE_LOG_LEVEL")) {
        if (!isValidLogLevel(logLevel)) {
            return fail("Unsupported PCM_BRIDGE_LOG_LEVEL. Use one of: debug|info|warn|error");
        }
        opt.logLevel = parseLogLevel(logLevel);
    }
    if (const char *iterations = getenvFn("PCM_BRIDGE_ITERATIONS")) {
        const std::string buffer{iterations};
        char *end = nullptr;
        long parsed = std::strtol(buffer.c_str(), &end, 10);
        if (!end || *end != '\0') {
            return fail("PCM_BRIDGE_ITERATIONS must be an integer (negative for infinite)");
        }
        opt.iterations = static_cast<int>(parsed);
    }
    return true;
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
              << " [--iterations -1] [--help] [--version]" << std::endl
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
              << "  --iterations     Loop count (-1 = infinite)" << std::endl
              << "  -h, --help       Show this help and exit" << std::endl
              << "  -V, --version    Show version and exit" << std::endl
              << std::endl
              << "Environment overrides: PCM_BRIDGE_DEVICE, PCM_BRIDGE_HOST, PCM_BRIDGE_PORT, "
                 "PCM_BRIDGE_RATE, PCM_BRIDGE_FORMAT, PCM_BRIDGE_FRAMES, PCM_BRIDGE_LOG_LEVEL, "
                 "PCM_BRIDGE_ITERATIONS"
              << std::endl;
}

ParseOptionsResult parseOptions(int argc, char **argv, std::string_view programName,
                                const std::function<const char *(const char *)> &getenvFn) {
    Options opt{};
    ParseOptionsResult result{};

    if (!applyEnvOverrides(opt, result, getenvFn)) {
        return result;
    }

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
            auto parsed = parsePortValue(argv[++i]);
            if (!parsed) {
                result.hasError = true;
                result.errorMessage = "Port must be in 1-65535";
                return result;
            }
            opt.port = *parsed;
        } else if ((arg == "-r" || arg == "--rate") && i + 1 < argc) {
            auto parsed = parseRateValue(argv[++i]);
            if (!parsed) {
                result.hasError = true;
                result.errorMessage =
                    "Unsupported rate. Allowed: "
                    "44100|48000|88200|96000|176400|192000|352800|384000|705600|768000";
                return result;
            }
            opt.rate = *parsed;
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
            if (!isValidLogLevel(lvl)) {
                result.hasError = true;
                result.errorMessage = "Unsupported log level. Use one of: debug|info|warn|error";
                return result;
            }
            opt.logLevel = parseLogLevel(lvl);
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
