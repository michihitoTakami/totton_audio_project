// CLI parsing for rpi_rtp_sender
#include "RtpOptions.h"

#include "Options.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {

std::string toLower(std::string_view s) {
    std::string out{s};
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
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

bool isValidLogLevel(std::string_view value) {
    const std::string lower = toLower(value);
    return lower == "debug" || lower == "d" || lower == "info" || lower == "i" || lower == "warn" ||
           lower == "warning" || lower == "w" || lower == "error" || lower == "err" || lower == "e";
}

bool applyEnvOverrides(RtpOptions &opt, ParseRtpOptionsResult &result,
                       const std::function<const char *(const char *)> &getenvFn) {
    auto fail = [&](const std::string &message) {
        result.hasError = true;
        result.errorMessage = message;
        return false;
    };

    if (const char *device = getenvFn("RTP_SENDER_DEVICE")) {
        opt.device = device;
    }
    if (const char *host = getenvFn("RTP_SENDER_HOST")) {
        opt.host = host;
    }
    if (const char *rtpPort = getenvFn("RTP_SENDER_RTP_PORT")) {
        auto parsed = parsePortValue(rtpPort);
        if (!parsed) {
            return fail("RTP_SENDER_RTP_PORT must be in 1-65535");
        }
        opt.rtpPort = *parsed;
    }
    if (const char *rtcpPort = getenvFn("RTP_SENDER_RTCP_PORT")) {
        auto parsed = parsePortValue(rtcpPort);
        if (!parsed) {
            return fail("RTP_SENDER_RTCP_PORT must be in 1-65535");
        }
        opt.rtcpSendPort = *parsed;
    }
    if (const char *rtcpListen = getenvFn("RTP_SENDER_RTCP_LISTEN_PORT")) {
        auto parsed = parsePortValue(rtcpListen);
        if (!parsed) {
            return fail("RTP_SENDER_RTCP_LISTEN_PORT must be in 1-65535");
        }
        opt.rtcpListenPort = *parsed;
    }
    if (const char *payload = getenvFn("RTP_SENDER_PAYLOAD_TYPE")) {
        const std::string buffer{payload};
        char *end = nullptr;
        long parsed = std::strtol(buffer.c_str(), &end, 10);
        if (!end || *end != '\0' || parsed < 0 || parsed > 127) {
            return fail("RTP_SENDER_PAYLOAD_TYPE must be 0-127");
        }
        opt.payloadType = static_cast<int>(parsed);
    }
    if (const char *format = getenvFn("RTP_SENDER_FORMAT")) {
        auto parsed = parseFormat(format);
        if (!parsed) {
            return fail("Unsupported RTP_SENDER_FORMAT. Use S16_LE | S24_3LE | S32_LE");
        }
        opt.formatOverride = *parsed;
    }
    if (const char *poll = getenvFn("RTP_SENDER_POLL_MS")) {
        const std::string buffer{poll};
        char *end = nullptr;
        long parsed = std::strtol(buffer.c_str(), &end, 10);
        if (!end || *end != '\0' || parsed < 50 || parsed > 5000) {
            return fail("RTP_SENDER_POLL_MS must be 50-5000");
        }
        opt.pollIntervalMs = static_cast<int>(parsed);
    }
    if (const char *notify = getenvFn("RTP_SENDER_NOTIFY_URL")) {
        opt.rateNotifyUrl = notify;
    }
    if (const char *logLevel = getenvFn("RTP_SENDER_LOG_LEVEL")) {
        if (!isValidLogLevel(logLevel)) {
            return fail("Unsupported RTP_SENDER_LOG_LEVEL. Use debug|info|warn|error");
        }
        opt.logLevel = parseLogLevel(logLevel);
    }
    if (const char *dryRun = getenvFn("RTP_SENDER_DRY_RUN")) {
        const std::string value{dryRun};
        if (value == "1" || toLower(value) == "true") {
            opt.dryRun = true;
        }
    }
    return true;
}

}  // namespace

ParseRtpOptionsResult parseRtpOptions(int argc, char **argv, std::string_view programName,
                                      const std::function<const char *(const char *)> &getenvFn) {
    RtpOptions opt{};
    ParseRtpOptionsResult result{};

    if (!applyEnvOverrides(opt, result, getenvFn)) {
        return result;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "-h" || arg == "--help") {
            printRtpHelp(programName);
            result.showHelp = true;
            return result;
        } else if (arg == "-V" || arg == "--version") {
            printRtpVersion(programName);
            result.showVersion = true;
            return result;
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            opt.device = argv[++i];
        } else if ((arg == "-H" || arg == "--host") && i + 1 < argc) {
            opt.host = argv[++i];
        } else if (arg == "--rtp-port" && i + 1 < argc) {
            auto parsed = parsePortValue(argv[++i]);
            if (!parsed) {
                result.hasError = true;
                result.errorMessage = "rtp-port must be in 1-65535";
                return result;
            }
            opt.rtpPort = *parsed;
        } else if (arg == "--rtcp-port" && i + 1 < argc) {
            auto parsed = parsePortValue(argv[++i]);
            if (!parsed) {
                result.hasError = true;
                result.errorMessage = "rtcp-port must be in 1-65535";
                return result;
            }
            opt.rtcpSendPort = *parsed;
        } else if (arg == "--rtcp-listen-port" && i + 1 < argc) {
            auto parsed = parsePortValue(argv[++i]);
            if (!parsed) {
                result.hasError = true;
                result.errorMessage = "rtcp-listen-port must be in 1-65535";
                return result;
            }
            opt.rtcpListenPort = *parsed;
        } else if (arg == "--payload-type" && i + 1 < argc) {
            const std::string buffer{argv[++i]};
            char *end = nullptr;
            long parsed = std::strtol(buffer.c_str(), &end, 10);
            if (!end || *end != '\0' || parsed < 0 || parsed > 127) {
                result.hasError = true;
                result.errorMessage = "payload-type must be 0-127";
                return result;
            }
            opt.payloadType = static_cast<int>(parsed);
        } else if (arg == "--format" && i + 1 < argc) {
            auto fmt = parseFormat(argv[++i]);
            if (!fmt) {
                result.hasError = true;
                result.errorMessage = "Unsupported format. Use S16_LE | S24_3LE | S32_LE";
                return result;
            }
            opt.formatOverride = *fmt;
        } else if (arg == "--log-level" && i + 1 < argc) {
            auto lvl = toLower(argv[++i]);
            if (!isValidLogLevel(lvl)) {
                result.hasError = true;
                result.errorMessage = "Unsupported log level. Use debug|info|warn|error";
                return result;
            }
            opt.logLevel = parseLogLevel(lvl);
        } else if (arg == "--poll-ms" && i + 1 < argc) {
            const std::string buffer{argv[++i]};
            char *end = nullptr;
            long parsed = std::strtol(buffer.c_str(), &end, 10);
            if (!end || *end != '\0' || parsed < 50 || parsed > 5000) {
                result.hasError = true;
                result.errorMessage = "poll-ms must be between 50 and 5000";
                return result;
            }
            opt.pollIntervalMs = static_cast<int>(parsed);
        } else if (arg == "--rate-notify-url" && i + 1 < argc) {
            opt.rateNotifyUrl = argv[++i];
        } else if (arg == "--dry-run") {
            opt.dryRun = true;
        } else {
            result.hasError = true;
            result.errorMessage = std::string("Unknown argument: ") + std::string(arg);
            return result;
        }
    }

    result.options = opt;
    return result;
}

void printRtpHelp(std::string_view programName) {
    std::cout << "Usage: " << programName
              << " [--device hw:0,0] [--host 127.0.0.1] [--rtp-port 46000] [--rtcp-port 46001]"
              << " [--rtcp-listen-port 46002] [--payload-type 96]"
              << " [--format S16_LE|S24_3LE|S32_LE] [--poll-ms 250] [--rate-notify-url URL]"
              << " [--log-level warn] [--dry-run] [--help] [--version]" << std::endl
              << std::endl
              << "RTP sender options:" << std::endl
              << "  -d, --device            ALSA capture device (e.g., hw:0,0)" << std::endl
              << "  -H, --host              Jetson RTP receiver host/IP" << std::endl
              << "  --rtp-port              RTP UDP port (1-65535)" << std::endl
              << "  --rtcp-port             RTCP send port to Jetson (1-65535)" << std::endl
              << "  --rtcp-listen-port      RTCP listen port from Jetson (1-65535)" << std::endl
              << "  --payload-type          RTP payload type (0-127, default 96)" << std::endl
              << "  --format                Override format (S16_LE|S24_3LE|S32_LE); if omitted, "
                 "uses ALSA hw_params"
              << std::endl
              << "  --poll-ms               hw_params polling interval (50-5000 ms)" << std::endl
              << "  --rate-notify-url       Optional URL to POST rate changes" << std::endl
              << "  --log-level             debug | info | warn | error" << std::endl
              << "  --dry-run               Do not spawn gst-launch; log pipeline only" << std::endl
              << "  -h, --help              Show this help and exit" << std::endl
              << "  -V, --version           Show version and exit" << std::endl
              << std::endl
              << "Environment overrides: RTP_SENDER_DEVICE, RTP_SENDER_HOST, RTP_SENDER_RTP_PORT, "
                 "RTP_SENDER_RTCP_PORT, RTP_SENDER_RTCP_LISTEN_PORT, RTP_SENDER_PAYLOAD_TYPE, "
                 "RTP_SENDER_FORMAT, RTP_SENDER_POLL_MS, RTP_SENDER_NOTIFY_URL, "
                 "RTP_SENDER_LOG_LEVEL, RTP_SENDER_DRY_RUN"
              << std::endl;
}

void printRtpVersion(std::string_view programName) {
    std::cout << programName << " version 0.1.0" << std::endl;
}
