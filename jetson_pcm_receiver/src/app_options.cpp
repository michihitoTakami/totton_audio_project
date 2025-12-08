#include "app_options.h"

#include "output_device.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool parseBool(const std::string &value, bool &out) {
    if (value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "yes") {
        out = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "FALSE" || value == "off" || value == "no") {
        out = false;
        return true;
    }
    return false;
}

bool parsePriorityList(const std::string &value, std::vector<std::string> &out) {
    std::string current;
    for (char c : value) {
        if (c == ',') {
            if (!current.empty()) {
                out.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return true;
}

bool parseEnvInt(const char *name, int &target, std::string &error) {
    if (const char *env = std::getenv(name)) {
        char *end = nullptr;
        long value = std::strtol(env, &end, 10);
        if (end && *end == '\0') {
            target = static_cast<int>(value);
            return true;
        }
        error = std::string("環境変数 ") + name + " の値が整数として解釈できません: " + env;
        return false;
    }
    return true;
}

bool parseEnvSize(const char *name, std::size_t &target, std::string &error) {
    if (const char *env = std::getenv(name)) {
        char *end = nullptr;
        unsigned long value = std::strtoul(env, &end, 10);
        if (end && *end == '\0') {
            target = static_cast<std::size_t>(value);
            return true;
        }
        error = std::string("環境変数 ") + name + " の値が数値として解釈できません: " + env;
        return false;
    }
    return true;
}

OutputDeviceSpec defaultDevice() {
    auto parsed = parseOutputDevice("loopback");
    return parsed.ok ? parsed.spec : OutputDeviceSpec{};
}

}  // namespace

AppOptions makeDefaultOptions() {
    AppOptions options;
    options.device = defaultDevice();
    return options;
}

bool applyEnvOverrides(AppOptions &options, std::string &error) {
    if (!parseEnvInt("JPR_PORT", options.port, error)) {
        return false;
    }
    if (const char *dev = std::getenv("JPR_DEVICE")) {
        auto parsed = parseOutputDevice(dev);
        if (!parsed.ok) {
            error = std::string("JPR_DEVICEの解析に失敗しました: ") + parsed.error;
            return false;
        }
        options.device = parsed.spec;
    }
    if (const char *lvl = std::getenv("JPR_LOG_LEVEL")) {
        options.logLevel = parseLogLevel(lvl);
    }
    if (!parseEnvSize("JPR_RING_FRAMES", options.ringBufferFrames, error)) {
        return false;
    }
    if (!parseEnvSize("JPR_WATERMARK_FRAMES", options.watermarkFrames, error)) {
        return false;
    }
    if (const char *rbDisable = std::getenv("JPR_DISABLE_RING_BUFFER")) {
        bool disabled = false;
        if (!parseBool(rbDisable, disabled)) {
            error = "JPR_DISABLE_RING_BUFFER は true/false で指定してください";
            return false;
        }
        if (disabled) {
            options.ringBufferFrames = 0;
        }
    }
    if (!parseEnvInt("JPR_RECV_TIMEOUT_MS", options.recvTimeoutMs, error)) {
        return false;
    }
    if (!parseEnvInt("JPR_RECV_TIMEOUT_SLEEP_MS", options.recvTimeoutSleepMs, error)) {
        return false;
    }
    if (!parseEnvInt("JPR_ACCEPT_COOLDOWN_MS", options.acceptCooldownMs, error)) {
        return false;
    }
    if (!parseEnvInt("JPR_MAX_CONSEC_TIMEOUTS", options.maxConsecutiveTimeouts, error)) {
        return false;
    }

    if (const char *cm = std::getenv("JPR_CONNECTION_MODE")) {
        options.connectionMode = parseConnectionMode(cm);
    }
    if (const char *priority = std::getenv("JPR_PRIORITY_CLIENTS")) {
        options.priorityClients.clear();
        parsePriorityList(priority, options.priorityClients);
    }

    if (const char *zmqEndpoint = std::getenv("JPR_ZMQ_ENDPOINT")) {
        options.zmqEndpoint = zmqEndpoint;
    }
    if (const char *zmqToken = std::getenv("JPR_ZMQ_TOKEN")) {
        options.zmqToken = zmqToken;
    }
    if (!parseEnvInt("JPR_ZMQ_PUB_INTERVAL_MS", options.zmqPublishIntervalMs, error)) {
        return false;
    }
    if (const char *zmqDisable = std::getenv("JPR_DISABLE_ZMQ")) {
        bool disable = false;
        if (!parseBool(zmqDisable, disable)) {
            error = "JPR_DISABLE_ZMQ は true/false で指定してください";
            return false;
        }
        options.enableZmq = !disable;
    }
    if (const char *rateNotify = std::getenv("JPR_ENABLE_ZMQ_RATE_NOTIFY")) {
        bool enabled = false;
        if (!parseBool(rateNotify, enabled)) {
            error = "JPR_ENABLE_ZMQ_RATE_NOTIFY は true/false で指定してください";
            return false;
        }
        options.enableZmqRateNotify = enabled;
    }

    return true;
}

void printHelp(const char *exeName) {
    std::cout << "jetson-pcm-receiver\n";
    std::cout << "Usage: " << exeName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -p, --port <number>     TCP listen port (default: 46001)\n";
    std::cout << "  -d, --device <name>     Output device alias/ALSA name (default: loopback)\n";
    std::cout << "                          Accepted: loopback | null | alsa:<pcm> | <raw pcm>\n";
    std::cout << "  -l, --log-level <lvl>   error/warn/info/debug (default: info)\n";
    std::cout << "  --ring-buffer-frames N  enable jitter buffer with N frames (default: 8192)\n";
    std::cout
        << "  --ring-buffer-watermark N  watermark frames for warning (default: 75% of buffer)\n";
    std::cout << "  --no-ring-buffer        disable jitter buffer\n";
    std::cout << "  --recv-timeout-ms N     per-recv timeout before EAGAIN (default: 250)\n";
    std::cout << "  --recv-timeout-sleep-ms N  sleep between recv timeouts (default: 50)\n";
    std::cout << "  --accept-cooldown-ms N  cooldown before re-accept (default: 250)\n";
    std::cout << "  --connection-mode MODE  single|takeover|priority (default: single)\n";
    std::cout << "  --priority-client ADDR  priority client IP (repeatable)\n";
    std::cout << "  --disable-zmq           disable ZeroMQ status/control API\n";
    std::cout << "  --enable-zmq            explicitly enable ZeroMQ API (default: on)\n";
    std::cout << "  --disable-zmq-rate-notify  disable header change PUB events\n";
    std::cout << "  --zmq-endpoint <uri>    ZeroMQ REP endpoint (default: "
                 "ipc:///tmp/jetson_pcm_receiver.sock)\n";
    std::cout << "  --zmq-token <token>     shared token for ZeroMQ commands\n";
    std::cout
        << "  --zmq-pub-interval <ms> status publish interval for PUB socket (default: 1000)\n";
    std::cout << "  -h, --help              Show this help\n";
    std::cout << std::endl;
}

bool parseArgs(int argc, char **argv, AppOptions &options, bool &showHelp, std::string &error) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            showHelp = true;
            return false;
        }
        if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            options.port = std::atoi(argv[++i]);
            continue;
        }
        if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            const std::string deviceArg = argv[++i];
            auto parsed = parseOutputDevice(deviceArg);
            if (!parsed.ok) {
                error = std::string("デバイス指定エラー: ") + parsed.error;
                printHelp(argv[0]);
                return false;
            }
            options.device = parsed.spec;
            continue;
        }
        if ((arg == "-l" || arg == "--log-level") && i + 1 < argc) {
            options.logLevel = parseLogLevel(argv[++i]);
            continue;
        }
        if (arg == "--ring-buffer-frames" && i + 1 < argc) {
            options.ringBufferFrames =
                static_cast<std::size_t>(std::strtoul(argv[++i], nullptr, 10));
            continue;
        }
        if (arg == "--ring-buffer-watermark" && i + 1 < argc) {
            options.watermarkFrames =
                static_cast<std::size_t>(std::strtoul(argv[++i], nullptr, 10));
            continue;
        }
        if (arg == "--no-ring-buffer") {
            options.ringBufferFrames = 0;
            continue;
        }
        if (arg == "--recv-timeout-ms" && i + 1 < argc) {
            options.recvTimeoutMs = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--recv-timeout-sleep-ms" && i + 1 < argc) {
            options.recvTimeoutSleepMs = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--accept-cooldown-ms" && i + 1 < argc) {
            options.acceptCooldownMs = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--connection-mode" && i + 1 < argc) {
            options.connectionMode = parseConnectionMode(argv[++i]);
            continue;
        }
        if (arg == "--priority-client" && i + 1 < argc) {
            std::string value = argv[++i];
            parsePriorityList(value, options.priorityClients);
            continue;
        }
        if (arg == "--disable-zmq") {
            options.enableZmq = false;
            continue;
        }
        if (arg == "--enable-zmq") {
            options.enableZmq = true;
            continue;
        }
        if (arg == "--disable-zmq-rate-notify") {
            options.enableZmqRateNotify = false;
            continue;
        }
        if (arg == "--zmq-endpoint" && i + 1 < argc) {
            options.zmqEndpoint = argv[++i];
            continue;
        }
        if (arg == "--zmq-token" && i + 1 < argc) {
            options.zmqToken = argv[++i];
            continue;
        }
        if ((arg == "--zmq-pub-interval" || arg == "--zmq-pub-interval-ms") && i + 1 < argc) {
            options.zmqPublishIntervalMs = std::atoi(argv[++i]);
            continue;
        }

        error = std::string("未知の引数: ") + arg;
        printHelp(argv[0]);
        return false;
    }
    return true;
}
