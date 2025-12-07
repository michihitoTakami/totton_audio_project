#include "alsa_playback.h"
#include "connection_mode.h"
#include "logging.h"
#include "output_device.h"
#include "pcm_stream_handler.h"
#include "status_tracker.h"
#include "tcp_server.h"
#include "zmq_status_server.h"

#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace {

std::atomic_bool *gStopFlag = nullptr;

void handleSignal(int) {
    if (gStopFlag) {
        gStopFlag->store(true, std::memory_order_relaxed);
    }
}

OutputDeviceSpec defaultDevice() {
    auto parsed = parseOutputDevice("loopback");
    return parsed.ok ? parsed.spec : OutputDeviceSpec{};
}

struct AppOptions {
    int port = 46001;
    OutputDeviceSpec device = defaultDevice();
    LogLevel logLevel = LogLevel::Info;
    std::size_t ringBufferFrames = 8192;  // 0で無効
    std::size_t watermarkFrames = 0;      // 0で自動 (75%)
    bool enableZmq = true;
    std::string zmqEndpoint = "ipc:///tmp/jetson_pcm_receiver.sock";
    std::string zmqToken;
    int zmqPublishIntervalMs = 1000;
    int recvTimeoutMs = 250;
    int recvTimeoutSleepMs = 50;
    int acceptCooldownMs = 250;
    int maxConsecutiveTimeouts = 3;
    ConnectionMode connectionMode = ConnectionMode::Single;
    std::vector<std::string> priorityClients;
};

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
    std::cout << "  --zmq-endpoint <uri>    ZeroMQ REP endpoint (default: "
                 "ipc:///tmp/jetson_pcm_receiver.sock)\n";
    std::cout << "  --zmq-token <token>     shared token for ZeroMQ commands\n";
    std::cout
        << "  --zmq-pub-interval <ms> status publish interval for PUB socket (default: 1000)\n";
    std::cout << "  -h, --help              Show this help\n";
    std::cout << std::endl;
}

bool parseArgs(int argc, char **argv, AppOptions &options, bool &showHelp) {
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
                std::cerr << "デバイス指定エラー: " << parsed.error << std::endl;
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
            std::string current;
            for (char c : value) {
                if (c == ',') {
                    if (!current.empty()) {
                        options.priorityClients.push_back(current);
                        current.clear();
                    }
                } else {
                    current.push_back(c);
                }
            }
            if (!current.empty()) {
                options.priorityClients.push_back(current);
            }
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

        std::cerr << "未知の引数: " << arg << std::endl;
        printHelp(argv[0]);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    AppOptions options;
    bool showHelp = false;
    if (!parseArgs(argc, argv, options, showHelp)) {
        return showHelp ? 0 : 1;
    }

    setLogLevel(options.logLevel);

    std::atomic_bool stopRequested{false};
    gStopFlag = &stopRequested;

    struct sigaction sa {};
    sa.sa_handler = handleSignal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    logInfo("[jetson-pcm-receiver] start");
    logInfo("  - port:   " + std::to_string(options.port));
    logInfo("  - device: " + options.device.describe());
    if (options.ringBufferFrames == 0) {
        logInfo("  - ring buffer: disabled");
    } else {
        logInfo("  - ring buffer: " + std::to_string(options.ringBufferFrames) + " frames");
        if (options.watermarkFrames > 0) {
            logInfo("  - watermark:   " + std::to_string(options.watermarkFrames) + " frames");
        }
    }
    if (options.enableZmq) {
        logInfo("  - ZeroMQ REP: " + options.zmqEndpoint);
        logInfo("  - ZeroMQ PUB: derived from REP (.pub suffix or +1 port)");
        if (!options.zmqToken.empty()) {
            logInfo("  - ZeroMQ token: <configured>");
        }
    } else {
        logInfo("  - ZeroMQ API: disabled");
    }
    logInfo("  - recv timeout: " + std::to_string(options.recvTimeoutMs) + " ms");
    logInfo("  - recv timeout sleep: " + std::to_string(options.recvTimeoutSleepMs) + " ms");
    logInfo("  - accept cooldown: " + std::to_string(options.acceptCooldownMs) + " ms");
    logInfo("  - max consecutive timeouts: " + std::to_string(options.maxConsecutiveTimeouts));
    logInfo("  - connection mode: " + toString(options.connectionMode));
    if (!options.priorityClients.empty()) {
        std::string joined;
        for (std::size_t idx = 0; idx < options.priorityClients.size(); ++idx) {
            if (idx > 0) {
                joined += ", ";
            }
            joined += options.priorityClients[idx];
        }
        logInfo("  - priority clients: " + joined);
    }

    StatusTracker status;
    status.updateRingConfig(options.ringBufferFrames, options.watermarkFrames);

    TcpServerOptions serverOptions;
    serverOptions.connectionMode = options.connectionMode;
    serverOptions.priorityClients = options.priorityClients;
    serverOptions.backlog = 8;
    TcpServer server(options.port, serverOptions);
    AlsaPlayback playback(options.device.alsaName);
    playback.setStatusTracker(&status);
    std::mutex configMutex;
    PcmStreamConfig cfg;
    cfg.ringBufferFrames = options.ringBufferFrames;
    cfg.watermarkFrames = options.watermarkFrames;
    cfg.recvTimeoutMs = options.recvTimeoutMs;
    cfg.recvTimeoutSleepMs = options.recvTimeoutSleepMs;
    cfg.acceptCooldownMs = options.acceptCooldownMs;
    cfg.maxConsecutiveTimeouts = options.maxConsecutiveTimeouts;
    cfg.connectionMode = options.connectionMode;
    cfg.priorityClients = options.priorityClients;
    PcmStreamHandler handler(playback, server, stopRequested, cfg, &configMutex, &status);

    ZmqStatusServer zmqServer(status, cfg, configMutex, stopRequested);
    if (options.enableZmq) {
        ZmqStatusServer::Options zmqOpts;
        zmqOpts.enabled = true;
        zmqOpts.endpoint = options.zmqEndpoint;
        zmqOpts.token = options.zmqToken;
        zmqOpts.publishIntervalMs = options.zmqPublishIntervalMs;
        if (!zmqServer.start(zmqOpts)) {
            logError("[jetson-pcm-receiver] failed to start ZeroMQ server");
            return 1;
        }
        if (zmqServer.running()) {
            logInfo("  - ZeroMQ REP bound to: " + zmqServer.endpoint());
            logInfo("  - ZeroMQ PUB bound to: " + zmqServer.pubEndpoint());
        }
    }

    if (!server.start()) {
        logError("[jetson-pcm-receiver] failed to start TCP server");
        zmqServer.stop();
        return 1;
    }
    handler.run();
    server.stop();
    zmqServer.stop();

    if (stopRequested.load(std::memory_order_relaxed)) {
        logInfo("[jetson-pcm-receiver] terminated by signal");
    }
    logInfo("[jetson-pcm-receiver] 終了");
    return 0;
}
