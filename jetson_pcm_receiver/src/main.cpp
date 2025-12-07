#include "alsa_playback.h"
#include "logging.h"
#include "pcm_stream_handler.h"
#include "tcp_server.h"

#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

std::atomic_bool *gStopFlag = nullptr;

void handleSignal(int) {
    if (gStopFlag) {
        gStopFlag->store(true, std::memory_order_relaxed);
    }
}

struct AppOptions {
    int port = 46001;
    std::string device = "hw:Loopback,0,0";
    LogLevel logLevel = LogLevel::Info;
    std::size_t ringBufferFrames = 8192;  // 0で無効
    std::size_t watermarkFrames = 0;      // 0で自動 (75%)
};

void printHelp(const char *exeName) {
    std::cout << "jetson-pcm-receiver\n";
    std::cout << "Usage: " << exeName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -p, --port <number>     TCP listen port (default: 46001)\n";
    std::cout << "  -d, --device <name>     ALSA playback device (default: hw:Loopback,0,0)\n";
    std::cout << "  -l, --log-level <lvl>   error/warn/info/debug (default: info)\n";
    std::cout << "  --ring-buffer-frames N  enable jitter buffer with N frames (default: 8192)\n";
    std::cout
        << "  --ring-buffer-watermark N  watermark frames for warning (default: 75% of buffer)\n";
    std::cout << "  --no-ring-buffer        disable jitter buffer\n";
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
            options.device = argv[++i];
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
    logInfo("  - device: " + options.device);
    if (options.ringBufferFrames == 0) {
        logInfo("  - ring buffer: disabled");
    } else {
        logInfo("  - ring buffer: " + std::to_string(options.ringBufferFrames) + " frames");
        if (options.watermarkFrames > 0) {
            logInfo("  - watermark:   " + std::to_string(options.watermarkFrames) + " frames");
        }
    }

    TcpServer server(options.port);
    AlsaPlayback playback(options.device);
    PcmStreamConfig cfg;
    cfg.ringBufferFrames = options.ringBufferFrames;
    cfg.watermarkFrames = options.watermarkFrames;
    PcmStreamHandler handler(playback, server, stopRequested, cfg);

    if (!server.start()) {
        logError("[jetson-pcm-receiver] failed to start TCP server");
        return 1;
    }
    handler.run();
    server.stop();

    if (stopRequested.load(std::memory_order_relaxed)) {
        logInfo("[jetson-pcm-receiver] terminated by signal");
    }
    logInfo("[jetson-pcm-receiver] 終了");
    return 0;
}
