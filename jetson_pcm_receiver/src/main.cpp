#include "alsa_playback.h"
#include "app_options.h"
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

}  // namespace

int main(int argc, char **argv) {
    AppOptions options = makeDefaultOptions();
    bool showHelp = false;
    std::string optionError;
    if (!applyEnvOverrides(options, optionError)) {
        std::cerr << optionError << std::endl;
        return 1;
    }
    if (!parseArgs(argc, argv, options, showHelp, optionError)) {
        if (!optionError.empty()) {
            std::cerr << optionError << std::endl;
        }
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
        logInfo(std::string("  - ZeroMQ rate notify: ") +
                (options.enableZmqRateNotify ? "enabled" : "disabled"));
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

    ZmqStatusServer zmqServer(status, cfg, configMutex, stopRequested);
    if (options.enableZmq) {
        ZmqStatusServer::Options zmqOpts;
        zmqOpts.enabled = true;
        zmqOpts.endpoint = options.zmqEndpoint;
        zmqOpts.token = options.zmqToken;
        zmqOpts.publishIntervalMs = options.zmqPublishIntervalMs;
        zmqOpts.publishHeaderEvents = options.enableZmqRateNotify;
        if (!zmqServer.start(zmqOpts)) {
            logError("[jetson-pcm-receiver] failed to start ZeroMQ server");
            return 1;
        }
        if (zmqServer.running()) {
            logInfo("  - ZeroMQ REP bound to: " + zmqServer.endpoint());
            logInfo("  - ZeroMQ PUB bound to: " + zmqServer.pubEndpoint());
        }
    }

    PcmStreamHandler handler(
        playback, server, stopRequested, cfg, &configMutex, &status,
        options.enableZmq && options.enableZmqRateNotify ? &zmqServer : nullptr);

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
