// Entry point for rpi_rtp_sender
#include "HwParamsMonitor.h"
#include "RtpOptions.h"
#include "RtpPipelineBuilder.h"
#include "logging.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

extern char **environ;

namespace {

std::atomic<bool> gStopRequested{false};

constexpr int MIN_LATENCY_MS = 10;
constexpr int MAX_LATENCY_MS = 500;
constexpr int DEFAULT_LATENCY_MS = 100;
constexpr char DEFAULT_STATS_PATH[] = "/var/run/rtp-bridge/rtp_receiver_stats.json";
constexpr char DEFAULT_LATENCY_PATH[] = "/var/run/rtp-bridge/rtp_receiver_latency_ms";

void handleSignal(int /*signum*/) {
    gStopRequested.store(true);
}

bool spawnProcess(const std::vector<std::string> &args, pid_t &pid) {
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto &arg : args) {
        argv.push_back(const_cast<char *>(arg.c_str()));
    }
    argv.push_back(nullptr);

    int rc = posix_spawnp(&pid, args[0].c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc != 0) {
        logError("[rpi_rtp_sender] Failed to spawn process: " + std::to_string(rc) + " (" +
                 std::strerror(rc) + ")");
        return false;
    }
    return true;
}

bool runCurlPost(const std::string &url, const std::string &body) {
    std::vector<std::string> args = {"curl", "-s", "-X", "POST", "-d", body, url};
    pid_t pid = -1;
    if (!spawnProcess(args, pid)) {
        return false;
    }
    int status = 0;
    const pid_t waited = waitpid(pid, &status, 0);
    if (waited != pid) {
        return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

void waitForExit(pid_t pid, int timeoutMs) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        pid_t ret = waitpid(pid, &status, WNOHANG);
        if (ret == pid) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // Force kill if still running
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
}

void stopProcess(pid_t pid) {
    if (pid <= 0) {
        return;
    }
    kill(pid, SIGINT);
    waitForExit(pid, 500);
    kill(pid, SIGTERM);
    waitForExit(pid, 500);
}

CaptureParams applyOverrides(const CaptureParams &params,
                             const std::optional<AlsaCapture::SampleFormat> &overrideFormat) {
    if (!overrideFormat) {
        return params;
    }
    CaptureParams adjusted = params;
    adjusted.format = *overrideFormat;
    return adjusted;
}

std::string formatToString(AlsaCapture::SampleFormat fmt) {
    switch (fmt) {
    case AlsaCapture::SampleFormat::S16_LE:
        return "S16_LE";
    case AlsaCapture::SampleFormat::S24_3LE:
        return "S24_3LE";
    case AlsaCapture::SampleFormat::S32_LE:
        return "S32_LE";
    }
    return "S24_3LE";
}

int clampLatencyMs(int value) {
    if (value < MIN_LATENCY_MS) {
        return MIN_LATENCY_MS;
    }
    if (value > MAX_LATENCY_MS) {
        return MAX_LATENCY_MS;
    }
    return value;
}

std::optional<int> loadLatencyMs(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return std::nullopt;
    }
    int value = 0;
    ifs >> value;
    if (!ifs.good()) {
        return std::nullopt;
    }
    return clampLatencyMs(value);
}

void writeStatsFile(const std::string &path, const CaptureParams &params, int latencyMs) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
        return;
    }
    ofs << "{";
    ofs << R"("packets_received":0,)";
    ofs << R"("packets_lost":0,)";
    ofs << R"("jitter_ms":0.0,)";
    ofs << R"("clock_drift_ppm":0.0,)";
    ofs << R"("sample_rate":)" << params.sampleRate << ",";
    ofs << R"("latency_ms":)" << latencyMs;
    ofs << "}";
}

void notifyRateChange(const RtpOptions &opt, const CaptureParams &params) {
    if (opt.rateNotifyUrl.empty()) {
        return;
    }
    const std::string body = "rate=" + std::to_string(params.sampleRate) +
                             "&channels=" + std::to_string(params.channels) +
                             "&format=" + formatToString(params.format);
    if (!runCurlPost(opt.rateNotifyUrl, body)) {
        logWarn("[rpi_rtp_sender] Rate notify failed");
    }
}

}  // namespace

int main(int argc, char **argv) {
    const auto parsed = parseRtpOptions(argc, argv, "rpi_rtp_sender");
    if (parsed.showHelp || parsed.showVersion) {
        return 0;
    }
    if (parsed.hasError || !parsed.options) {
        logError(parsed.errorMessage);
        return 1;
    }
    const RtpOptions opt = *parsed.options;
    setLogLevel(opt.logLevel);

    struct sigaction sa {};
    sa.sa_handler = handleSignal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    HwParamsMonitor monitor(opt.device);
    auto currentParams = monitor.readCurrent();
    if (!currentParams) {
        logError("[rpi_rtp_sender] Failed to read ALSA hw_params from " + monitor.describe());
        return 1;
    }
    *currentParams = applyOverrides(*currentParams, opt.formatOverride);

    RtpPipelineConfig cfg;
    cfg.host = opt.host;
    cfg.rtpPort = opt.rtpPort;
    cfg.rtcpSendPort = opt.rtcpSendPort;
    cfg.rtcpListenPort = opt.rtcpListenPort;
    cfg.payloadType = opt.payloadType;
    cfg.device = opt.device;

    const std::string statsPath = std::getenv("RTP_BRIDGE_STATS_PATH")
                                      ? std::getenv("RTP_BRIDGE_STATS_PATH")
                                      : DEFAULT_STATS_PATH;
    const std::string latencyPath = std::getenv("RTP_BRIDGE_LATENCY_PATH")
                                        ? std::getenv("RTP_BRIDGE_LATENCY_PATH")
                                        : DEFAULT_LATENCY_PATH;

    int latencyMs = DEFAULT_LATENCY_MS;
    if (auto lat = loadLatencyMs(latencyPath)) {
        latencyMs = *lat;
    }
    cfg.latencyMs = latencyMs;

    auto pipelineArgs = RtpPipelineBuilder::build(cfg, *currentParams);
    const std::string pipelineStr = RtpPipelineBuilder::toCommandString(pipelineArgs);
    logInfo("[rpi_rtp_sender] Starting pipeline: " + pipelineStr);

    if (opt.dryRun) {
        return 0;
    }

    pid_t pid = -1;
    if (!spawnProcess(pipelineArgs, pid)) {
        return 1;
    }

    notifyRateChange(opt, *currentParams);
    writeStatsFile(statsPath, *currentParams, latencyMs);

    const std::chrono::milliseconds pollInterval{opt.pollIntervalMs};
    while (!gStopRequested.load()) {
        int status = 0;
        if (pid > 0) {
            const pid_t ret = waitpid(pid, &status, WNOHANG);
            if (ret == pid) {
                logWarn("[rpi_rtp_sender] Pipeline process exited; restarting");
                if (!spawnProcess(pipelineArgs, pid)) {
                    logError("[rpi_rtp_sender] Failed to restart after exit");
                    return 1;
                }
                notifyRateChange(opt, *currentParams);
            }
        }

        std::this_thread::sleep_for(pollInterval);
        auto params = monitor.readCurrent();
        if (!params) {
            logWarn("[rpi_rtp_sender] hw_params unavailable at " + monitor.describe());
            continue;
        }
        *params = applyOverrides(*params, opt.formatOverride);
        if (*params != *currentParams) {
            logInfo("[rpi_rtp_sender] Detected rate/format change -> restarting pipeline");
            stopProcess(pid);
            pipelineArgs = RtpPipelineBuilder::build(cfg, *params);
            const std::string cmdString = RtpPipelineBuilder::toCommandString(pipelineArgs);
            logInfo("[rpi_rtp_sender] Rebuilt pipeline: " + cmdString);
            if (!spawnProcess(pipelineArgs, pid)) {
                logError("[rpi_rtp_sender] Failed to restart pipeline");
                return 1;
            }
            notifyRateChange(opt, *params);
            writeStatsFile(statsPath, *params, latencyMs);
            *currentParams = *params;
        }

        // Latency file watcher
        if (auto lat = loadLatencyMs(latencyPath)) {
            if (*lat != latencyMs) {
                latencyMs = *lat;
                logInfo("[rpi_rtp_sender] Applying latency change to " + std::to_string(latencyMs) +
                        " ms");
                stopProcess(pid);
                cfg.latencyMs = latencyMs;
                pipelineArgs = RtpPipelineBuilder::build(cfg, *currentParams);
                const std::string cmdString = RtpPipelineBuilder::toCommandString(pipelineArgs);
                logInfo("[rpi_rtp_sender] Rebuilt pipeline with latency: " + cmdString);
                if (!spawnProcess(pipelineArgs, pid)) {
                    logError("[rpi_rtp_sender] Failed to restart after latency change");
                    return 1;
                }
                writeStatsFile(statsPath, *currentParams, latencyMs);
            }
        }
    }

    logInfo("[rpi_rtp_sender] Signal received, stopping pipeline");
    stopProcess(pid);
    return 0;
}
