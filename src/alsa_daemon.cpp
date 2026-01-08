// IMPORTANT: DO NOT GROW THIS FILE AGAIN.
//
// Keep `src/alsa_daemon.cpp` as a small entrypoint + wiring only.
// Any non-trivial logic (ALSA/input threads/rate switching/filter switching/stats/etc) MUST live in
// dedicated modules under `src/daemon/**` (with headers in `include/daemon/**`).

#include "audio/audio_utils.h"
#include "audio/fallback_manager.h"
#include "audio/soft_mute.h"
#include "convolution_engine.h"
#include "core/config_loader.h"
#include "core/daemon_constants.h"
#include "core/partition_runtime_utils.h"
#include "daemon/api/dependencies.h"
#include "daemon/api/events.h"
#include "daemon/app/process_resources.h"
#include "daemon/app/runtime_state.h"
#include "daemon/audio/crossfeed_manager.h"
#include "daemon/audio/upsampler_builder.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/audio_pipeline/filter_manager.h"
#include "daemon/audio_pipeline/headroom_controller.h"
#include "daemon/audio_pipeline/rate_switcher.h"
#include "daemon/audio_pipeline/soft_mute_runner.h"
#include "daemon/audio_pipeline/stream_buffer_sizing.h"
#include "daemon/audio_pipeline/streaming_cache_manager.h"
#include "daemon/audio_pipeline/switch_actions.h"
#include "daemon/control/control_plane.h"
#include "daemon/control/handlers/handler_registry.h"
#include "daemon/core/thread_priority.h"
#include "daemon/input/i2s_capture.h"
#include "daemon/input/loopback_capture.h"
#include "daemon/metrics/runtime_stats.h"
#include "daemon/output/alsa_output.h"
#include "daemon/output/alsa_output_thread.h"
#include "daemon/output/alsa_pcm_controller.h"
#include "daemon/output/alsa_write_loop.h"
#include "daemon/output/playback_buffer_access.h"
#include "daemon/output/playback_buffer_manager.h"
#include "daemon/pcm/dac_manager.h"
#include "daemon/shutdown_manager.h"
#include "io/dac_capability.h"
#include "io/playback_buffer.h"
#include "logging/logger.h"
#include "logging/metrics.h"

#include <algorithm>
#include <alsa/asoundlib.h>
#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <net/if.h>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// systemd notification support (optional)
#ifdef HAVE_SYSTEMD
#include <systemd/sd-daemon.h>
#endif

// NOTE: Avoid easy bloat; keep components modular whenever possible.

// PID file path (also serves as lock file)
constexpr const char* PID_FILE_PATH = "/tmp/gpu_upsampler_alsa.pid";

static void enforce_phase_partition_constraints(AppConfig& config) {
    if (config.partitionedConvolution.enabled && config.phaseType == PhaseType::Linear) {
        std::cout << "[Partition] Linear phase is incompatible with low-latency mode. "
                  << "Switching to minimum phase." << '\n';
        config.phaseType = PhaseType::Minimum;
    }
}

// Stats file path (JSON format for Web API)
constexpr const char* STATS_FILE_PATH = "/tmp/gpu_upsampler_stats.json";

static runtime_stats::Dependencies build_runtime_stats_dependencies();

// Default configuration values (using common constants)
using namespace DaemonConstants;
using StreamFloatVector = ConvolutionEngine::StreamFloatVector;
constexpr const char* DEFAULT_ALSA_DEVICE = "hw:USB";
constexpr const char* DEFAULT_LOOPBACK_DEVICE = "hw:Loopback,1,0";
constexpr uint32_t DEFAULT_LOOPBACK_PERIOD_FRAMES = 1024;
constexpr const char* DEFAULT_FILTER_PATH = "data/coefficients/filter_44k_16x_640k_min_phase.bin";
constexpr const char* CONFIG_FILE_PATH = DEFAULT_CONFIG_FILE;

static constexpr std::array<const char*, 1> kSupportedOutputModes = {"usb"};

static std::string normalize_output_mode(const std::string& value) {
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return normalized;
}

static bool is_supported_output_mode(const std::string& normalized) {
    for (const auto* mode : kSupportedOutputModes) {
        if (normalized == mode) {
            return true;
        }
    }
    return false;
}

static void set_preferred_output_device(AppConfig& config, const std::string& device) {
    std::string resolved = device;
    if (resolved.empty()) {
        resolved = DEFAULT_ALSA_DEVICE;
    }
    config.output.mode = OutputMode::Usb;
    config.output.usb.preferredDevice = resolved;
    config.alsaDevice = resolved;
}

static void ensure_output_config(AppConfig& config) {
    std::string current = outputModeToString(config.output.mode);
    std::string normalized = normalize_output_mode(current);
    if (!is_supported_output_mode(normalized)) {
        std::cout << "Config: Unsupported output mode '" << current << "', forcing 'usb'" << '\n';
        config.output.mode = OutputMode::Usb;
        normalized = "usb";
    }

    if (config.output.usb.preferredDevice.empty()) {
        if (!config.alsaDevice.empty()) {
            config.output.usb.preferredDevice = config.alsaDevice;
        } else {
            config.output.usb.preferredDevice = DEFAULT_ALSA_DEVICE;
        }
    }

    if (config.alsaDevice.empty()) {
        config.alsaDevice = config.output.usb.preferredDevice;
    }
}

// Runtime configuration (loaded from config.json)
static daemon_app::RuntimeState g_state;

static audio_pipeline::HeadroomControllerDependencies make_headroom_dependencies(
    daemon_core::api::EventDispatcher* dispatcher) {
    audio_pipeline::HeadroomControllerDependencies deps{};
    deps.dispatcher = dispatcher;
    deps.config = &g_state.config;
    deps.headroomCache = &g_state.headroomCache;
    deps.headroomGain = &g_state.gains.headroom;
    deps.outputGain = &g_state.gains.output;
    deps.effectiveGain = &g_state.gains.effective;
    deps.activeRateFamily = []() { return g_state.rates.activeRateFamily; };
    deps.activePhaseType = []() { return g_state.rates.activePhaseType; };
    return deps;
}

static void update_daemon_dependencies() {
    g_state.managers.daemonDependencies.config = &g_state.config;
    g_state.managers.daemonDependencies.running = &g_state.flags.running;
    g_state.managers.daemonDependencies.outputReady = &g_state.flags.outputReady;
    g_state.managers.daemonDependencies.crossfeedEnabled = &g_state.crossfeed.enabled;
    g_state.managers.daemonDependencies.currentInputRate = &g_state.rates.currentInputRate;
    g_state.managers.daemonDependencies.currentOutputRate = &g_state.rates.currentOutputRate;
    g_state.managers.daemonDependencies.softMute = &g_state.softMute.controller;
    g_state.managers.daemonDependencies.upsampler = &g_state.upsampler;
    g_state.managers.daemonDependencies.audioPipeline = &g_state.managers.audioPipelineRaw;
    g_state.managers.daemonDependencies.dacManager = &g_state.managers.dacManagerRaw;
    g_state.managers.daemonDependencies.streamingMutex = &g_state.streaming.streamingMutex;
    g_state.managers.daemonDependencies.refreshHeadroom = [](const std::string& reason) {
        if (g_state.managers.headroomController) {
            g_state.managers.headroomController->refreshCurrentHeadroom(reason);
        }
    };
}

static void initialize_event_modules() {
    g_state.managers.eventDispatcher = std::make_unique<daemon_core::api::EventDispatcher>();
    update_daemon_dependencies();

    g_state.managers.rateSwitcher =
        std::make_unique<audio_pipeline::RateSwitcher>(audio_pipeline::RateSwitcherDependencies{
            .dispatcher = g_state.managers.eventDispatcher.get(),
            .deps = g_state.managers.daemonDependencies,
            .pendingRate = &g_state.rates.pendingRateChange});
    g_state.managers.filterManager =
        std::make_unique<audio_pipeline::FilterManager>(audio_pipeline::FilterManagerDependencies{
            .dispatcher = g_state.managers.eventDispatcher.get(),
            .deps = g_state.managers.daemonDependencies});
    g_state.managers.headroomController = std::make_unique<audio_pipeline::HeadroomController>(
        make_headroom_dependencies(g_state.managers.eventDispatcher.get()));
    g_state.managers.softMuteRunner =
        std::make_unique<audio_pipeline::SoftMuteRunner>(audio_pipeline::SoftMuteRunnerDependencies{
            .dispatcher = g_state.managers.eventDispatcher.get(),
            .deps = g_state.managers.daemonDependencies});
    g_state.managers.alsaOutputInterface = std::make_unique<daemon_output::AlsaOutput>(
        daemon_output::AlsaOutputDependencies{.dispatcher = g_state.managers.eventDispatcher.get(),
                                              .deps = g_state.managers.daemonDependencies});
    g_state.managers.handlerRegistry = std::make_unique<daemon_control::handlers::HandlerRegistry>(
        daemon_control::handlers::HandlerRegistryDependencies{
            .dispatcher = g_state.managers.eventDispatcher.get()});

    g_state.managers.rateSwitcher->start();
    g_state.managers.filterManager->start();
    g_state.managers.headroomController->start();
    g_state.managers.softMuteRunner->start();
    g_state.managers.alsaOutputInterface->start();
    g_state.managers.handlerRegistry->registerDefaults();
}

static void publish_rate_change_event(int detected_rate) {
    if (!g_state.managers.rateSwitcher || !g_state.managers.eventDispatcher) {
        return;
    }
    daemon_core::api::RateChangeRequested event;
    event.detectedInputRate = detected_rate;
    event.rateFamily = ConvolutionEngine::detectRateFamily(detected_rate);
    g_state.managers.eventDispatcher->publish(event);
}

static void publish_filter_switch_event(const std::string& filterPath, PhaseType phaseType,
                                        bool reloadHeadroom) {
    daemon_core::api::FilterSwitchRequested event;
    event.filterPath = filterPath;
    event.phaseType = phaseType;
    event.reloadHeadroom = reloadHeadroom;
    if (g_state.managers.eventDispatcher) {
        g_state.managers.eventDispatcher->publish(event);
    }
}

static void initialize_streaming_cache_manager();

static void initialize_streaming_cache_manager() {
    streaming_cache::StreamingCacheDependencies deps;
    deps.flushAction = [](std::chrono::nanoseconds /*gap*/) -> bool {
        // DAC 切断/入力ギャップ後の復帰で「古いオーバーラップ状態」を引きずらないよう、
        // 制御系スレッドで RT を静止させた上でキャッシュをフラッシュする。
        if (g_state.audioPipeline) {
            g_state.audioPipeline->requestRtPause();
            if (!g_state.audioPipeline->waitForRtQuiescent(std::chrono::milliseconds(500))) {
                g_state.audioPipeline->resumeRtPause();
                return false;
            }
        }

        struct RtPauseReleaseGuard {
            audio_pipeline::AudioPipeline* pipeline = nullptr;
            explicit RtPauseReleaseGuard(audio_pipeline::AudioPipeline* p) : pipeline(p) {}
            ~RtPauseReleaseGuard() {
                if (pipeline) {
                    pipeline->resumeRtPause();
                }
            }
        } pauseRelease(g_state.audioPipeline.get());

        if (g_state.softMute.controller) {
            g_state.softMute.controller->startFadeOut();
        }

        daemon_output::playbackBuffer(g_state).reset();

        {
            std::lock_guard<std::mutex> streamLock(g_state.streaming.streamingMutex);
            if (!g_state.streaming.streamInputLeft.empty()) {
                std::fill(g_state.streaming.streamInputLeft.begin(),
                          g_state.streaming.streamInputLeft.end(), 0.0f);
            }
            if (!g_state.streaming.streamInputRight.empty()) {
                std::fill(g_state.streaming.streamInputRight.begin(),
                          g_state.streaming.streamInputRight.end(), 0.0f);
            }
            g_state.streaming.streamAccumulatedLeft = 0;
            g_state.streaming.streamAccumulatedRight = 0;
            g_state.streaming.upsamplerOutputLeft.clear();
            g_state.streaming.upsamplerOutputRight.clear();
        }

        if (g_state.upsampler) {
            g_state.upsampler->resetStreaming();
        }

        g_state.crossfeed.resetRequested.store(true, std::memory_order_release);

        if (g_state.softMute.controller) {
            g_state.softMute.controller->startFadeIn();
        }
        return true;
    };
    g_state.managers.streamingCacheManager =
        std::make_unique<streaming_cache::StreamingCacheManager>(deps);
}

// ========== DAC Device Monitoring ==========

static inline int64_t get_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static dac::DacManager::Dependencies make_dac_dependencies(
    std::function<void(const nlohmann::json&)> eventPublisher = {}) {
    dac::DacManager::Dependencies deps;
    deps.config = &g_state.config;
    deps.runningFlag = &g_state.flags.running;
    deps.timestampProvider = get_timestamp_ms;
    deps.eventPublisher = std::move(eventPublisher);
    deps.defaultDevice = DEFAULT_ALSA_DEVICE;
    return deps;
}

static runtime_stats::Dependencies build_runtime_stats_dependencies() {
    runtime_stats::Dependencies deps;
    deps.config = &g_state.config;
    deps.upsampler = g_state.upsampler;
    deps.headroomCache = &g_state.headroomCache;
    deps.dacManager = g_state.managers.dacManager.get();
    deps.fallbackManager = g_state.fallbackManager;
    deps.fallbackActive = &g_state.fallbackActive;
    deps.inputSampleRate = &g_state.rates.inputSampleRate;
    deps.headroomGain = &g_state.gains.headroom;
    deps.outputGain = &g_state.gains.output;
    deps.limiterGain = &g_state.gains.limiter;
    deps.effectiveGain = &g_state.gains.effective;
    deps.delimiterMode = &g_state.delimiter.mode;
    deps.delimiterFallbackReason = &g_state.delimiter.fallbackReason;
    deps.delimiterBypassLocked = &g_state.delimiter.bypassLocked;
    deps.delimiterEnabled = &g_state.delimiter.enabled;
    deps.delimiterWarmup = &g_state.delimiter.warmup;
    deps.delimiterQueueSamples = &g_state.delimiter.queueSamples;
    deps.delimiterQueueSeconds = &g_state.delimiter.queueSeconds;
    deps.delimiterLastInferenceMs = &g_state.delimiter.lastInferenceMs;
    deps.delimiterBackendAvailable = &g_state.delimiter.backendAvailable;
    deps.delimiterBackendValid = &g_state.delimiter.backendValid;
    deps.delimiterTargetMode = &g_state.delimiter.targetMode;
    deps.playbackBuffer = &daemon_output::playbackBuffer(g_state);
    deps.outputReady = &g_state.flags.outputReady;

    return deps;
}

// ========== Configuration ==========

static void print_config_summary(const AppConfig& cfg) {
    int outputRate = g_state.rates.inputSampleRate * cfg.upsampleRatio;
    LOG_INFO("Config:");
    LOG_INFO("  ALSA device:    {}", cfg.alsaDevice);
    LOG_INFO("  Output mode:    {} (preferred USB device: {})", outputModeToString(cfg.output.mode),
             cfg.output.usb.preferredDevice);
    LOG_INFO("  I2S input:      {} device={} rate={} ch={} fmt={} period={}",
             (cfg.i2s.enabled ? "enabled" : "disabled"), cfg.i2s.device, cfg.i2s.sampleRate,
             cfg.i2s.channels, cfg.i2s.format, cfg.i2s.periodFrames);
    LOG_INFO("  Loopback:       {} device={} rate={} ch={} fmt={} period={}",
             (cfg.loopback.enabled ? "enabled" : "disabled"), cfg.loopback.device,
             cfg.loopback.sampleRate, cfg.loopback.channels, cfg.loopback.format,
             cfg.loopback.periodFrames);
    LOG_INFO("  Input rate:     {} Hz (auto-negotiated)", g_state.rates.inputSampleRate);
    LOG_INFO("  Output rate:    {} Hz ({:.1f} kHz)", outputRate, outputRate / 1000.0);
    LOG_INFO("  Buffer size:    {}", cfg.bufferSize);
    LOG_INFO("  Period size:    {}", cfg.periodSize);
    LOG_INFO("  Upsample ratio: {}", cfg.upsampleRatio);
    LOG_INFO("  Block size:     {}", cfg.blockSize);
    LOG_INFO("  Gain:           {}", cfg.gain);
    LOG_INFO("  Headroom tgt:   {}", cfg.headroomTarget);
    LOG_INFO("  Headroom mode:  {}", headroomModeToString(cfg.headroomMode));
    LOG_INFO("  Base filter path: {}", cfg.filterPath);
    LOG_INFO("  Filter path 44k min: {}", cfg.filterPath44kMin);
    LOG_INFO("  Filter path 48k min: {}", cfg.filterPath48kMin);
    LOG_INFO("  Filter path 44k linear: {}", cfg.filterPath44kLinear);
    LOG_INFO("  Filter path 48k linear: {}", cfg.filterPath48kLinear);
    LOG_INFO("  EQ enabled:     {}", (cfg.eqEnabled ? "yes" : "no"));
    if (cfg.eqEnabled && !cfg.eqProfilePath.empty()) {
        LOG_INFO("  EQ profile:     {}", cfg.eqProfilePath);
    }
    // Update metrics with audio configuration
    gpu_upsampler::metrics::setAudioConfig(g_state.rates.inputSampleRate, outputRate,
                                           cfg.upsampleRatio);
}

static void reset_runtime_state() {
    g_state.managers.streamingCacheManager.reset();

    daemon_output::playbackBuffer(g_state).reset();
    g_state.streaming.streamInputLeft.clear();
    g_state.streaming.streamInputRight.clear();
    g_state.streaming.streamAccumulatedLeft = 0;
    g_state.streaming.streamAccumulatedRight = 0;
    g_state.streaming.upsamplerOutputLeft.clear();
    g_state.streaming.upsamplerOutputRight.clear();

    // Reset crossfeed streaming buffers
    daemon_audio::clearCrossfeedRuntimeBuffers(g_state.crossfeed);

    g_state.delimiter.mode.store(static_cast<int>(delimiter::ProcessingMode::Active),
                                 std::memory_order_relaxed);
    g_state.delimiter.targetMode.store(static_cast<int>(delimiter::ProcessingMode::Active),
                                       std::memory_order_relaxed);
    g_state.delimiter.fallbackReason.store(static_cast<int>(delimiter::FallbackReason::None),
                                           std::memory_order_relaxed);
    g_state.delimiter.bypassLocked.store(false, std::memory_order_relaxed);
    g_state.delimiter.enabled.store(false, std::memory_order_relaxed);
    g_state.delimiter.warmup.store(false, std::memory_order_relaxed);
    g_state.delimiter.queueSamples.store(0, std::memory_order_relaxed);
    g_state.delimiter.queueSeconds.store(0.0, std::memory_order_relaxed);
    g_state.delimiter.lastInferenceMs.store(0.0, std::memory_order_relaxed);
    g_state.delimiter.backendAvailable.store(false, std::memory_order_relaxed);
    g_state.delimiter.backendValid.store(false, std::memory_order_relaxed);
}

static void load_runtime_config() {
    AppConfig loaded;
    bool found = loadAppConfig(CONFIG_FILE_PATH, loaded);
    g_state.config = loaded;
    ensure_output_config(g_state.config);

    if (g_state.config.alsaDevice.empty()) {
        set_preferred_output_device(g_state.config, DEFAULT_ALSA_DEVICE);
    }
    if (g_state.config.filterPath.empty()) {
        g_state.config.filterPath = DEFAULT_FILTER_PATH;
    }
    if (g_state.config.upsampleRatio <= 0) {
        g_state.config.upsampleRatio = DEFAULT_UPSAMPLE_RATIO;
    }
    if (g_state.config.blockSize <= 0) {
        g_state.config.blockSize = DEFAULT_BLOCK_SIZE;
    }
    if (g_state.config.bufferSize <= 0) {
        g_state.config.bufferSize = 262144;
    }
    if (g_state.config.periodSize <= 0) {
        g_state.config.periodSize = 32768;
    }
    if (g_state.config.loopback.device.empty()) {
        g_state.config.loopback.device = DEFAULT_LOOPBACK_DEVICE;
    }
    if (g_state.config.loopback.sampleRate == 0) {
        g_state.config.loopback.sampleRate = DEFAULT_INPUT_SAMPLE_RATE;
    }
    if (g_state.config.loopback.channels == 0) {
        g_state.config.loopback.channels = CHANNELS;
    }
    if (g_state.config.loopback.periodFrames == 0) {
        g_state.config.loopback.periodFrames = DEFAULT_LOOPBACK_PERIOD_FRAMES;
    }
    if (g_state.config.loopback.format.empty()) {
        g_state.config.loopback.format = "S16_LE";
    }
    if (g_state.config.loopback.enabled) {
        g_state.rates.inputSampleRate = static_cast<int>(g_state.config.loopback.sampleRate);
    }
    if (g_state.config.i2s.enabled) {
        // MVP: i2s.sampleRate can be 0 (= follow runtime/negotiated rate).
        // If explicitly specified, adopt it as the engine input sample rate.
        if (g_state.config.i2s.sampleRate != 0) {
            g_state.rates.inputSampleRate = static_cast<int>(g_state.config.i2s.sampleRate);
        }
    }
    if (g_state.rates.inputSampleRate != 44100 && g_state.rates.inputSampleRate != 48000) {
        g_state.rates.inputSampleRate = DEFAULT_INPUT_SAMPLE_RATE;
    }

    // De-limiterは起動時に必ずOFFから始める（手動で明示ONするまで有効化しない）
    g_state.config.delimiter.enabled = false;

    g_state.delimiter.enabled.store(g_state.config.delimiter.enabled, std::memory_order_relaxed);
    g_state.delimiter.warmup.store(g_state.config.delimiter.enabled, std::memory_order_relaxed);
    g_state.delimiter.backendAvailable.store(g_state.config.delimiter.enabled,
                                             std::memory_order_relaxed);
    g_state.delimiter.backendValid.store(false, std::memory_order_relaxed);
    g_state.delimiter.queueSamples.store(0, std::memory_order_relaxed);
    g_state.delimiter.queueSeconds.store(0.0, std::memory_order_relaxed);
    g_state.delimiter.lastInferenceMs.store(0.0, std::memory_order_relaxed);
    g_state.delimiter.targetMode.store(static_cast<int>(delimiter::ProcessingMode::Active),
                                       std::memory_order_relaxed);
    g_state.delimiter.fallbackReason.store(static_cast<int>(delimiter::FallbackReason::None),
                                           std::memory_order_relaxed);
    g_state.delimiter.bypassLocked.store(false, std::memory_order_relaxed);

    if (!found) {
        std::cout << "Config: Using defaults (no config.json found)" << '\n';
    }

    enforce_phase_partition_constraints(g_state.config);

    print_config_summary(g_state.config);
    if (!g_state.managers.headroomController) {
        g_state.managers.headroomController = std::make_unique<audio_pipeline::HeadroomController>(
            make_headroom_dependencies(nullptr));
    }
    if (g_state.managers.headroomController) {
        g_state.managers.headroomController->setMode(g_state.config.headroomMode);
        if (g_state.config.headroomMode == HeadroomMode::FamilyMax) {
            std::vector<std::string> preloadPaths = {
                g_state.config.filterPath44kMin, g_state.config.filterPath48kMin,
                g_state.config.filterPath44kLinear, g_state.config.filterPath48kLinear};
            g_state.headroomCache.preloadDirectory(g_state.config.coefficientDir);
            g_state.headroomCache.preload(preloadPaths);
        }
        g_state.managers.headroomController->setTargetPeak(g_state.config.headroomTarget);
        g_state.managers.headroomController->resetEffectiveGain(
            "config load (pending filter headroom)");
    }
    float initialOutput = g_state.gains.output.load(std::memory_order_relaxed);
    g_state.gains.limiter.store(1.0f, std::memory_order_relaxed);
    g_state.gains.effective.store(initialOutput, std::memory_order_relaxed);
}

// ALSA output thread (705.6kHz direct to DAC)
int main(int argc, char* argv[]) {
    // Early initialization with stderr output only (before PID lock)
    // This allows logging during PID lock acquisition
    gpu_upsampler::logging::initializeEarly();

    daemon_app::ProcessResources::Options resourceOptions;
    resourceOptions.pidFilePath = PID_FILE_PATH;
    resourceOptions.statsFilePath = STATS_FILE_PATH;
    auto processResources = daemon_app::ProcessResources::acquire(resourceOptions);
    if (!processResources) {
        return 1;
    }

    // Full initialization from config file (after PID lock acquired)
    // This replaces stderr-only logger with configured logger (file + console)
    gpu_upsampler::logging::initializeFromConfig(CONFIG_FILE_PATH);

    LOG_INFO("========================================");
    LOG_INFO("  GPU Audio Upsampler - ALSA Direct Output");
    LOG_INFO("========================================");
    LOG_INFO("PID: {} (file: {})", getpid(), processResources->pidLock().path());

    shutdown_manager::ShutdownManager::Dependencies shutdownDeps{
        &g_state.softMute.controller, &g_state.flags.running, &g_state.flags.reloadRequested,
        nullptr};
    shutdown_manager::ShutdownManager shutdownManager(shutdownDeps);
    shutdownManager.installSignalHandlers();

    int exitCode = 0;

    do {
        shutdownManager.reset();
        g_state.flags.running = true;
        g_state.flags.reloadRequested = false;
        g_state.flags.zmqBindFailed.store(false);
        reset_runtime_state();

        // Load configuration from config.json (if exists)
        load_runtime_config();

        // Environment variable overrides config.json
        if (const char* env_dev = std::getenv("ALSA_DEVICE")) {
            set_preferred_output_device(g_state.config, env_dev);
            std::cout << "Config: ALSA_DEVICE env override: " << env_dev << '\n';
        }

        // Command line argument overrides filter path
        if (argc > 1) {
            g_state.config.filterPath = argv[1];
            std::cout << "Config: CLI filter path override: " << argv[1] << '\n';
        }

        initialize_event_modules();

        PartitionRuntime::RuntimeRequest partitionRequest{
            g_state.config.partitionedConvolution.enabled, g_state.config.eqEnabled,
            g_state.config.crossfeed.enabled};

        auto buildResult = daemon_audio::buildUpsampler(
            g_state.config, g_state.rates.inputSampleRate, partitionRequest, g_state.flags.running);
        if (buildResult.status == daemon_audio::UpsamplerBuildStatus::Failure) {
            exitCode = 1;
            break;
        }
        if (buildResult.status == daemon_audio::UpsamplerBuildStatus::Interrupted) {
            break;
        }

        g_state.upsampler = buildResult.upsampler.release();
        ConvolutionEngine::RateFamily activeFamily = buildResult.initialRateFamily;

        if (g_state.config.multiRateEnabled) {
            g_state.rates.currentInputRate.store(buildResult.currentInputRate,
                                                 std::memory_order_release);
            g_state.rates.currentOutputRate.store(buildResult.currentOutputRate,
                                                  std::memory_order_release);

            activeFamily = ConvolutionEngine::detectRateFamily(g_state.rates.inputSampleRate);
            if (activeFamily == ConvolutionEngine::RateFamily::RATE_UNKNOWN) {
                activeFamily = ConvolutionEngine::RateFamily::RATE_44K;
            }
            g_state.rates.currentRateFamilyInt.store(static_cast<int>(activeFamily),
                                                     std::memory_order_release);
        }

        g_state.rates.activeRateFamily = activeFamily;
        g_state.rates.activePhaseType = g_state.config.phaseType;
        std::string filterPath;
        if (g_state.managers.headroomController) {
            filterPath = g_state.managers.headroomController->currentFilterPath();
        }
        publish_filter_switch_event(filterPath, g_state.rates.activePhaseType, true);

        // Pre-allocate streaming input buffers (avoid RT reallocations)
        size_t buffer_capacity = audio_pipeline::computeStreamBufferCapacity(
            g_state, g_state.upsampler->getStreamValidInputPerBlock());
        g_state.streaming.streamInputLeft.resize(buffer_capacity, 0.0f);
        g_state.streaming.streamInputRight.resize(buffer_capacity, 0.0f);
        std::cout << "Streaming buffer capacity: " << buffer_capacity
                  << " samples (pre-sized for RT path)" << '\n';
        g_state.streaming.streamAccumulatedLeft = 0;
        g_state.streaming.streamAccumulatedRight = 0;
        size_t upsampler_output_capacity =
            buffer_capacity * static_cast<size_t>(g_state.config.upsampleRatio);
        g_state.streaming.upsamplerOutputLeft.reserve(upsampler_output_capacity);
        g_state.streaming.upsamplerOutputRight.reserve(upsampler_output_capacity);
        initialize_streaming_cache_manager();

        (void)daemon_audio::initializeCrossfeed(g_state,
                                                g_state.config.partitionedConvolution.enabled);

        std::cout << '\n';

        if (!g_state.audioPipeline && g_state.upsampler) {
            audio_pipeline::Dependencies pipelineDeps{};
            pipelineDeps.config = &g_state.config;
            pipelineDeps.upsampler.available = true;
            if (auto* cudaUpsampler =
                    dynamic_cast<ConvolutionEngine::GPUUpsampler*>(g_state.upsampler)) {
                pipelineDeps.upsampler.streamLeft = cudaUpsampler->streamLeft_;
                pipelineDeps.upsampler.streamRight = cudaUpsampler->streamRight_;
            } else {
                pipelineDeps.upsampler.streamLeft = nullptr;
                pipelineDeps.upsampler.streamRight = nullptr;
            }
            pipelineDeps.output.outputGain = &g_state.gains.output;
            pipelineDeps.output.limiterGain = &g_state.gains.limiter;
            pipelineDeps.output.effectiveGain = &g_state.gains.effective;
            pipelineDeps.running = &g_state.flags.running;
            pipelineDeps.upsampler.process =
                [](const float* data, size_t frames, ConvolutionEngine::StreamFloatVector& output,
                   cudaStream_t stream, ConvolutionEngine::StreamFloatVector& streamInput,
                   size_t& streamAccumulated) {
                    if (!g_state.upsampler) {
                        return false;
                    }
                    return g_state.upsampler->processStreamBlock(data, frames, output, stream,
                                                                 streamInput, streamAccumulated);
                };
            pipelineDeps.fallbackActive = &g_state.fallbackActive;
            pipelineDeps.outputReady = &g_state.flags.outputReady;
            pipelineDeps.streamingCacheManager = g_state.managers.streamingCacheManager.get();
            pipelineDeps.streamInputLeft = &g_state.streaming.streamInputLeft;
            pipelineDeps.streamInputRight = &g_state.streaming.streamInputRight;
            pipelineDeps.streamAccumulatedLeft = &g_state.streaming.streamAccumulatedLeft;
            pipelineDeps.streamAccumulatedRight = &g_state.streaming.streamAccumulatedRight;
            pipelineDeps.upsamplerOutputLeft = &g_state.streaming.upsamplerOutputLeft;
            pipelineDeps.upsamplerOutputRight = &g_state.streaming.upsamplerOutputRight;
            pipelineDeps.cfStreamInputLeft = &g_state.crossfeed.cfStreamInputLeft;
            pipelineDeps.cfStreamInputRight = &g_state.crossfeed.cfStreamInputRight;
            pipelineDeps.cfStreamAccumulatedLeft = &g_state.crossfeed.cfStreamAccumulatedLeft;
            pipelineDeps.cfStreamAccumulatedRight = &g_state.crossfeed.cfStreamAccumulatedRight;
            pipelineDeps.cfOutputLeft = &g_state.crossfeed.cfOutputLeft;
            pipelineDeps.cfOutputRight = &g_state.crossfeed.cfOutputRight;
            pipelineDeps.crossfeedEnabled = &g_state.crossfeed.enabled;
            pipelineDeps.crossfeedResetRequested = &g_state.crossfeed.resetRequested;
            pipelineDeps.crossfeedProcessor = g_state.crossfeed.processor;
            pipelineDeps.buffer.playbackBuffer = &daemon_output::playbackBuffer(g_state);
            pipelineDeps.maxOutputBufferFrames = []() {
                return daemon_output::maxOutputBufferFrames(g_state);
            };
            pipelineDeps.currentInputRate = []() {
                return g_state.rates.currentInputRate.load(std::memory_order_acquire);
            };
            pipelineDeps.currentOutputRate = []() {
                return g_state.rates.currentOutputRate.load(std::memory_order_acquire);
            };
            pipelineDeps.delimiterMode = &g_state.delimiter.mode;
            pipelineDeps.delimiterFallbackReason = &g_state.delimiter.fallbackReason;
            pipelineDeps.delimiterBypassLocked = &g_state.delimiter.bypassLocked;
            pipelineDeps.delimiterEnabled = &g_state.delimiter.enabled;
            pipelineDeps.delimiterWarmup = &g_state.delimiter.warmup;
            pipelineDeps.delimiterQueueSamples = &g_state.delimiter.queueSamples;
            pipelineDeps.delimiterQueueSeconds = &g_state.delimiter.queueSeconds;
            pipelineDeps.delimiterLastInferenceMs = &g_state.delimiter.lastInferenceMs;
            pipelineDeps.delimiterBackendAvailable = &g_state.delimiter.backendAvailable;
            pipelineDeps.delimiterBackendValid = &g_state.delimiter.backendValid;
            pipelineDeps.delimiterTargetMode = &g_state.delimiter.targetMode;
            g_state.audioPipeline =
                std::make_unique<audio_pipeline::AudioPipeline>(std::move(pipelineDeps));
            g_state.managers.audioPipelineRaw = g_state.audioPipeline.get();
        }

        // Check for early abort before starting threads
        if (!g_state.flags.running) {
            std::cout << "Startup interrupted by signal" << '\n';
            delete g_state.upsampler;
            g_state.upsampler = nullptr;
            break;
        }

        // Initialize soft mute controller with output sample rate
        using namespace DaemonConstants;
        int outputSampleRate = g_state.rates.inputSampleRate * g_state.config.upsampleRatio;
        g_state.softMute.controller =
            new SoftMute::Controller(DEFAULT_SOFT_MUTE_FADE_MS, outputSampleRate);
        std::cout << "Soft mute initialized (" << DEFAULT_SOFT_MUTE_FADE_MS << "ms fade at "
                  << outputSampleRate << "Hz)" << '\n';

        // Initialize fallback manager (Issue #139)
        if (g_state.config.fallback.enabled) {
            g_state.fallbackManager = new FallbackManager::Manager();
            FallbackManager::FallbackConfig fbConfig;
            fbConfig.gpuThreshold = g_state.config.fallback.gpuThreshold;
            fbConfig.gpuThresholdCount = g_state.config.fallback.gpuThresholdCount;
            fbConfig.gpuRecoveryThreshold = g_state.config.fallback.gpuRecoveryThreshold;
            fbConfig.gpuRecoveryCount = g_state.config.fallback.gpuRecoveryCount;
            fbConfig.xrunTriggersFallback = g_state.config.fallback.xrunTriggersFallback;
            fbConfig.monitorIntervalMs = g_state.config.fallback.monitorIntervalMs;

            // State change callback: update atomic flag and notify via ZeroMQ
            auto stateCallback = [](FallbackManager::FallbackState state) {
                bool isFallback = (state == FallbackManager::FallbackState::Fallback);
                g_state.fallbackActive.store(isFallback, std::memory_order_relaxed);

                // ZeroMQ notification is handled by STATS command response
                LOG_INFO("Fallback state changed: {}", isFallback ? "FALLBACK" : "NORMAL");
            };

            if (g_state.fallbackManager->initialize(fbConfig, stateCallback)) {
                std::cout << "Fallback manager initialized (GPU threshold: "
                          << fbConfig.gpuThreshold << "%, XRUN fallback: "
                          << (fbConfig.xrunTriggersFallback ? "enabled" : "disabled") << ")"
                          << '\n';
            } else {
                std::cerr << "Warning: Failed to initialize fallback manager" << '\n';
                delete g_state.fallbackManager;
                g_state.fallbackManager = nullptr;
                g_state.fallbackActive.store(false, std::memory_order_relaxed);
            }
        } else {
            std::cout << "Fallback manager disabled" << '\n';
            g_state.fallbackActive.store(false, std::memory_order_relaxed);
        }

        std::unique_ptr<daemon_control::ControlPlane> controlPlane;
        // Legacy RTP path was removed: ALSA/TCP-only configuration

        if (!g_state.managers.dacManager) {
            g_state.managers.dacManager =
                std::make_unique<dac::DacManager>(make_dac_dependencies());
        }
        if (!g_state.managers.dacManager) {
            std::cerr << "Failed to initialize DAC manager" << '\n';
            exitCode = 1;
            break;
        }
        g_state.managers.dacManagerRaw = g_state.managers.dacManager.get();

        daemon_control::ControlPlaneDependencies controlDeps{};
        controlDeps.config = &g_state.config;
        controlDeps.runningFlag = &g_state.flags.running;
        controlDeps.reloadRequested = &g_state.flags.reloadRequested;
        controlDeps.zmqBindFailed = &g_state.flags.zmqBindFailed;
        controlDeps.currentOutputRate = &g_state.rates.currentOutputRate;
        controlDeps.softMute = &g_state.softMute.controller;
        controlDeps.activePhaseType = &g_state.rates.activePhaseType;
        controlDeps.inputSampleRate = &g_state.rates.inputSampleRate;
        controlDeps.defaultAlsaDevice = DEFAULT_ALSA_DEVICE;
        controlDeps.dispatcher = g_state.managers.eventDispatcher.get();
        controlDeps.quitMainLoop = []() {};
        controlDeps.buildRuntimeStats = []() { return build_runtime_stats_dependencies(); };
        controlDeps.bufferCapacityFrames = []() {
            return daemon_output::maxOutputBufferFrames(g_state);
        };
        controlDeps.applySoftMuteForFilterSwitch = [](std::function<bool()> fn) {
            audio_pipeline::applySoftMuteForFilterSwitch(g_state, std::move(fn));
        };
        controlDeps.resetStreamingCachesForSwitch = []() {
            return audio_pipeline::resetStreamingCachesForSwitch(g_state);
        };
        controlDeps.refreshHeadroom = [](const std::string& reason) {
            if (g_state.managers.headroomController) {
                g_state.managers.headroomController->refreshCurrentHeadroom(reason);
            }
        };
        controlDeps.reinitializeStreamingForLegacyMode = []() {
            return audio_pipeline::reinitializeStreamingForLegacyMode(g_state);
        };
        controlDeps.setPreferredOutputDevice = [](AppConfig& cfg, const std::string& device) {
            set_preferred_output_device(cfg, device);
        };
        controlDeps.delimiterEnable = []() {
            return g_state.audioPipeline ? g_state.audioPipeline->requestDelimiterEnable() : false;
        };
        controlDeps.delimiterDisable = []() {
            return g_state.audioPipeline ? g_state.audioPipeline->requestDelimiterDisable() : false;
        };
        controlDeps.delimiterStatus = []() {
            if (!g_state.audioPipeline) {
                audio_pipeline::DelimiterStatusSnapshot snapshot;
                snapshot.enabled = false;
                snapshot.backendAvailable = false;
                snapshot.backendValid = false;
                return snapshot;
            }
            return g_state.audioPipeline->delimiterStatus();
        };
        controlDeps.dacManager = g_state.managers.dacManager.get();
        controlDeps.upsampler = &g_state.upsampler;
        controlDeps.crossfeed.processor = &g_state.crossfeed.processor;
        controlDeps.crossfeed.enabledFlag = &g_state.crossfeed.enabled;
        controlDeps.crossfeed.mutex = &g_state.crossfeed.crossfeedMutex;
        controlDeps.crossfeed.resetStreamingState = []() {
            g_state.crossfeed.resetRequested.store(true, std::memory_order_release);
        };
        controlDeps.statsFilePath = STATS_FILE_PATH;

        controlPlane = std::make_unique<daemon_control::ControlPlane>(std::move(controlDeps));
        if (controlPlane && controlPlane->start() && g_state.managers.dacManager) {
            g_state.managers.dacManager->setEventPublisher(controlPlane->eventPublisher());
        }

        if (g_state.managers.dacManager) {
            g_state.managers.dacManager->initialize();
        }

        // Start ALSA output thread
        if (g_state.managers.dacManager) {
            g_state.managers.dacManager->start();
        }
        std::cout << "Starting ALSA output thread..." << '\n';
        std::thread alsa_thread(daemon_output::alsaOutputThread, std::ref(g_state));

        std::thread loopback_thread;
        std::thread i2s_thread;
        bool startupFailed = false;

        auto failStartup = [&](const std::string& reason) {
            LOG_ERROR("Startup failed: {}", reason);
            exitCode = 1;
            startupFailed = true;
            g_state.flags.running = false;
            daemon_output::playbackBuffer(g_state).cv().notify_all();
        };

        if (g_state.config.i2s.enabled && g_state.config.loopback.enabled) {
            failStartup("Config error: i2s.enabled and loopback.enabled cannot both be true");
        }

        if (!startupFailed && g_state.config.i2s.enabled) {
            if (!daemon_input::validateI2sConfig(g_state.config)) {
                failStartup("Invalid I2S config");
            } else {
                snd_pcm_format_t i2s_format =
                    daemon_input::parseI2sFormat(g_state.config.i2s.format);
                // i2s.sampleRate==0 means "follow engine/runtime negotiated input rate".
                // ALSA hw_params typically requires an explicit rate, so default to current engine
                // rate.
                unsigned int i2s_rate =
                    (g_state.config.i2s.sampleRate != 0)
                        ? static_cast<unsigned int>(g_state.config.i2s.sampleRate)
                        : static_cast<unsigned int>(g_state.rates.inputSampleRate);
                std::cout << "Starting I2S capture thread (" << g_state.config.i2s.device
                          << ", fmt=" << g_state.config.i2s.format << ", rate=" << i2s_rate
                          << ", period=" << g_state.config.i2s.periodFrames << ")" << '\n';
                i2s_thread = std::thread(
                    daemon_input::i2sCaptureThread, std::ref(g_state), g_state.config.i2s.device,
                    i2s_format, i2s_rate, g_state.config.i2s.channels,
                    static_cast<snd_pcm_uframes_t>(g_state.config.i2s.periodFrames));

                // Wait briefly for I2S thread to become ready
                bool i2s_ready = false;
                for (int i = 0; i < 40; ++i) {  // up to ~2s
                    if (g_state.i2s.captureReady.load(std::memory_order_acquire)) {
                        i2s_ready = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (!i2s_ready) {
                    failStartup("[I2S] Failed to start capture thread (not ready)");
                }
            }
        }
        if (!startupFailed && g_state.config.loopback.enabled) {
            if (!daemon_input::validateLoopbackConfig(g_state.config)) {
                failStartup("Invalid loopback config");
            } else {
                snd_pcm_format_t lb_format =
                    daemon_input::parseLoopbackFormat(g_state.config.loopback.format);
                std::cout << "Starting loopback capture thread (" << g_state.config.loopback.device
                          << ", fmt=" << g_state.config.loopback.format
                          << ", rate=" << g_state.config.loopback.sampleRate
                          << ", period=" << g_state.config.loopback.periodFrames << ")" << '\n';

                daemon_input::LoopbackCaptureDependencies loopbackDeps{
                    .playbackBuffer = &daemon_output::playbackBuffer(g_state),
                    .running = &g_state.flags.running,
                    .currentOutputRate = &g_state.rates.currentOutputRate,
                    .audioPipeline = &g_state.managers.audioPipelineRaw,
                    .handleMutex = &g_state.loopback.handleMutex,
                    .handle = &g_state.loopback.handle,
                    .captureRunning = &g_state.loopback.captureRunning,
                    .captureReady = &g_state.loopback.captureReady,
                };

                loopback_thread = std::thread(
                    daemon_input::loopbackCaptureThread, g_state.config.loopback.device, lb_format,
                    g_state.config.loopback.sampleRate, g_state.config.loopback.channels,
                    static_cast<snd_pcm_uframes_t>(g_state.config.loopback.periodFrames),
                    loopbackDeps);

                // Wait briefly for loopback thread to become ready
                bool loopback_ready = false;
                for (int i = 0; i < 40; ++i) {  // up to ~2s
                    if (g_state.loopback.captureReady.load(std::memory_order_acquire)) {
                        loopback_ready = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (!loopback_ready) {
                    failStartup("[Loopback] Failed to start capture thread (not ready)");
                }
            }
        }

        if (!startupFailed) {
            double outputRateKHz =
                g_state.rates.inputSampleRate * g_state.config.upsampleRatio / 1000.0;
            std::cout << '\n';
            if (g_state.config.loopback.enabled) {
                std::cout << "System ready (Loopback capture mode). Audio routing configured:"
                          << '\n';
                std::cout << "  1. Loopback capture → GPU Upsampler ("
                          << g_state.config.upsampleRatio << "x upsampling)" << '\n';
                std::cout << "  2. GPU Upsampler → ALSA → SMSL DAC (" << outputRateKHz
                          << "kHz direct)" << '\n';
            } else if (g_state.config.i2s.enabled) {
                std::cout << "System ready (I2S capture mode). Audio routing configured:" << '\n';
                std::cout << "  1. I2S capture → GPU Upsampler (" << g_state.config.upsampleRatio
                          << "x upsampling)" << '\n';
                std::cout << "  2. GPU Upsampler → ALSA → DAC (" << outputRateKHz << "kHz direct)"
                          << '\n';
            } else {
                std::cout << "System ready (TCP/loopback input). Audio routing configured:" << '\n';
                std::cout << "  1. Network or loopback source → GPU Upsampler ("
                          << g_state.config.upsampleRatio << "x upsampling)" << '\n';
                std::cout << "  2. GPU Upsampler → ALSA → SMSL DAC (" << outputRateKHz
                          << "kHz direct)" << '\n';
            }
            std::cout << "Press Ctrl+C to stop." << '\n';
            std::cout << "========================================" << '\n';

            shutdownManager.notifyReady();

            auto runMainLoop = [&]() {
                while (g_state.flags.running.load() && !g_state.flags.reloadRequested.load() &&
                       !g_state.flags.zmqBindFailed.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    if (g_state.managers.streamingCacheManager) {
                        (void)g_state.managers.streamingCacheManager->drainFlushRequests();
                    }
                    // Input rate follow-up hook (Issue #906; network-driven detection is #824)
                    int pending =
                        g_state.rates.pendingRateChange.exchange(0, std::memory_order_acq_rel);
                    if (pending > 0 && (pending == 44100 || pending == 48000)) {
                        (void)audio_pipeline::handleRateSwitch(g_state, pending);
                    }
                    shutdownManager.tick();
                }
            };

            if (g_state.flags.zmqBindFailed.load()) {
                std::cerr << "Startup aborted due to ZeroMQ bind failure." << '\n';
            } else {
                runMainLoop();
            }
        }

        shutdownManager.runShutdownSequence();

        // Step 5: Signal worker threads to stop and wait for them
        std::cout << "  Step 5: Stopping worker threads..." << '\n';
        g_state.flags.running = false;
        daemon_output::playbackBuffer(g_state).cv().notify_all();
        if (controlPlane) {
            controlPlane->stop();
        }
        alsa_thread.join();  // ALSA thread will call snd_pcm_drain() before exit
        if (i2s_thread.joinable()) {
            i2s_thread.join();
        }
        if (loopback_thread.joinable()) {
            loopback_thread.join();
        }

        // Step 6: Release audio processing resources
        std::cout << "  Step 6: Releasing resources..." << '\n';
        if (g_state.fallbackManager) {
            g_state.fallbackManager->shutdown();
            delete g_state.fallbackManager;
            g_state.fallbackManager = nullptr;
            g_state.fallbackActive.store(false, std::memory_order_relaxed);
        }
        delete g_state.softMute.controller;
        g_state.softMute.controller = nullptr;
        daemon_audio::shutdownCrossfeed(g_state.crossfeed);
        if (g_state.audioPipeline) {
            g_state.audioPipeline.reset();  // Tear down pipeline before deleting GPU upsampler
            g_state.managers.audioPipelineRaw = nullptr;
        }
        delete g_state.upsampler;
        g_state.upsampler = nullptr;
        if (g_state.managers.dacManager) {
            g_state.managers.dacManager->stop();
        }

        // Don't reload if ZMQ bind failed - exit completely
        if (g_state.flags.zmqBindFailed.load()) {
            std::cerr << "Exiting due to ZeroMQ initialization failure." << '\n';
            exitCode = 1;
            break;
        }

        if (g_state.flags.reloadRequested) {
            std::cout << "Reload requested. Restarting daemon with updated config..." << '\n';
        }
    } while (g_state.flags.reloadRequested);

    std::cout << "Goodbye!" << '\n';
    return exitCode;
}
