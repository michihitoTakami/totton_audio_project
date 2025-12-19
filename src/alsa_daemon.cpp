#include "audio/audio_utils.h"
#include "audio/eq_parser.h"
#include "audio/eq_to_fir.h"
#include "audio/fallback_manager.h"
#include "audio/filter_headroom.h"
#include "audio/soft_mute.h"
#include "convolution_engine.h"
#include "core/config_loader.h"
#include "core/daemon_constants.h"
#include "core/partition_runtime_utils.h"
#include "daemon/api/dependencies.h"
#include "daemon/api/events.h"
#include "daemon/app/runtime_state.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/audio_pipeline/filter_manager.h"
#include "daemon/audio_pipeline/rate_switcher.h"
#include "daemon/audio_pipeline/soft_mute_runner.h"
#include "daemon/audio_pipeline/streaming_cache_manager.h"
#include "daemon/control/control_plane.h"
#include "daemon/control/handlers/handler_registry.h"
#include "daemon/metrics/runtime_stats.h"
#include "daemon/output/alsa_output.h"
#include "daemon/output/alsa_pcm_controller.h"
#include "daemon/output/alsa_write_loop.h"
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
#include <filesystem>
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
#include <sys/file.h>
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

// PID file path (also serves as lock file)
constexpr const char* PID_FILE_PATH = "/tmp/gpu_upsampler_alsa.pid";

static void refresh_current_headroom(const std::string& reason);

static void enforce_phase_partition_constraints(AppConfig& config) {
    if (config.partitionedConvolution.enabled && config.phaseType == PhaseType::Linear) {
        std::cout << "[Partition] Linear phase is incompatible with low-latency mode. "
                  << "Switching to minimum phase." << '\n';
        config.phaseType = PhaseType::Minimum;
    }
}

static bool parse_env_bool(const char* value, bool defaultValue) {
    if (!value) {
        return defaultValue;
    }
    std::string v(value);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (v == "1" || v == "true" || v == "yes" || v == "y" || v == "on") {
        return true;
    }
    if (v == "0" || v == "false" || v == "no" || v == "n" || v == "off") {
        return false;
    }
    return defaultValue;
}

static void apply_partition_env_overrides(AppConfig& config) {
    // Debug/mitigation toggles for Issue #969 investigations.
    //
    // - MAGICBOX_PARTITIONED_CONVOLUTION_ENABLED: force enable/disable (true/false/1/0).
    // - MAGICBOX_PARTITIONED_CONVOLUTION_MAX_PARTITIONS: cap maxPartitions (>0).
    if (const char* v = std::getenv("MAGICBOX_PARTITIONED_CONVOLUTION_ENABLED")) {
        bool enabled = parse_env_bool(v, config.partitionedConvolution.enabled);
        if (enabled != config.partitionedConvolution.enabled) {
            std::cout << "[Partition] Override: partitionedConvolution.enabled "
                      << (config.partitionedConvolution.enabled ? "true" : "false") << " -> "
                      << (enabled ? "true" : "false") << '\n';
        }
        config.partitionedConvolution.enabled = enabled;
    }
    if (const char* v = std::getenv("MAGICBOX_PARTITIONED_CONVOLUTION_MAX_PARTITIONS")) {
        try {
            int cap = std::stoi(std::string(v));
            if (cap > 0 && cap != config.partitionedConvolution.maxPartitions) {
                std::cout << "[Partition] Override: partitionedConvolution.maxPartitions "
                          << config.partitionedConvolution.maxPartitions << " -> " << cap << '\n';
                config.partitionedConvolution.maxPartitions = cap;
            }
        } catch (...) {
            // ignore invalid
        }
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
constexpr const char* DEFAULT_FILTER_PATH = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
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

inline ConvolutionEngine::RateFamily g_get_rate_family() {
    return static_cast<ConvolutionEngine::RateFamily>(
        g_state.rates.currentRateFamilyInt.load(std::memory_order_acquire));
}

inline void g_set_rate_family(ConvolutionEngine::RateFamily family) {
    g_state.rates.currentRateFamilyInt.store(static_cast<int>(family), std::memory_order_release);
}

static size_t get_max_output_buffer_frames() {
    using namespace DaemonConstants;
    auto seconds = static_cast<double>(MAX_OUTPUT_BUFFER_SECONDS);
    if (seconds <= 0.0) {
        return DEFAULT_MAX_OUTPUT_BUFFER_FRAMES;
    }

    int outputRate = g_state.rates.currentOutputRate.load(std::memory_order_acquire);
    if (outputRate <= 0) {
        outputRate = DEFAULT_OUTPUT_SAMPLE_RATE;
    }

    double frames = seconds * static_cast<double>(outputRate);
    if (frames <= 0.0) {
        return DEFAULT_MAX_OUTPUT_BUFFER_FRAMES;
    }
    return static_cast<size_t>(frames);
}

static size_t compute_stream_buffer_capacity(size_t streamValidInputPerBlock) {
    using namespace DaemonConstants;
    size_t frames = static_cast<size_t>(DEFAULT_BLOCK_SIZE);
    if (g_state.config.blockSize > 0) {
        frames = std::max(frames, static_cast<size_t>(g_state.config.blockSize));
    }
    if (g_state.config.periodSize > 0) {
        frames = std::max(frames, static_cast<size_t>(g_state.config.periodSize));
    }
    if (g_state.config.loopback.periodFrames > 0) {
        frames = std::max(frames, static_cast<size_t>(g_state.config.loopback.periodFrames));
    }
    frames = std::max(frames, streamValidInputPerBlock);
    // 2x safety margin for bursty upstream (no reallocation in RT path)
    return frames * 2;
}

// Pending rate change (set by input event handlers, processed in main loop)
// Value: 0 = no change pending, >0 = detected input sample rate

static void maybe_restore_soft_mute_params() {
    if (!g_state.softMute.controller) {
        return;
    }
    if (!g_state.softMute.restorePending.load(std::memory_order_acquire)) {
        return;
    }
    SoftMute::MuteState st = g_state.softMute.controller->getState();
    if (st != SoftMute::MuteState::PLAYING && st != SoftMute::MuteState::MUTED) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_state.softMute.opMutex);
    if (!g_state.softMute.controller) {
        return;
    }
    int fadeMs = g_state.softMute.restoreFadeMs.load(std::memory_order_relaxed);
    int sr = g_state.softMute.restoreSampleRate.load(std::memory_order_relaxed);
    g_state.softMute.controller->setFadeDuration(fadeMs);
    g_state.softMute.controller->setSampleRate(sr);
    g_state.softMute.restorePending.store(false, std::memory_order_release);
}

// Helper function for soft mute during filter switching (Issue #266)
// Fade-out: 1.5 seconds, perform filter switch, fade-in: 1.5 seconds
//
// Thread safety & responsiveness:
// - Called from ZeroMQ command thread, guarded by a mutex to serialize parameter updates
// - Non-blocking: start fade-out, perform switch, then trigger fade-in with minimal wait
// - Original fade parameters are restored in the audio thread once the transition settles

static void applySoftMuteForFilterSwitch(std::function<bool()> filterSwitchFunc) {
    using namespace DaemonConstants;

    if (!g_state.softMute.controller) {
        // If soft mute not initialized, perform switch without mute
        filterSwitchFunc();
        return;
    }

    std::lock_guard<std::mutex> lock(g_state.softMute.opMutex);

    // Cancel any stale pending restore (new switch supersedes)
    g_state.softMute.restorePending.store(false, std::memory_order_release);

    // Save current fade duration for restoration
    int originalFadeDuration = g_state.softMute.controller->getFadeDuration();
    int outputSampleRate = g_state.softMute.controller->getSampleRate();

    // Update fade duration for filter switching
    // Note: This is called from command thread, but audio thread may be processing.
    // The fade calculation will use the new duration from the next audio frame.
    g_state.softMute.controller->setFadeDuration(FILTER_SWITCH_FADE_MS);
    g_state.softMute.controller->setSampleRate(outputSampleRate);

    std::cout << "[Filter Switch] Starting fade-out (" << (FILTER_SWITCH_FADE_MS / 1000.0)
              << "s)..." << '\n';
    g_state.softMute.controller->startFadeOut();

    // Wait briefly until mostly muted (or timeout) to avoid pops, but keep command responsive
    auto fade_start = std::chrono::steady_clock::now();
    const auto quick_timeout = std::chrono::milliseconds(250);
    while (true) {
        SoftMute::MuteState st = g_state.softMute.controller->getState();
        float gain = g_state.softMute.controller->getCurrentGain();
        if (st == SoftMute::MuteState::MUTED || gain < 0.05f) {
            break;
        }
        if (std::chrono::steady_clock::now() - fade_start > quick_timeout) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Perform filter switch while fade-out is progressing
    bool pauseOk = true;
    if (g_state.audioPipeline) {
        g_state.audioPipeline->requestRtPause();
        pauseOk = g_state.audioPipeline->waitForRtQuiescent(std::chrono::milliseconds(500));
        if (!pauseOk) {
            LOG_ERROR("[Filter Switch] RT pause handshake timed out (aborting switch)");
        }
    }

    bool switch_success = false;
    if (pauseOk) {
        switch_success = filterSwitchFunc();
    }

    if (g_state.audioPipeline) {
        g_state.audioPipeline->resumeRtPause();
    }

    if (switch_success) {
        // Start fade-in after filter switch
        std::cout << "[Filter Switch] Starting fade-in (" << (FILTER_SWITCH_FADE_MS / 1000.0)
                  << "s)..." << '\n';
        g_state.softMute.controller->startFadeIn();

        // Mark pending restoration to be applied once transition completes
        g_state.softMute.restoreFadeMs.store(originalFadeDuration, std::memory_order_relaxed);
        g_state.softMute.restoreSampleRate.store(outputSampleRate, std::memory_order_relaxed);
        g_state.softMute.restorePending.store(true, std::memory_order_release);
    } else {
        // If switch failed, restore original state immediately
        std::cerr << "[Filter Switch] Switch failed, restoring audio state" << '\n';
        g_state.softMute.controller->setPlaying();
        g_state.softMute.controller->setFadeDuration(originalFadeDuration);
        g_state.softMute.controller->setSampleRate(outputSampleRate);
    }
}

// Audio buffer for thread communication (managed component)
static daemon_output::PlaybackBufferManager& playback_buffer() {
    if (!g_state.playback.buffer) {
        g_state.playback.buffer = std::make_unique<daemon_output::PlaybackBufferManager>(
            []() { return get_max_output_buffer_frames(); });
    }
    return *g_state.playback.buffer;
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
        refresh_current_headroom(reason);
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

static void elevate_realtime_priority(const char* name, int priority = 65) {
#ifdef __linux__
    // RT scheduling can freeze remote shells if something spins.
    // Allow disabling via env for containerized debugging.
    // - MAGICBOX_ENABLE_RT=0 disables SCHED_FIFO attempts.
    const char* enableRt = std::getenv("MAGICBOX_ENABLE_RT");
    if (enableRt && std::string(enableRt) == "0") {
        LOG_WARN("[RT] {} thread: SCHED_FIFO disabled via MAGICBOX_ENABLE_RT=0", name);
        return;
    }
    sched_param params{};
    params.sched_priority = priority;
    int ret = pthread_setschedparam(pthread_self(), SCHED_FIFO, &params);
    if (ret != 0) {
        LOG_WARN("[RT] Failed to set {} thread to SCHED_FIFO (errno={}): {}", name, ret,
                 std::strerror(ret));
    } else {
        LOG_INFO("[RT] {} thread priority set to SCHED_FIFO {}", name, priority);
    }
#else
    (void)name;
    (void)priority;
#endif
}

static size_t get_playback_ready_threshold(size_t period_size) {
    bool crossfeedActive = false;
    size_t crossfeedBlockSize = 0;
    size_t producerBlockSize = 0;

    if (g_state.crossfeed.enabled.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> cf_lock(g_state.crossfeed.crossfeedMutex);
        if (g_state.crossfeed.processor) {
            crossfeedActive = true;
            crossfeedBlockSize = g_state.crossfeed.processor->getStreamValidInputPerBlock();
        }
    }

    if (g_state.upsampler) {
        // StreamValidInputPerBlock() is in input frames. Multiply by upsample ratio to obtain the
        // number of samples the producer actually contributes to the playback ring per block so the
        // ALSA thread can wake up as soon as a full GPU block finishes.
        size_t streamBlock = g_state.upsampler->getStreamValidInputPerBlock();
        int upsampleRatio = g_state.upsampler->getUpsampleRatio();
        if (streamBlock > 0 && upsampleRatio > 0) {
            producerBlockSize = streamBlock * static_cast<size_t>(upsampleRatio);
        }
    }

    return PlaybackBuffer::computeReadyThreshold(period_size, crossfeedActive, crossfeedBlockSize,
                                                 producerBlockSize);
}

// Fallback manager (Issue #139)

// Resets crossfeed streaming buffers and GPU overlap state.
// Caller must hold g_state.crossfeed.crossfeedMutex.
static void reset_crossfeed_stream_state_locked() {
    if (!g_state.crossfeed.cfStreamInputLeft.empty()) {
        std::fill(g_state.crossfeed.cfStreamInputLeft.begin(),
                  g_state.crossfeed.cfStreamInputLeft.end(), 0.0f);
    }
    if (!g_state.crossfeed.cfStreamInputRight.empty()) {
        std::fill(g_state.crossfeed.cfStreamInputRight.begin(),
                  g_state.crossfeed.cfStreamInputRight.end(), 0.0f);
    }
    g_state.crossfeed.cfStreamAccumulatedLeft = 0;
    g_state.crossfeed.cfStreamAccumulatedRight = 0;
    g_state.crossfeed.cfOutputLeft.clear();
    g_state.crossfeed.cfOutputRight.clear();
    if (g_state.crossfeed.processor) {
        g_state.crossfeed.processor->resetStreaming();
    }
}

// Crossfeed enable/disable safety (Issue #888)
// - Avoid mixing pre/post switch audio by clearing playback + streaming caches.
// - Do not touch SoftMute here; caller wraps this with a fade-out/in.
static bool reset_streaming_caches_for_switch() {
    struct RtPauseGuard {
        audio_pipeline::AudioPipeline* pipeline = nullptr;
        bool ok = true;
        explicit RtPauseGuard(audio_pipeline::AudioPipeline* p) : pipeline(p) {
            if (!pipeline) {
                return;
            }
            pipeline->requestRtPause();
            ok = pipeline->waitForRtQuiescent(std::chrono::milliseconds(500));
            if (!ok) {
                pipeline->resumeRtPause();
            }
        }
        ~RtPauseGuard() {
            if (pipeline && ok) {
                pipeline->resumeRtPause();
            }
        }
    } pauseGuard(g_state.audioPipeline.get());

    if (!pauseGuard.ok) {
        return false;
    }

    playback_buffer().reset();
    playback_buffer().cv().notify_all();

    {
        std::lock_guard<std::mutex> lock(g_state.streaming.streamingMutex);
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
        if (g_state.upsampler) {
            g_state.upsampler->resetStreaming();
        }
    }

    { g_state.crossfeed.resetRequested.store(true, std::memory_order_release); }

    return true;
}

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

        playback_buffer().reset();

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
    return deps;
}

// ========== PID File Lock (flock-based) ==========

// File descriptor for the PID lock file (kept open while running)

// Read PID from lock file (for display purposes)
static pid_t read_pid_from_lockfile() {
    std::ifstream pidfile(PID_FILE_PATH);
    if (!pidfile.is_open()) {
        return 0;
    }
    pid_t pid = 0;
    pidfile >> pid;
    return pid;
}

// Acquire exclusive lock on PID file using flock()
// Returns true if lock acquired, false if another instance is running
static bool acquire_pid_lock() {
    // Open or create the lock file
    g_state.pidLockFd = open(PID_FILE_PATH, O_RDWR | O_CREAT, 0644);
    if (g_state.pidLockFd < 0) {
        LOG_ERROR("Cannot open PID file: {} ({})", PID_FILE_PATH, strerror(errno));
        return false;
    }

    // Try to acquire exclusive lock (non-blocking)
    if (flock(g_state.pidLockFd, LOCK_EX | LOCK_NB) < 0) {
        if (errno == EWOULDBLOCK) {
            // Another process holds the lock
            pid_t existing_pid = read_pid_from_lockfile();
            if (existing_pid > 0) {
                LOG_ERROR("Another instance is already running (PID: {})", existing_pid);
            } else {
                LOG_ERROR("Another instance is already running");
            }
            LOG_ERROR("Use './scripts/daemon.sh stop' to stop it.");
        } else {
            LOG_ERROR("Cannot lock PID file: {}", strerror(errno));
        }
        close(g_state.pidLockFd);
        g_state.pidLockFd = -1;
        return false;
    }

    // Lock acquired - write our PID to the file
    if (ftruncate(g_state.pidLockFd, 0) < 0) {
        LOG_WARN("Cannot truncate PID file");
    }
    dprintf(g_state.pidLockFd, "%d\n", getpid());
    fsync(g_state.pidLockFd);  // Ensure PID is written to disk

    return true;
}

// Release PID lock and remove file
// Note: Lock is automatically released when process exits (even on crash)
static void release_pid_lock() {
    if (g_state.pidLockFd >= 0) {
        flock(g_state.pidLockFd, LOCK_UN);
        close(g_state.pidLockFd);
        g_state.pidLockFd = -1;
    }
    // Remove the PID file on clean shutdown
    unlink(PID_FILE_PATH);
    // Remove the stats file on clean shutdown
    unlink(STATS_FILE_PATH);
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

static std::string resolve_filter_path_for(ConvolutionEngine::RateFamily family, PhaseType phase) {
    if (phase == PhaseType::Linear) {
        return (family == ConvolutionEngine::RateFamily::RATE_44K)
                   ? g_state.config.filterPath44kLinear
                   : g_state.config.filterPath48kLinear;
    }
    return (family == ConvolutionEngine::RateFamily::RATE_44K) ? g_state.config.filterPath44kMin
                                                               : g_state.config.filterPath48kMin;
}

static std::string current_filter_path() {
    return resolve_filter_path_for(g_state.rates.activeRateFamily, g_state.rates.activePhaseType);
}

static void update_effective_gain(float headroomGain, const std::string& reason) {
    g_state.gains.headroom.store(headroomGain, std::memory_order_relaxed);
    float effective = g_state.config.gain * headroomGain;
    g_state.gains.output.store(effective, std::memory_order_relaxed);
    LOG_INFO("Gain [{}]: user {:.4f} * headroom {:.4f} = {:.4f}", reason, g_state.config.gain,
             headroomGain, effective);
}

static void apply_headroom_for_path(const std::string& path, const std::string& reason) {
    if (path.empty()) {
        LOG_WARN("Headroom [{}]: empty filter path, falling back to unity gain", reason);
        update_effective_gain(1.0f, reason);
        return;
    }

    FilterHeadroomInfo info = g_state.headroomCache.get(path);
    update_effective_gain(info.safeGain, reason);

    if (!info.metadataFound) {
        LOG_WARN("Headroom [{}]: metadata missing for {} (using safe gain {:.4f})", reason, path,
                 info.safeGain);
    } else {
        LOG_INFO("Headroom [{}]: {} max_coef={:.6f} safeGain={:.4f} target={:.2f}", reason, path,
                 info.maxCoefficient, info.safeGain, info.targetPeak);
    }
}

static void refresh_current_headroom(const std::string& reason) {
    apply_headroom_for_path(current_filter_path(), reason);
}

static void reset_runtime_state() {
    g_state.managers.streamingCacheManager.reset();

    playback_buffer().reset();
    g_state.streaming.streamInputLeft.clear();
    g_state.streaming.streamInputRight.clear();
    g_state.streaming.streamAccumulatedLeft = 0;
    g_state.streaming.streamAccumulatedRight = 0;
    g_state.streaming.upsamplerOutputLeft.clear();
    g_state.streaming.upsamplerOutputRight.clear();

    // Reset crossfeed streaming buffers
    g_state.crossfeed.cfStreamInputLeft.clear();
    g_state.crossfeed.cfStreamInputRight.clear();
    g_state.crossfeed.cfStreamAccumulatedLeft = 0;
    g_state.crossfeed.cfStreamAccumulatedRight = 0;
    g_state.crossfeed.cfOutputBufferLeft.clear();
    g_state.crossfeed.cfOutputBufferRight.clear();
}

static bool reinitialize_streaming_for_legacy_mode() {
    if (!g_state.upsampler) {
        return false;
    }

    std::lock_guard<std::mutex> streamLock(g_state.streaming.streamingMutex);
    g_state.upsampler->resetStreaming();
    playback_buffer().reset();

    g_state.streaming.streamInputLeft.clear();
    g_state.streaming.streamInputRight.clear();
    g_state.streaming.streamAccumulatedLeft = 0;
    g_state.streaming.streamAccumulatedRight = 0;
    g_state.streaming.upsamplerOutputLeft.clear();
    g_state.streaming.upsamplerOutputRight.clear();

    // Rebuild legacy streams so the buffers match the full FFT (avoids invalid cudaMemset after
    // disabling partitions).
    if (!g_state.upsampler->initializeStreaming()) {
        std::cerr << "[Partition] Failed to initialize legacy streaming buffers" << '\n';
        return false;
    }

    size_t buffer_capacity =
        compute_stream_buffer_capacity(g_state.upsampler->getStreamValidInputPerBlock());
    g_state.streaming.streamInputLeft.resize(buffer_capacity, 0.0f);
    g_state.streaming.streamInputRight.resize(buffer_capacity, 0.0f);
    g_state.streaming.streamAccumulatedLeft = 0;
    g_state.streaming.streamAccumulatedRight = 0;

    size_t upsampler_output_capacity =
        buffer_capacity * static_cast<size_t>(g_state.upsampler->getUpsampleRatio());
    g_state.streaming.upsamplerOutputLeft.reserve(upsampler_output_capacity);
    g_state.streaming.upsamplerOutputRight.reserve(upsampler_output_capacity);

    return true;
}

static bool handle_rate_switch(int newInputRate) {
    if (!g_state.upsampler || !g_state.upsampler->isMultiRateEnabled()) {
        std::cerr << "[Rate] Multi-rate mode not enabled" << '\n';
        return false;
    }

    int currentRate = g_state.upsampler->getCurrentInputRate();
    if (currentRate == newInputRate) {
        std::cout << "[Rate] Already at target rate: " << newInputRate << " Hz" << '\n';
        return true;
    }

    std::cout << "[Rate] Switching: " << currentRate << " Hz -> " << newInputRate << " Hz" << '\n';

    int savedRate = currentRate;

    if (g_state.softMute.controller) {
        g_state.softMute.controller->startFadeOut();
        auto startTime = std::chrono::steady_clock::now();
        while (g_state.softMute.controller->isTransitioning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto elapsed = std::chrono::steady_clock::now() - startTime;
            if (elapsed > std::chrono::milliseconds(200)) {
                std::cerr << "[Rate] Warning: Fade-out timeout" << '\n';
                break;
            }
        }
    }

    int newOutputRate = g_state.upsampler->getOutputSampleRate();
    int newUpsampleRatio = g_state.upsampler->getUpsampleRatio();
    size_t buffer_capacity = 0;

    struct RtPauseGuard {
        audio_pipeline::AudioPipeline* pipeline = nullptr;
        bool ok = true;
        explicit RtPauseGuard(audio_pipeline::AudioPipeline* p) : pipeline(p) {
            if (!pipeline) {
                return;
            }
            pipeline->requestRtPause();
            ok = pipeline->waitForRtQuiescent(std::chrono::milliseconds(500));
            if (!ok) {
                LOG_ERROR("[Rate] RT pause handshake timed out (aborting rate switch)");
            }
        }
        ~RtPauseGuard() {
            if (pipeline) {
                pipeline->resumeRtPause();
            }
        }
    } pauseGuard(g_state.audioPipeline.get());

    if (!pauseGuard.ok) {
        if (g_state.softMute.controller) {
            g_state.softMute.controller->setPlaying();
        }
        return false;
    }

    {
        std::lock_guard<std::mutex> streamLock(g_state.streaming.streamingMutex);

        g_state.upsampler->resetStreaming();

        playback_buffer().reset();
        g_state.streaming.streamInputLeft.clear();
        g_state.streaming.streamInputRight.clear();
        g_state.streaming.streamAccumulatedLeft = 0;
        g_state.streaming.streamAccumulatedRight = 0;

        if (!g_state.upsampler->switchToInputRate(newInputRate)) {
            std::cerr << "[Rate] Failed to switch rate, rolling back" << '\n';
            if (g_state.upsampler->switchToInputRate(savedRate)) {
                std::cout << "[Rate] Rollback successful: restored to " << savedRate << " Hz"
                          << '\n';
            } else {
                std::cerr << "[Rate] ERROR: Rollback failed!" << '\n';
            }
            if (g_state.softMute.controller) {
                g_state.softMute.controller->startFadeIn();
            }
            return false;
        }

        if (!g_state.upsampler->initializeStreaming()) {
            std::cerr << "[Rate] Failed to re-initialize streaming mode, rolling back" << '\n';
            if (g_state.upsampler->switchToInputRate(savedRate)) {
                if (g_state.upsampler->initializeStreaming()) {
                    std::cout << "[Rate] Rollback successful: restored to " << savedRate << " Hz"
                              << '\n';
                }
            }
            if (g_state.softMute.controller) {
                g_state.softMute.controller->startFadeIn();
            }
            return false;
        }

        g_state.rates.inputSampleRate = newInputRate;
        newOutputRate = g_state.upsampler->getOutputSampleRate();
        newUpsampleRatio = g_state.upsampler->getUpsampleRatio();

        buffer_capacity =
            compute_stream_buffer_capacity(g_state.upsampler->getStreamValidInputPerBlock());
        g_state.streaming.streamInputLeft.resize(buffer_capacity, 0.0f);
        g_state.streaming.streamInputRight.resize(buffer_capacity, 0.0f);
        g_state.streaming.streamAccumulatedLeft = 0;
        g_state.streaming.streamAccumulatedRight = 0;
        size_t upsampler_output_capacity =
            buffer_capacity * static_cast<size_t>(g_state.upsampler->getUpsampleRatio());
        g_state.streaming.upsamplerOutputLeft.reserve(upsampler_output_capacity);
        g_state.streaming.upsamplerOutputRight.reserve(upsampler_output_capacity);

        if (g_state.softMute.controller) {
            delete g_state.softMute.controller;
        }
        g_state.softMute.controller = new SoftMute::Controller(50, newOutputRate);
    }

    if (g_state.softMute.controller) {
        g_state.softMute.controller->startFadeIn();
    }

    std::cout << "[Rate] Switch complete: " << newInputRate << " Hz (" << newUpsampleRatio
              << "x -> " << newOutputRate << " Hz)" << '\n';
    std::cout << "[Rate] Streaming buffers re-initialized: " << buffer_capacity
              << " samples capacity" << '\n';

    return true;
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

    if (!found) {
        std::cout << "Config: Using defaults (no config.json found)" << '\n';
    }

    apply_partition_env_overrides(g_state.config);
    enforce_phase_partition_constraints(g_state.config);

    print_config_summary(g_state.config);
    g_state.headroomCache.setTargetPeak(g_state.config.headroomTarget);
    update_effective_gain(1.0f, "config load (pending filter headroom)");
    float initialOutput = g_state.gains.output.load(std::memory_order_relaxed);
    g_state.gains.limiter.store(1.0f, std::memory_order_relaxed);
    g_state.gains.effective.store(initialOutput, std::memory_order_relaxed);
}

static snd_pcm_format_t parse_loopback_format(const std::string& formatStr) {
    std::string lower = formatStr;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "s16_le" || lower == "s16") {
        return SND_PCM_FORMAT_S16_LE;
    }
    if (lower == "s24_3le" || lower == "s24") {
        return SND_PCM_FORMAT_S24_3LE;
    }
    if (lower == "s32_le" || lower == "s32") {
        return SND_PCM_FORMAT_S32_LE;
    }
    return SND_PCM_FORMAT_UNKNOWN;
}

static snd_pcm_format_t parse_i2s_format(const std::string& formatStr) {
    // Same set as loopback (MVP)
    return parse_loopback_format(formatStr);
}

static bool validate_loopback_config(const AppConfig& cfg) {
    if (!cfg.loopback.enabled) {
        return true;
    }
    if (cfg.loopback.sampleRate != 44100 && cfg.loopback.sampleRate != 48000) {
        LOG_ERROR("[Loopback] Unsupported sample rate {} (expected 44100 or 48000)",
                  cfg.loopback.sampleRate);
        return false;
    }
    if (cfg.loopback.channels != CHANNELS) {
        LOG_ERROR("[Loopback] Unsupported channels {} (expected {})", cfg.loopback.channels,
                  CHANNELS);
        return false;
    }
    if (cfg.loopback.periodFrames == 0) {
        LOG_ERROR("[Loopback] periodFrames must be > 0");
        return false;
    }
    if (parse_loopback_format(cfg.loopback.format) == SND_PCM_FORMAT_UNKNOWN) {
        LOG_ERROR("[Loopback] Unsupported format '{}'", cfg.loopback.format);
        return false;
    }
    return true;
}

static bool validate_i2s_config(const AppConfig& cfg) {
    if (!cfg.i2s.enabled) {
        return true;
    }
    if (cfg.i2s.device.empty()) {
        LOG_ERROR("[I2S] device must not be empty");
        return false;
    }
    if (cfg.i2s.channels != CHANNELS) {
        LOG_ERROR("[I2S] Unsupported channels {} (expected {})", cfg.i2s.channels, CHANNELS);
        return false;
    }
    if (cfg.i2s.periodFrames == 0) {
        LOG_ERROR("[I2S] periodFrames must be > 0");
        return false;
    }
    if (parse_i2s_format(cfg.i2s.format) == SND_PCM_FORMAT_UNKNOWN) {
        LOG_ERROR("[I2S] Unsupported format '{}'", cfg.i2s.format);
        return false;
    }
    // sampleRate may be 0 (follow runtime/negotiated); otherwise accept MVP rates.
    if (cfg.i2s.sampleRate != 0 && cfg.i2s.sampleRate != 44100 && cfg.i2s.sampleRate != 48000) {
        LOG_ERROR("[I2S] Unsupported sample rate {} (expected 0/44100/48000)", cfg.i2s.sampleRate);
        return false;
    }
    return true;
}

static snd_pcm_t* open_loopback_capture(const std::string& device, snd_pcm_format_t format,
                                        unsigned int rate, unsigned int channels,
                                        snd_pcm_uframes_t& period_frames) {
    snd_pcm_t* handle = nullptr;
    // Use non-blocking mode so shutdown doesn't hang on snd_pcm_readi().
    int err = snd_pcm_open(&handle, device.c_str(), SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK);
    if (err < 0) {
        LOG_ERROR("[Loopback] Cannot open capture device {}: {}", device, snd_strerror(err));
        return nullptr;
    }

    snd_pcm_hw_params_t* hw_params;
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(handle, hw_params);

    if ((err = snd_pcm_hw_params_set_access(handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) <
            0 ||
        (err = snd_pcm_hw_params_set_format(handle, hw_params, format)) < 0) {
        LOG_ERROR("[Loopback] Cannot set access/format: {}", snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }

    if ((err = snd_pcm_hw_params_set_channels(handle, hw_params, channels)) < 0) {
        LOG_ERROR("[Loopback] Cannot set channels {}: {}", channels, snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }

    unsigned int rate_near = rate;
    if ((err = snd_pcm_hw_params_set_rate_near(handle, hw_params, &rate_near, nullptr)) < 0) {
        LOG_ERROR("[Loopback] Cannot set rate {}: {}", rate, snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }
    if (rate_near != rate) {
        LOG_ERROR("[Loopback] Requested rate {} not supported (got {})", rate, rate_near);
        snd_pcm_close(handle);
        return nullptr;
    }

    snd_pcm_uframes_t buffer_frames =
        static_cast<snd_pcm_uframes_t>(std::max<uint32_t>(period_frames * 4, period_frames));
    if ((err = snd_pcm_hw_params_set_period_size_near(handle, hw_params, &period_frames, nullptr)) <
        0) {
        LOG_ERROR("[Loopback] Cannot set period size: {}", snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }
    buffer_frames = std::max<snd_pcm_uframes_t>(buffer_frames, period_frames * 2);
    if ((err = snd_pcm_hw_params_set_buffer_size_near(handle, hw_params, &buffer_frames)) < 0) {
        LOG_ERROR("[Loopback] Cannot set buffer size: {}", snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }

    if ((err = snd_pcm_hw_params(handle, hw_params)) < 0) {
        LOG_ERROR("[Loopback] Cannot apply hardware parameters: {}", snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }

    snd_pcm_hw_params_get_period_size(hw_params, &period_frames, nullptr);
    snd_pcm_hw_params_get_buffer_size(hw_params, &buffer_frames);

    if ((err = snd_pcm_prepare(handle)) < 0) {
        LOG_ERROR("[Loopback] Cannot prepare capture device: {}", snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }

    // Ensure non-blocking is effective even if driver flips it.
    snd_pcm_nonblock(handle, 1);

    LOG_INFO(
        "[Loopback] Capture device {} configured ({} Hz, {} ch, period {} frames, buffer {} "
        "frames)",
        device, rate_near, channels, period_frames, buffer_frames);
    return handle;
}

static snd_pcm_t* open_i2s_capture(const std::string& device, snd_pcm_format_t format,
                                   unsigned int requested_rate, unsigned int channels,
                                   snd_pcm_uframes_t& period_frames, unsigned int& actual_rate) {
    const snd_pcm_uframes_t requested_period = period_frames;
    actual_rate = 0;

    auto expand_device_candidates = [](const std::string& configured) {
        std::vector<std::string> candidates;
        candidates.reserve(3);
        candidates.push_back(configured);

        auto add_unique = [&](std::string v) {
            if (v.empty()) {
                return;
            }
            for (const auto& existing : candidates) {
                if (existing == v) {
                    return;
                }
            }
            candidates.push_back(std::move(v));
        };

        auto has_prefix = [&](const char* prefix) { return configured.rfind(prefix, 0) == 0; };

        // ALSA "hw/plughw" allow both:
        // - card,device
        // - card,device,subdevice
        // Field reports show Jetson environments where one form works and the other fails.
        if (has_prefix("hw:") || has_prefix("plughw:")) {
            const int commas =
                static_cast<int>(std::count(configured.begin(), configured.end(), ','));
            if (commas == 1) {
                add_unique(configured + ",0");
            } else if (commas >= 2) {
                const auto lastComma = configured.rfind(',');
                if (lastComma != std::string::npos) {
                    add_unique(configured.substr(0, lastComma));
                }
            }
        }

        return candidates;
    };

    const auto device_candidates = expand_device_candidates(device);

    auto try_open = [&](const std::string& target_device,
                        unsigned int candidate_rate) -> snd_pcm_t* {
        const bool auto_rate = candidate_rate == 0;
        const std::string rate_label = auto_rate ? "auto" : std::to_string(candidate_rate);
        snd_pcm_uframes_t local_period = requested_period;
        snd_pcm_t* handle = nullptr;

        // Use non-blocking mode so shutdown doesn't hang on snd_pcm_readi().
        int err =
            snd_pcm_open(&handle, target_device.c_str(), SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK);
        if (err < 0) {
            LOG_ERROR("[I2S] Cannot open capture device {}: {}", target_device, snd_strerror(err));
            return nullptr;
        }

        snd_pcm_hw_params_t* hw_params;
        snd_pcm_hw_params_alloca(&hw_params);
        snd_pcm_hw_params_any(handle, hw_params);

        if ((err = snd_pcm_hw_params_set_access(handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) <
                0 ||
            (err = snd_pcm_hw_params_set_format(handle, hw_params, format)) < 0) {
            LOG_ERROR("[I2S] Cannot set access/format: {}", snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }

        if ((err = snd_pcm_hw_params_set_channels(handle, hw_params, channels)) < 0) {
            LOG_ERROR("[I2S] Cannot set channels {}: {}", channels, snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }

        unsigned int rate_near = candidate_rate;
        if (!auto_rate) {
            if ((err = snd_pcm_hw_params_set_rate_near(handle, hw_params, &rate_near, nullptr)) <
                0) {
                LOG_ERROR("[I2S] Cannot set rate {}: {}", candidate_rate, snd_strerror(err));
                snd_pcm_close(handle);
                return nullptr;
            }
            if (rate_near != candidate_rate) {
                LOG_ERROR("[I2S] Requested rate {} not supported (got {})", candidate_rate,
                          rate_near);
                snd_pcm_close(handle);
                return nullptr;
            }
        }

        snd_pcm_uframes_t buffer_frames =
            static_cast<snd_pcm_uframes_t>(std::max<uint32_t>(local_period * 4, local_period));
        if ((err = snd_pcm_hw_params_set_period_size_near(handle, hw_params, &local_period,
                                                          nullptr)) < 0) {
            LOG_ERROR("[I2S] Cannot set period size: {}", snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }
        buffer_frames = std::max<snd_pcm_uframes_t>(buffer_frames, local_period * 2);
        if ((err = snd_pcm_hw_params_set_buffer_size_near(handle, hw_params, &buffer_frames)) < 0) {
            LOG_ERROR("[I2S] Cannot set buffer size: {}", snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }

        if ((err = snd_pcm_hw_params(handle, hw_params)) < 0) {
            LOG_ERROR(
                "[I2S] Cannot apply hardware parameters (rate {}, ch {}, fmt {}, "
                "period {} frames, buffer {} frames): {}",
                rate_label, channels, snd_pcm_format_name(format), local_period, buffer_frames,
                snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }

        snd_pcm_hw_params_get_period_size(hw_params, &local_period, nullptr);
        snd_pcm_hw_params_get_buffer_size(hw_params, &buffer_frames);
        unsigned int effective_rate = candidate_rate;
        snd_pcm_hw_params_get_rate(hw_params, &effective_rate, nullptr);
        if (!auto_rate && effective_rate != candidate_rate) {
            LOG_ERROR("[I2S] Negotiated rate {} differs from requested {}", effective_rate,
                      candidate_rate);
            snd_pcm_close(handle);
            return nullptr;
        }

        if ((err = snd_pcm_prepare(handle)) < 0) {
            LOG_ERROR("[I2S] Cannot prepare capture device: {}", snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }

        // Ensure non-blocking is effective even if driver flips it.
        snd_pcm_nonblock(handle, 1);

        period_frames = local_period;
        actual_rate = effective_rate;

        LOG_INFO(
            "[I2S] Capture device {} configured ({} Hz, {} ch, fmt={}, period {} frames, "
            "buffer {} frames)",
            target_device, actual_rate, channels, snd_pcm_format_name(format), period_frames,
            buffer_frames);

        return handle;
    };

    std::vector<unsigned int> rate_candidates;
    if (requested_rate == 0) {
        rate_candidates = {0u, 44100u, 48000u};
    } else {
        rate_candidates = {requested_rate};
    }

    for (const auto& candidate_device : device_candidates) {
        for (unsigned int candidate_rate : rate_candidates) {
            snd_pcm_t* handle = try_open(candidate_device, candidate_rate);
            if (handle) {
                if (candidate_device != device) {
                    LOG_WARN("[I2S] Using ALSA device fallback '{}' (configured was '{}')",
                             candidate_device, device);
                }
                return handle;
            }
        }
    }

    return nullptr;
}

static bool convert_pcm_to_float(const void* src, snd_pcm_format_t format, size_t frames,
                                 unsigned int channels, std::vector<float>& dst) {
    const size_t samples = frames * static_cast<size_t>(channels);
    dst.resize(samples);

    if (format == SND_PCM_FORMAT_S16_LE) {
        const auto* in = static_cast<const int16_t*>(src);
        constexpr float scale = 1.0f / 32768.0f;
        for (size_t i = 0; i < samples; ++i) {
            dst[i] = static_cast<float>(in[i]) * scale;
        }
        return true;
    }

    if (format == SND_PCM_FORMAT_S32_LE) {
        const auto* in = static_cast<const int32_t*>(src);
        constexpr float scale = 1.0f / 2147483648.0f;
        for (size_t i = 0; i < samples; ++i) {
            dst[i] = static_cast<float>(in[i]) * scale;
        }
        return true;
    }

    if (format == SND_PCM_FORMAT_S24_3LE) {
        const auto* in = static_cast<const uint8_t*>(src);
        constexpr float scale = 1.0f / 8388608.0f;
        for (size_t i = 0; i < samples; ++i) {
            size_t idx = i * 3;
            int32_t value = static_cast<int32_t>(in[idx]) |
                            (static_cast<int32_t>(in[idx + 1]) << 8) |
                            (static_cast<int32_t>(in[idx + 2]) << 16);
            if (value & 0x00800000) {
                value |= 0xFF000000;  // sign extend 24-bit
            }
            dst[i] = static_cast<float>(value) * scale;
        }
        return true;
    }

    return false;
}

static void wait_for_capture_ready(snd_pcm_t* handle) {
    if (!handle) {
        return;
    }
    // snd_pcm_wait is safe for non-blocking handles and prevents a tight EAGAIN spin.
    int ret = snd_pcm_wait(handle, 100);
    if (ret < 0) {
        // Keep it quiet; transient errors may happen on device disconnect.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

static void i2s_capture_thread(const std::string& device, snd_pcm_format_t format,
                               unsigned int requested_rate, unsigned int channels,
                               snd_pcm_uframes_t period_frames) {
    elevate_realtime_priority("I2S capture");

    if (channels != static_cast<unsigned int>(CHANNELS)) {
        LOG_ERROR("[I2S] Unsupported channel count {} (expected {})", channels, CHANNELS);
        return;
    }

    const snd_pcm_uframes_t configured_period_frames = period_frames;
    std::vector<int16_t> buffer16;
    std::vector<int32_t> buffer32;
    std::vector<uint8_t> buffer24;
    std::vector<float> floatBuffer;

    g_state.i2s.captureRunning.store(true, std::memory_order_release);

    while (g_state.flags.running.load(std::memory_order_acquire)) {
        snd_pcm_uframes_t negotiated_period = configured_period_frames;
        unsigned int actual_rate = requested_rate;
        snd_pcm_t* handle = open_i2s_capture(device, format, requested_rate, channels,
                                             negotiated_period, actual_rate);
        {
            std::lock_guard<std::mutex> lock(g_state.i2s.handleMutex);
            g_state.i2s.handle = handle;
        }

        if (!handle) {
            // Device not ready / unplugged. Retry with backoff.
            g_state.i2s.captureReady.store(false, std::memory_order_release);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        auto resizeBuffers = [&](snd_pcm_uframes_t frames) {
            if (format == SND_PCM_FORMAT_S16_LE) {
                buffer16.assign(frames * channels, 0);
            } else if (format == SND_PCM_FORMAT_S32_LE) {
                buffer32.assign(frames * channels, 0);
            } else if (format == SND_PCM_FORMAT_S24_3LE) {
                buffer24.assign(static_cast<size_t>(frames) * channels * 3, 0);
            }
        };
        resizeBuffers(negotiated_period);

        // If the hardware rate differs from current engine input rate, schedule follow-up.
        // (Actual reinit is handled in main loop; network-driven follow is #824.)
        if (actual_rate == 44100 || actual_rate == 48000) {
            int current = g_state.rates.inputSampleRate;
            if (current != static_cast<int>(actual_rate)) {
                LOG_WARN("[I2S] Detected input rate {} Hz (engine {} Hz). Scheduling rate follow.",
                         actual_rate, current);
                g_state.rates.pendingRateChange.store(static_cast<int>(actual_rate),
                                                      std::memory_order_release);
            }
        }

        g_state.i2s.captureReady.store(true, std::memory_order_release);

        bool needReopen = false;
        while (g_state.flags.running.load(std::memory_order_acquire)) {
            // Important: For I2S capture in external-clock/slave scenarios, we must keep draining
            // the ALSA capture buffer. Applying backpressure here can stall reads long enough to
            // trigger capture overruns (XRUN), resulting in periodic "burst + noise" artifacts.
            //
            // If downstream is behind, we prefer to drop at the daemon-level playback queue
            // (PlaybackBufferManager::enqueue enforces capacity) rather than blocking capture.

            void* rawBuffer = nullptr;
            if (format == SND_PCM_FORMAT_S16_LE) {
                rawBuffer = buffer16.data();
            } else if (format == SND_PCM_FORMAT_S32_LE) {
                rawBuffer = buffer32.data();
            } else if (format == SND_PCM_FORMAT_S24_3LE) {
                rawBuffer = buffer24.data();
            } else {
                needReopen = false;
                break;
            }

            snd_pcm_sframes_t frames = snd_pcm_readi(handle, rawBuffer, negotiated_period);
            if (frames == -EAGAIN) {
                wait_for_capture_ready(handle);
                continue;
            }
            if (frames == -EPIPE) {
                LOG_WARN("[I2S] XRUN detected, recovering");
                // Use recover() to handle driver-specific XRUN recovery requirements.
                if (snd_pcm_recover(handle, frames, 1) < 0) {
                    snd_pcm_prepare(handle);
                }
                continue;
            }
            if (frames < 0) {
                LOG_WARN("[I2S] Read error: {}", snd_strerror(frames));
                if (snd_pcm_recover(handle, frames, 1) < 0) {
                    // Treat as a device fault; close and attempt reopen.
                    LOG_ERROR("[I2S] Recover failed, attempting reopen");
                    needReopen = true;
                    break;
                }
                continue;
            }
            if (frames == 0) {
                wait_for_capture_ready(handle);
                continue;
            }

            if (!convert_pcm_to_float(rawBuffer, format, static_cast<size_t>(frames), channels,
                                      floatBuffer)) {
                LOG_ERROR("[I2S] Unsupported format during conversion");
                needReopen = true;
                break;
            }

            if (g_state.audioPipeline) {
                g_state.audioPipeline->process(floatBuffer.data(), static_cast<uint32_t>(frames));
            }
        }

        snd_pcm_drop(handle);
        snd_pcm_close(handle);
        {
            std::lock_guard<std::mutex> lock(g_state.i2s.handleMutex);
            g_state.i2s.handle = nullptr;
        }
        g_state.i2s.captureReady.store(false, std::memory_order_release);

        if (!needReopen) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    g_state.i2s.captureRunning.store(false, std::memory_order_release);
    g_state.i2s.captureReady.store(false, std::memory_order_release);
    LOG_INFO("[I2S] Capture thread terminated");
}

static void loopback_capture_thread(const std::string& device, snd_pcm_format_t format,
                                    unsigned int rate, unsigned int channels,
                                    snd_pcm_uframes_t period_frames) {
    if (channels != static_cast<unsigned int>(CHANNELS)) {
        LOG_ERROR("[Loopback] Unsupported channel count {} (expected {})", channels, CHANNELS);
        return;
    }

    snd_pcm_uframes_t negotiated_period = period_frames;
    snd_pcm_t* handle = open_loopback_capture(device, format, rate, channels, negotiated_period);
    {
        std::lock_guard<std::mutex> lock(g_state.loopback.handleMutex);
        g_state.loopback.handle = handle;
    }

    if (!handle) {
        return;
    }

    g_state.loopback.captureRunning.store(true, std::memory_order_release);
    g_state.loopback.captureReady.store(true, std::memory_order_release);

    std::vector<int16_t> buffer16;
    std::vector<int32_t> buffer32;
    std::vector<uint8_t> buffer24;
    std::vector<float> floatBuffer;

    auto resizeBuffers = [&](snd_pcm_uframes_t frames) {
        if (format == SND_PCM_FORMAT_S16_LE) {
            buffer16.assign(frames * channels, 0);
        } else if (format == SND_PCM_FORMAT_S32_LE) {
            buffer32.assign(frames * channels, 0);
        } else if (format == SND_PCM_FORMAT_S24_3LE) {
            buffer24.assign(static_cast<size_t>(frames) * channels * 3, 0);
        }
    };
    resizeBuffers(negotiated_period);

    while (g_state.flags.running.load(std::memory_order_acquire)) {
        // Apply backpressure before pulling more input.
        // This prevents software loopback / bursty upstream from running ahead of ALSA playback.
        playback_buffer().throttleProducerIfFull(g_state.flags.running, []() {
            return g_state.rates.currentOutputRate.load(std::memory_order_acquire);
        });

        void* rawBuffer = nullptr;
        if (format == SND_PCM_FORMAT_S16_LE) {
            rawBuffer = buffer16.data();
        } else if (format == SND_PCM_FORMAT_S32_LE) {
            rawBuffer = buffer32.data();
        } else if (format == SND_PCM_FORMAT_S24_3LE) {
            rawBuffer = buffer24.data();
        } else {
            break;
        }

        snd_pcm_sframes_t frames = snd_pcm_readi(handle, rawBuffer, negotiated_period);
        if (frames == -EAGAIN) {
            wait_for_capture_ready(handle);
            continue;
        }
        if (frames == -EPIPE) {
            LOG_WARN("[Loopback] XRUN detected, recovering");
            snd_pcm_prepare(handle);
            continue;
        }
        if (frames < 0) {
            LOG_WARN("[Loopback] Read error: {}", snd_strerror(frames));
            if (snd_pcm_recover(handle, frames, 1) < 0) {
                LOG_ERROR("[Loopback] Recover failed, restarting read loop");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            continue;
        }
        if (frames == 0) {
            wait_for_capture_ready(handle);
            continue;
        }

        const void* src = rawBuffer;
        if (!convert_pcm_to_float(src, format, static_cast<size_t>(frames), channels,
                                  floatBuffer)) {
            LOG_ERROR("[Loopback] Unsupported format during conversion");
            break;
        }

        if (g_state.audioPipeline) {
            g_state.audioPipeline->process(floatBuffer.data(), static_cast<uint32_t>(frames));
        }
    }

    snd_pcm_drop(handle);
    snd_pcm_close(handle);
    {
        std::lock_guard<std::mutex> lock(g_state.loopback.handleMutex);
        g_state.loopback.handle = nullptr;
    }
    g_state.loopback.captureRunning.store(false, std::memory_order_release);
    g_state.loopback.captureReady.store(false, std::memory_order_release);
    LOG_INFO("[Loopback] Capture thread terminated");
}

// ALSA output thread (705.6kHz direct to DAC)
void alsa_output_thread() {
    elevate_realtime_priority("ALSA output");

    daemon_output::AlsaPcmController pcmController(daemon_output::AlsaPcmControllerDependencies{
        .config = &g_state.config,
        .dacManager = g_state.managers.dacManager.get(),
        .streamingCacheManager = g_state.managers.streamingCacheManager.get(),
        .fallbackManager = g_state.fallbackManager,
        .running = &g_state.flags.running,
        .outputReady = &g_state.flags.outputReady,
        .currentOutputRate =
            []() { return g_state.rates.currentOutputRate.load(std::memory_order_acquire); },
    });

    if (!pcmController.openSelected()) {
        return;
    }

    auto period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
    if (period_size == 0) {
        period_size = static_cast<snd_pcm_uframes_t>(
            (g_state.config.periodSize > 0) ? g_state.config.periodSize : 32768);
    }
    std::vector<int32_t> interleaved_buffer(period_size * CHANNELS);
    std::vector<float> float_buffer(period_size * CHANNELS);  // for soft mute processing
    auto& bufferManager = playback_buffer();

    // Main playback loop
    while (g_state.flags.running) {
        // Heartbeat check every few hundred loops
        static int alive_counter = 0;
        if (++alive_counter > 200) {  // ~200 iterations ~ a few seconds depending on buffer wait
            alive_counter = 0;
            if (!pcmController.alive()) {
                LOG_EVERY_N(WARN, 5, "[ALSA] PCM disconnected/suspended, attempting reopen...");
                pcmController.close();
                while (g_state.flags.running && !pcmController.openSelected()) {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                }
                period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                if (period_size == 0) {
                    period_size = static_cast<snd_pcm_uframes_t>(
                        (g_state.config.periodSize > 0) ? g_state.config.periodSize : 32768);
                }
                interleaved_buffer.resize(period_size * CHANNELS);
                float_buffer.resize(period_size * CHANNELS);
                continue;
            }
        }

        // Issue #219: Check for pending ALSA reconfiguration (output rate changed)
        if (g_state.rates.alsaReconfigureNeeded.exchange(false, std::memory_order_acquire)) {
            int new_output_rate = g_state.rates.alsaNewOutputRate.load(std::memory_order_acquire);
            if (new_output_rate > 0) {
                LOG_INFO("[Main] Reconfiguring ALSA for new output rate {} Hz", new_output_rate);

                // Reconfigure ALSA with new rate
                if (pcmController.reconfigure(new_output_rate)) {
                    period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                    if (period_size == 0) {
                        period_size = static_cast<snd_pcm_uframes_t>(
                            (g_state.config.periodSize > 0) ? g_state.config.periodSize : 32768);
                    }
                    interleaved_buffer.resize(period_size * CHANNELS);
                    float_buffer.resize(period_size * CHANNELS);

                    // Update soft mute sample rate
                    if (g_state.softMute.controller) {
                        g_state.softMute.controller->setSampleRate(new_output_rate);
                    }

                    LOG_INFO("[Main] ALSA reconfiguration successful");
                } else {
                    // Failed to reconfigure - try to reopen with old rate
                    LOG_ERROR("[Main] ALSA reconfiguration failed, attempting recovery...");
                    int old_rate = g_state.rates.currentOutputRate.load(std::memory_order_acquire);
                    if (!pcmController.reconfigure(old_rate)) {
                        LOG_ERROR("[Main] ALSA recovery failed, waiting for reconnect...");
                    }
                }
            }
        }

        if (auto pendingDevice = g_state.managers.dacManager->consumePendingChange()) {
            std::string nextDevice = *pendingDevice;
            if (!nextDevice.empty() && nextDevice != pcmController.device()) {
                LOG_INFO("[ALSA] Switching output to {}", nextDevice);
                if (!pcmController.switchDevice(nextDevice)) {
                    continue;
                }
                period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                if (period_size == 0) {
                    period_size = static_cast<snd_pcm_uframes_t>(
                        (g_state.config.periodSize > 0) ? g_state.config.periodSize : 32768);
                }
                interleaved_buffer.resize(period_size * CHANNELS);
                float_buffer.resize(period_size * CHANNELS);
            }
        }

        // Wait for GPU processed data (dynamic threshold to avoid underflow with crossfeed)
        std::unique_lock<std::mutex> lock(bufferManager.mutex());
        size_t ready_threshold = get_playback_ready_threshold(static_cast<size_t>(period_size));
        bufferManager.cv().wait_for(
            lock, std::chrono::milliseconds(200), [&bufferManager, ready_threshold] {
                return bufferManager.queuedFramesLocked() >= ready_threshold ||
                       !g_state.flags.running;
            });

        if (!g_state.flags.running) {
            break;
        }

        lock.unlock();

        audio_pipeline::RenderResult renderResult;
        if (g_state.audioPipeline) {
            renderResult = g_state.audioPipeline->renderOutput(static_cast<size_t>(period_size),
                                                               interleaved_buffer, float_buffer,
                                                               g_state.softMute.controller);
        } else {
            renderResult.framesRequested = period_size;
            renderResult.framesRendered = period_size;
            renderResult.wroteSilence = true;
            std::fill(interleaved_buffer.begin(), interleaved_buffer.end(), 0);
        }

        // Wake any producers that are throttling on "space available".
        // renderOutput() advances the output read position under its own lock; notifying here is
        // safe.
        bufferManager.cv().notify_all();

        // Apply pending soft mute parameter restoration once transition completes
        maybe_restore_soft_mute_params();

        // Write to ALSA device
        long frames_written = pcmController.writeInterleaved(interleaved_buffer.data(),
                                                             static_cast<size_t>(period_size));
        if (frames_written < 0) {
            // Device may be gone; attempt reopen
            LOG_EVERY_N(ERROR, 10, "[ALSA] Write error: {}, retrying reopen...",
                        snd_strerror(frames_written));
            pcmController.close();
            while (g_state.flags.running && !pcmController.openSelected()) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
            if (!pcmController.isOpen()) {
                continue;
            }
            period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
            if (period_size == 0) {
                period_size = static_cast<snd_pcm_uframes_t>(
                    (g_state.config.periodSize > 0) ? g_state.config.periodSize : 32768);
            }
            interleaved_buffer.resize(period_size * CHANNELS);
            float_buffer.resize(period_size * CHANNELS);
        }
    }

    // Cleanup
    pcmController.close();
    LOG_INFO("[ALSA] Output thread terminated");
}

int main(int argc, char* argv[]) {
    // Early initialization with stderr output only (before PID lock)
    // This allows logging during PID lock acquisition
    gpu_upsampler::logging::initializeEarly();

    // Acquire PID file lock (prevent multiple instances)
    // Note: At this point, logging outputs to stderr only
    if (!acquire_pid_lock()) {
        return 1;
    }

    // Full initialization from config file (after PID lock acquired)
    // This replaces stderr-only logger with configured logger (file + console)
    gpu_upsampler::logging::initializeFromConfig(CONFIG_FILE_PATH);

    LOG_INFO("========================================");
    LOG_INFO("  GPU Audio Upsampler - ALSA Direct Output");
    LOG_INFO("========================================");
    LOG_INFO("PID: {} (file: {})", getpid(), PID_FILE_PATH);

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

        // Auto-select filter based on sample rate if configured filter doesn't exist
        // but a sample-rate-specific version does
        if (!std::filesystem::exists(g_state.config.filterPath)) {
            // Try to find sample-rate-specific filter
            std::string basePath = g_state.config.filterPath;
            size_t dotPos = basePath.rfind('.');
            if (dotPos != std::string::npos) {
                std::string rateSpecificPath = basePath.substr(0, dotPos) + "_" +
                                               std::to_string(g_state.rates.inputSampleRate) +
                                               basePath.substr(dotPos);
                if (std::filesystem::exists(rateSpecificPath)) {
                    std::cout << "Config: Using sample-rate-specific filter: " << rateSpecificPath
                              << '\n';
                    g_state.config.filterPath = rateSpecificPath;
                }
            }
        }

        // Warn if using 44.1kHz filter with 48kHz input
        if (g_state.rates.inputSampleRate == 48000 &&
            g_state.config.filterPath.find("44100") == std::string::npos &&
            g_state.config.filterPath.find("48000") == std::string::npos) {
            std::cout << "Warning: Using generic filter with 48kHz input. "
                      << "For optimal quality, generate a 48kHz-optimized filter." << '\n';
        }

        // Initialize GPU upsampler with configured values
        std::cout << "Initializing GPU upsampler..." << '\n';
        g_state.upsampler = new ConvolutionEngine::GPUUpsampler();
        g_state.upsampler->setPartitionedConvolutionConfig(g_state.config.partitionedConvolution);

        bool initSuccess = false;
        ConvolutionEngine::RateFamily initialFamily = ConvolutionEngine::RateFamily::RATE_44K;
        if (g_state.config.multiRateEnabled) {
            std::cout << "Multi-rate mode enabled" << '\n';
            std::cout << "  Coefficient directory: " << g_state.config.coefficientDir << '\n';

            if (!std::filesystem::exists(g_state.config.coefficientDir)) {
                std::cerr << "Config error: Coefficient directory not found: "
                          << g_state.config.coefficientDir << '\n';
                delete g_state.upsampler;
                exitCode = 1;
                break;
            }

            initSuccess = g_state.upsampler->initializeMultiRate(g_state.config.coefficientDir,
                                                                 g_state.config.blockSize,
                                                                 g_state.rates.inputSampleRate);

            if (initSuccess) {
                g_state.rates.currentInputRate.store(g_state.upsampler->getInputSampleRate(),
                                                     std::memory_order_release);
                g_state.rates.currentOutputRate.store(g_state.upsampler->getOutputSampleRate(),
                                                      std::memory_order_release);
                g_set_rate_family(
                    ConvolutionEngine::detectRateFamily(g_state.rates.inputSampleRate));
            }
        } else {
            std::cout << "Quad-phase mode enabled" << '\n';

            bool allFilesExist = true;
            for (const auto& path :
                 {g_state.config.filterPath44kMin, g_state.config.filterPath48kMin,
                  g_state.config.filterPath44kLinear, g_state.config.filterPath48kLinear}) {
                if (!std::filesystem::exists(path)) {
                    std::cerr << "Config error: Filter file not found: " << path << '\n';
                    allFilesExist = false;
                }
            }
            if (!allFilesExist) {
                delete g_state.upsampler;
                exitCode = 1;
                break;
            }

            initialFamily = ConvolutionEngine::detectRateFamily(g_state.rates.inputSampleRate);
            if (initialFamily == ConvolutionEngine::RateFamily::RATE_UNKNOWN) {
                initialFamily = ConvolutionEngine::RateFamily::RATE_44K;
            }

            initSuccess = g_state.upsampler->initializeQuadPhase(
                g_state.config.filterPath44kMin, g_state.config.filterPath48kMin,
                g_state.config.filterPath44kLinear, g_state.config.filterPath48kLinear,
                g_state.config.upsampleRatio, g_state.config.blockSize, initialFamily,
                g_state.config.phaseType);
        }

        if (!initSuccess) {
            std::cerr << "Failed to initialize GPU upsampler" << '\n';
            delete g_state.upsampler;
            exitCode = 1;
            break;
        }

        // Check for early abort (signal received during GPU initialization)
        if (!g_state.flags.running) {
            std::cout << "Startup interrupted by signal" << '\n';
            delete g_state.upsampler;
            g_state.upsampler = nullptr;
            break;
        }

        if (g_state.config.multiRateEnabled) {
            std::cout << "GPU upsampler ready (multi-rate mode, " << g_state.config.blockSize
                      << " samples/block)" << '\n';
            std::cout << "  Current input rate: " << g_state.upsampler->getCurrentInputRate()
                      << " Hz" << '\n';
            std::cout << "  Upsample ratio: " << g_state.upsampler->getUpsampleRatio() << "x"
                      << '\n';
            std::cout << "  Output rate: " << g_state.upsampler->getOutputSampleRate() << " Hz"
                      << '\n';
        } else {
            std::cout << "GPU upsampler ready (" << g_state.config.upsampleRatio << "x upsampling, "
                      << g_state.config.blockSize << " samples/block)" << '\n';
        }

        // Set g_state.rates.activeRateFamily and g_state.rates.activePhaseType for headroom
        // tracking
        if (g_state.config.multiRateEnabled) {
            // Rate family already set during initializeMultiRate()
            // g_state.rates.activeRateFamily is set via g_set_rate_family() above
        } else {
            g_state.rates.activeRateFamily = initialFamily;
        }

        g_state.rates.activePhaseType = g_state.config.phaseType;
        publish_filter_switch_event(current_filter_path(), g_state.rates.activePhaseType, true);

        std::cout << "Input sample rate: " << g_state.upsampler->getInputSampleRate() << " Hz -> "
                  << g_state.upsampler->getOutputSampleRate() << " Hz output" << '\n';
        if (!g_state.config.multiRateEnabled) {
            std::cout << "Phase type: " << phaseTypeToString(g_state.config.phaseType) << '\n';
        }

        // Log latency warning for linear phase
        if (g_state.config.phaseType == PhaseType::Linear) {
            double latencySec = g_state.upsampler->getLatencySeconds();
            std::cout << "  WARNING: Linear phase latency: " << latencySec << " seconds ("
                      << g_state.upsampler->getLatencySamples() << " samples)" << '\n';
        }

        // Initialize streaming mode to preserve overlap buffers across input callbacks
        if (!g_state.upsampler->initializeStreaming()) {
            std::cerr << "Failed to initialize streaming mode" << '\n';
            delete g_state.upsampler;
            exitCode = 1;
            break;
        }
        // Issue #899: avoid per-period host blocking on GPU completion in steady-state playback.
        // Keep legacy semantics (blocking) for offline/tests unless explicitly enabled here.
        g_state.upsampler->setStreamingNonBlocking(true);
        PartitionRuntime::applyPartitionPolicy(partitionRequest, *g_state.upsampler, g_state.config,
                                               "ALSA");

        // Check for early abort
        if (!g_state.flags.running) {
            std::cout << "Startup interrupted by signal" << '\n';
            delete g_state.upsampler;
            g_state.upsampler = nullptr;
            break;
        }

        // Apply EQ profile if enabled
        if (g_state.config.eqEnabled && !g_state.config.eqProfilePath.empty()) {
            std::cout << "Loading EQ profile: " << g_state.config.eqProfilePath << '\n';
            EQ::EqProfile eqProfile;
            if (EQ::parseEqFile(g_state.config.eqProfilePath, eqProfile)) {
                std::cout << "  EQ: " << eqProfile.name << " (" << eqProfile.bands.size()
                          << " bands, preamp " << eqProfile.preampDb << " dB)" << '\n';

                // Compute EQ magnitude response and apply with minimum phase reconstruction
                size_t filterFftSize = g_state.upsampler->getFilterFftSize();  // N/2+1 (R2C output)
                size_t fullFftSize = g_state.upsampler->getFullFftSize();      // N (full FFT)
                double outputSampleRate = static_cast<double>(g_state.rates.inputSampleRate) *
                                          g_state.config.upsampleRatio;
                auto eqMagnitude = EQ::computeEqMagnitudeForFft(filterFftSize, fullFftSize,
                                                                outputSampleRate, eqProfile);
                double eqMax = 0.0;
                double eqMin = std::numeric_limits<double>::infinity();
                for (double v : eqMagnitude) {
                    eqMax = std::max(eqMax, v);
                    eqMin = std::min(eqMin, v);
                }
                std::cout << "  EQ magnitude stats: max=" << eqMax << " ("
                          << 20.0 * std::log10(std::max(eqMax, 1e-30)) << " dB), min=" << eqMin
                          << '\n';

                if (g_state.upsampler->applyEqMagnitude(eqMagnitude)) {
                    // Log message depends on phase type (already logged by applyEqMagnitude)
                } else {
                    std::cerr << "  EQ: Failed to apply frequency response" << '\n';
                }
            } else {
                std::cerr << "  EQ: Failed to parse profile: " << g_state.config.eqProfilePath
                          << '\n';
            }
        }

        // Pre-allocate streaming input buffers (avoid RT reallocations)
        size_t buffer_capacity =
            compute_stream_buffer_capacity(g_state.upsampler->getStreamValidInputPerBlock());
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

        if (!g_state.config.partitionedConvolution.enabled) {
            // Initialize HRTF processor for crossfeed (optional feature)
            // Crossfeed is disabled by default until enabled via ZeroMQ command
            std::string hrtfDir = "data/crossfeed/hrtf";
            if (std::filesystem::exists(hrtfDir)) {
                std::cout << "Initializing HRTF processor for crossfeed..." << '\n';
                g_state.crossfeed.processor = new ConvolutionEngine::FourChannelFIR();

                // Determine rate family based on input sample rate
                ConvolutionEngine::RateFamily rateFamily =
                    ConvolutionEngine::detectRateFamily(g_state.rates.inputSampleRate);
                if (rateFamily == ConvolutionEngine::RateFamily::RATE_UNKNOWN) {
                    rateFamily = ConvolutionEngine::RateFamily::RATE_44K;
                }
                ConvolutionEngine::HeadSize initialHeadSize =
                    ConvolutionEngine::stringToHeadSize(g_state.config.crossfeed.headSize);

                if (g_state.crossfeed.processor->initialize(hrtfDir, g_state.config.blockSize,
                                                            initialHeadSize, rateFamily)) {
                    if (g_state.crossfeed.processor->initializeStreaming()) {
                        std::cout << "  HRTF processor ready (head size: "
                                  << ConvolutionEngine::headSizeToString(initialHeadSize)
                                  << ", rate family: "
                                  << (rateFamily == ConvolutionEngine::RateFamily::RATE_44K ? "44k"
                                                                                            : "48k")
                                  << ")" << '\n';

                        // Pre-allocate crossfeed streaming buffers
                        size_t cf_buffer_capacity = compute_stream_buffer_capacity(
                            g_state.crossfeed.processor->getStreamValidInputPerBlock());
                        g_state.crossfeed.cfStreamInputLeft.resize(cf_buffer_capacity, 0.0f);
                        g_state.crossfeed.cfStreamInputRight.resize(cf_buffer_capacity, 0.0f);
                        g_state.crossfeed.cfStreamAccumulatedLeft = 0;
                        g_state.crossfeed.cfStreamAccumulatedRight = 0;
                        size_t cf_output_capacity =
                            std::max(cf_buffer_capacity,
                                     g_state.crossfeed.processor->getValidOutputPerBlock());
                        g_state.crossfeed.cfOutputLeft.reserve(cf_output_capacity);
                        g_state.crossfeed.cfOutputRight.reserve(cf_output_capacity);
                        g_state.crossfeed.cfOutputBufferLeft.reserve(cf_output_capacity);
                        g_state.crossfeed.cfOutputBufferRight.reserve(cf_output_capacity);
                        std::cout << "  Crossfeed buffer capacity: " << cf_buffer_capacity
                                  << " samples" << '\n';

                        // Crossfeed is initialized but disabled by default
                        g_state.crossfeed.enabled.store(false);
                        g_state.crossfeed.processor->setEnabled(false);
                        std::cout << "  Crossfeed: initialized (disabled by default)" << '\n';
                    } else {
                        std::cerr << "  HRTF: Failed to initialize streaming mode" << '\n';
                        delete g_state.crossfeed.processor;
                        g_state.crossfeed.processor = nullptr;
                    }
                } else {
                    std::cerr << "  HRTF: Failed to initialize processor" << '\n';
                    std::cerr << "  Hint: Run 'uv run python scripts/filters/generate_hrtf.py' to "
                                 "generate HRTF "
                                 "filters"
                              << '\n';
                    delete g_state.crossfeed.processor;
                    g_state.crossfeed.processor = nullptr;
                }
            } else {
                std::cout << "HRTF directory not found (" << hrtfDir
                          << "), crossfeed feature disabled" << '\n';
                std::cout << "  Hint: Run 'uv run python scripts/filters/generate_hrtf.py' to "
                             "generate HRTF "
                             "filters"
                          << '\n';
            }
        } else {
            std::cout << "[Partition] Crossfeed initialization skipped (low-latency mode)" << '\n';
        }

        std::cout << '\n';

        if (!g_state.audioPipeline && g_state.upsampler) {
            audio_pipeline::Dependencies pipelineDeps{};
            pipelineDeps.config = &g_state.config;
            pipelineDeps.upsampler.available = true;
            pipelineDeps.upsampler.streamLeft = g_state.upsampler->streamLeft_;
            pipelineDeps.upsampler.streamRight = g_state.upsampler->streamRight_;
            pipelineDeps.output.outputGain = &g_state.gains.output;
            pipelineDeps.output.limiterGain = &g_state.gains.limiter;
            pipelineDeps.output.effectiveGain = &g_state.gains.effective;
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
            pipelineDeps.buffer.playbackBuffer = &playback_buffer();
            pipelineDeps.maxOutputBufferFrames = []() { return get_max_output_buffer_frames(); };
            pipelineDeps.currentOutputRate = []() {
                return g_state.rates.currentOutputRate.load(std::memory_order_acquire);
            };
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
        controlDeps.bufferCapacityFrames = []() { return get_max_output_buffer_frames(); };
        controlDeps.applySoftMuteForFilterSwitch = [](std::function<bool()> fn) {
            applySoftMuteForFilterSwitch(std::move(fn));
        };
        controlDeps.resetStreamingCachesForSwitch = []() {
            return reset_streaming_caches_for_switch();
        };
        controlDeps.refreshHeadroom = [](const std::string& reason) {
            refresh_current_headroom(reason);
        };
        controlDeps.reinitializeStreamingForLegacyMode = []() {
            return reinitialize_streaming_for_legacy_mode();
        };
        controlDeps.setPreferredOutputDevice = [](AppConfig& cfg, const std::string& device) {
            set_preferred_output_device(cfg, device);
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
        std::thread alsa_thread(alsa_output_thread);

        std::thread loopback_thread;
        std::thread i2s_thread;
        bool startupFailed = false;

        auto failStartup = [&](const std::string& reason) {
            LOG_ERROR("Startup failed: {}", reason);
            exitCode = 1;
            startupFailed = true;
            g_state.flags.running = false;
            playback_buffer().cv().notify_all();
        };

        if (g_state.config.i2s.enabled && g_state.config.loopback.enabled) {
            failStartup("Config error: i2s.enabled and loopback.enabled cannot both be true");
        }

        if (!startupFailed && g_state.config.i2s.enabled) {
            if (!validate_i2s_config(g_state.config)) {
                failStartup("Invalid I2S config");
            } else {
                snd_pcm_format_t i2s_format = parse_i2s_format(g_state.config.i2s.format);
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
                i2s_thread =
                    std::thread(i2s_capture_thread, g_state.config.i2s.device, i2s_format, i2s_rate,
                                g_state.config.i2s.channels,
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
            if (!validate_loopback_config(g_state.config)) {
                failStartup("Invalid loopback config");
            } else {
                snd_pcm_format_t lb_format = parse_loopback_format(g_state.config.loopback.format);
                std::cout << "Starting loopback capture thread (" << g_state.config.loopback.device
                          << ", fmt=" << g_state.config.loopback.format
                          << ", rate=" << g_state.config.loopback.sampleRate
                          << ", period=" << g_state.config.loopback.periodFrames << ")" << '\n';
                loopback_thread = std::thread(
                    loopback_capture_thread, g_state.config.loopback.device, lb_format,
                    g_state.config.loopback.sampleRate, g_state.config.loopback.channels,
                    static_cast<snd_pcm_uframes_t>(g_state.config.loopback.periodFrames));

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
                        (void)handle_rate_switch(pending);
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
        playback_buffer().cv().notify_all();
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
        delete g_state.crossfeed.processor;
        g_state.crossfeed.processor = nullptr;
        g_state.crossfeed.enabled.store(false);
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

    // Release PID lock and remove file on clean exit
    release_pid_lock();
    std::cout << "Goodbye!" << '\n';
    return exitCode;
}
