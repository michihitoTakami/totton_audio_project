#include "daemon/app/app.h"

#include "audio/audio_utils.h"
#include "audio/fallback_manager.h"
#include "audio/soft_mute.h"
#include "convolution_engine.h"
#include "core/config_loader.h"
#include "core/daemon_constants.h"
#include "core/partition_runtime_utils.h"
#include "daemon/api/dependencies.h"
#include "daemon/api/events.h"
#include "daemon/app/runtime_state.h"
#include "daemon/audio/crossfeed_manager.h"
#include "daemon/audio/upsampler_builder.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/audio_pipeline/filter_manager.h"
#include "daemon/audio_pipeline/headroom_controller.h"
#include "daemon/audio_pipeline/rate_switcher.h"
#include "daemon/audio_pipeline/soft_mute_runner.h"
#include "daemon/audio_pipeline/streaming_cache_manager.h"
#include "daemon/control/control_plane.h"
#include "daemon/control/handlers/handler_registry.h"
#include "daemon/input/loopback_capture.h"
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

namespace daemon_app {

// NOTE: Avoid easy bloat; keep components modular whenever possible.

App::App() : state_(std::make_unique<RuntimeState>()) {}

App::~App() = default;

static void enforce_phase_partition_constraints(AppConfig& config) {
    if (config.partitionedConvolution.enabled && config.phaseType == PhaseType::Linear) {
        std::cout << "[Partition] Linear phase is incompatible with low-latency mode. "
                  << "Switching to minimum phase." << '\n';
        config.phaseType = PhaseType::Minimum;
    }
}

// Stats file path (JSON format for Web API)
constexpr const char* STATS_FILE_PATH = "/tmp/gpu_upsampler_stats.json";

static runtime_stats::Dependencies build_runtime_stats_dependencies(RuntimeState& state);

// Default configuration values (using common constants)
using namespace DaemonConstants;
using StreamFloatVector = ConvolutionEngine::StreamFloatVector;
constexpr const char* DEFAULT_ALSA_DEVICE = "hw:USB";
constexpr const char* DEFAULT_LOOPBACK_DEVICE = "hw:Loopback,1,0";
constexpr uint32_t DEFAULT_LOOPBACK_PERIOD_FRAMES = 1024;
constexpr const char* DEFAULT_FILTER_PATH = "data/coefficients/filter_44k_16x_2m_min_phase.bin";

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

inline ConvolutionEngine::RateFamily get_rate_family(const RuntimeState& state) {
    return static_cast<ConvolutionEngine::RateFamily>(
        state.rates.currentRateFamilyInt.load(std::memory_order_acquire));
}

inline void set_rate_family(RuntimeState& state, ConvolutionEngine::RateFamily family) {
    state.rates.currentRateFamilyInt.store(static_cast<int>(family), std::memory_order_release);
}

static size_t get_max_output_buffer_frames(const RuntimeState& state) {
    using namespace DaemonConstants;
    auto seconds = static_cast<double>(MAX_OUTPUT_BUFFER_SECONDS);
    if (seconds <= 0.0) {
        return DEFAULT_MAX_OUTPUT_BUFFER_FRAMES;
    }

    int outputRate = state.rates.currentOutputRate.load(std::memory_order_acquire);
    if (outputRate <= 0) {
        outputRate = DEFAULT_OUTPUT_SAMPLE_RATE;
    }

    double frames = seconds * static_cast<double>(outputRate);
    if (frames <= 0.0) {
        return DEFAULT_MAX_OUTPUT_BUFFER_FRAMES;
    }
    return static_cast<size_t>(frames);
}

static size_t compute_stream_buffer_capacity(const RuntimeState& state,
                                             size_t streamValidInputPerBlock) {
    using namespace DaemonConstants;
    size_t frames = static_cast<size_t>(DEFAULT_BLOCK_SIZE);
    if (state.config.blockSize > 0) {
        frames = std::max(frames, static_cast<size_t>(state.config.blockSize));
    }
    if (state.config.periodSize > 0) {
        frames = std::max(frames, static_cast<size_t>(state.config.periodSize));
    }
    if (state.config.loopback.periodFrames > 0) {
        frames = std::max(frames, static_cast<size_t>(state.config.loopback.periodFrames));
    }
    frames = std::max(frames, streamValidInputPerBlock);
    // 2x safety margin for bursty upstream (no reallocation in RT path)
    return frames * 2;
}

// Pending rate change (set by input event handlers, processed in main loop)
// Value: 0 = no change pending, >0 = detected input sample rate

static void maybe_restore_soft_mute_params(RuntimeState& state) {
    if (!state.softMute.controller) {
        return;
    }
    if (!state.softMute.restorePending.load(std::memory_order_acquire)) {
        return;
    }
    SoftMute::MuteState st = state.softMute.controller->getState();
    if (st != SoftMute::MuteState::PLAYING && st != SoftMute::MuteState::MUTED) {
        return;
    }

    std::lock_guard<std::mutex> lock(state.softMute.opMutex);
    if (!state.softMute.controller) {
        return;
    }
    int fadeMs = state.softMute.restoreFadeMs.load(std::memory_order_relaxed);
    int sr = state.softMute.restoreSampleRate.load(std::memory_order_relaxed);
    state.softMute.controller->setFadeDuration(fadeMs);
    state.softMute.controller->setSampleRate(sr);
    state.softMute.restorePending.store(false, std::memory_order_release);
}

// Helper function for soft mute during filter switching (Issue #266)
// Fade-out: 1.5 seconds, perform filter switch, fade-in: 1.5 seconds
//
// Thread safety & responsiveness:
// - Called from ZeroMQ command thread, guarded by a mutex to serialize parameter updates
// - Non-blocking: start fade-out, perform switch, then trigger fade-in with minimal wait
// - Original fade parameters are restored in the audio thread once the transition settles

static void applySoftMuteForFilterSwitch(RuntimeState& state,
                                         std::function<bool()> filterSwitchFunc) {
    using namespace DaemonConstants;

    if (!state.softMute.controller) {
        // If soft mute not initialized, perform switch without mute
        filterSwitchFunc();
        return;
    }

    std::lock_guard<std::mutex> lock(state.softMute.opMutex);

    // Cancel any stale pending restore (new switch supersedes)
    state.softMute.restorePending.store(false, std::memory_order_release);

    // Save current fade duration for restoration
    int originalFadeDuration = state.softMute.controller->getFadeDuration();
    int outputSampleRate = state.softMute.controller->getSampleRate();

    // Update fade duration for filter switching
    // Note: This is called from command thread, but audio thread may be processing.
    // The fade calculation will use the new duration from the next audio frame.
    state.softMute.controller->setFadeDuration(FILTER_SWITCH_FADE_MS);
    state.softMute.controller->setSampleRate(outputSampleRate);

    std::cout << "[Filter Switch] Starting fade-out (" << (FILTER_SWITCH_FADE_MS / 1000.0)
              << "s)..." << '\n';
    state.softMute.controller->startFadeOut();

    // Wait until near-silent (or timeout) before switching to avoid audible glitches.
    // NOTE: This runs on the command thread. Use a bounded timeout to avoid hanging the UI
    // if the audio thread is not advancing (e.g., output not running).
    auto fade_start = std::chrono::steady_clock::now();
    const auto fade_timeout = std::chrono::milliseconds(FILTER_SWITCH_FADE_TIMEOUT_MS);
    while (true) {
        SoftMute::MuteState st = state.softMute.controller->getState();
        float gain = state.softMute.controller->getCurrentGain();
        if (st == SoftMute::MuteState::MUTED || gain <= 0.001f) {
            break;
        }
        if (std::chrono::steady_clock::now() - fade_start > fade_timeout) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Perform filter switch while fade-out is progressing
    bool pauseOk = true;
    if (state.audioPipeline) {
        state.audioPipeline->requestRtPause();
        pauseOk = state.audioPipeline->waitForRtQuiescent(std::chrono::milliseconds(500));
        if (!pauseOk) {
            LOG_ERROR("[Filter Switch] RT pause handshake timed out (aborting switch)");
        }
    }

    bool switch_success = false;
    if (pauseOk) {
        switch_success = filterSwitchFunc();
    }

    if (state.audioPipeline) {
        state.audioPipeline->resumeRtPause();
    }

    if (switch_success) {
        // Start fade-in after filter switch
        std::cout << "[Filter Switch] Starting fade-in (" << (FILTER_SWITCH_FADE_MS / 1000.0)
                  << "s)..." << '\n';
        state.softMute.controller->startFadeIn();

        // Mark pending restoration to be applied once transition completes
        state.softMute.restoreFadeMs.store(originalFadeDuration, std::memory_order_relaxed);
        state.softMute.restoreSampleRate.store(outputSampleRate, std::memory_order_relaxed);
        state.softMute.restorePending.store(true, std::memory_order_release);
    } else {
        // If switch failed, restore original state immediately
        std::cerr << "[Filter Switch] Switch failed, restoring audio state" << '\n';
        state.softMute.controller->setPlaying();
        state.softMute.controller->setFadeDuration(originalFadeDuration);
        state.softMute.controller->setSampleRate(outputSampleRate);
    }
}

// Audio buffer for thread communication (managed component)
static daemon_output::PlaybackBufferManager& playback_buffer(RuntimeState& state) {
    if (!state.playback.buffer) {
        state.playback.buffer = std::make_unique<daemon_output::PlaybackBufferManager>(
            [&state]() { return get_max_output_buffer_frames(state); });
    }
    return *state.playback.buffer;
}

static audio_pipeline::HeadroomControllerDependencies make_headroom_dependencies(
    RuntimeState& state, daemon_core::api::EventDispatcher* dispatcher) {
    audio_pipeline::HeadroomControllerDependencies deps{};
    deps.dispatcher = dispatcher;
    deps.config = &state.config;
    deps.headroomCache = &state.headroomCache;
    deps.headroomGain = &state.gains.headroom;
    deps.outputGain = &state.gains.output;
    deps.effectiveGain = &state.gains.effective;
    deps.activeRateFamily = [&state]() { return state.rates.activeRateFamily; };
    deps.activePhaseType = [&state]() { return state.rates.activePhaseType; };
    return deps;
}

static void update_daemon_dependencies(RuntimeState& state) {
    state.managers.daemonDependencies.config = &state.config;
    state.managers.daemonDependencies.running = &state.flags.running;
    state.managers.daemonDependencies.outputReady = &state.flags.outputReady;
    state.managers.daemonDependencies.crossfeedEnabled = &state.crossfeed.enabled;
    state.managers.daemonDependencies.currentInputRate = &state.rates.currentInputRate;
    state.managers.daemonDependencies.currentOutputRate = &state.rates.currentOutputRate;
    state.managers.daemonDependencies.softMute = &state.softMute.controller;
    state.managers.daemonDependencies.upsampler = &state.upsampler;
    state.managers.daemonDependencies.audioPipeline = &state.managers.audioPipelineRaw;
    state.managers.daemonDependencies.dacManager = &state.managers.dacManagerRaw;
    state.managers.daemonDependencies.streamingMutex = &state.streaming.streamingMutex;
    state.managers.daemonDependencies.refreshHeadroom = [&state](const std::string& reason) {
        if (state.managers.headroomController) {
            state.managers.headroomController->refreshCurrentHeadroom(reason);
        }
    };
}

static void initialize_event_modules(RuntimeState& state) {
    state.managers.eventDispatcher = std::make_unique<daemon_core::api::EventDispatcher>();
    update_daemon_dependencies(state);

    state.managers.rateSwitcher = std::make_unique<audio_pipeline::RateSwitcher>(
        audio_pipeline::RateSwitcherDependencies{.dispatcher = state.managers.eventDispatcher.get(),
                                                 .deps = state.managers.daemonDependencies,
                                                 .pendingRate = &state.rates.pendingRateChange});
    state.managers.filterManager =
        std::make_unique<audio_pipeline::FilterManager>(audio_pipeline::FilterManagerDependencies{
            .dispatcher = state.managers.eventDispatcher.get(),
            .deps = state.managers.daemonDependencies});
    state.managers.headroomController = std::make_unique<audio_pipeline::HeadroomController>(
        make_headroom_dependencies(state, state.managers.eventDispatcher.get()));
    state.managers.softMuteRunner =
        std::make_unique<audio_pipeline::SoftMuteRunner>(audio_pipeline::SoftMuteRunnerDependencies{
            .dispatcher = state.managers.eventDispatcher.get(),
            .deps = state.managers.daemonDependencies});
    state.managers.alsaOutputInterface = std::make_unique<daemon_output::AlsaOutput>(
        daemon_output::AlsaOutputDependencies{.dispatcher = state.managers.eventDispatcher.get(),
                                              .deps = state.managers.daemonDependencies});
    state.managers.handlerRegistry = std::make_unique<daemon_control::handlers::HandlerRegistry>(
        daemon_control::handlers::HandlerRegistryDependencies{
            .dispatcher = state.managers.eventDispatcher.get()});

    state.managers.rateSwitcher->start();
    state.managers.filterManager->start();
    state.managers.headroomController->start();
    state.managers.softMuteRunner->start();
    state.managers.alsaOutputInterface->start();
    state.managers.handlerRegistry->registerDefaults();
}

static void publish_rate_change_event(RuntimeState& state, int detected_rate) {
    if (!state.managers.rateSwitcher || !state.managers.eventDispatcher) {
        return;
    }
    daemon_core::api::RateChangeRequested event;
    event.detectedInputRate = detected_rate;
    event.rateFamily = ConvolutionEngine::detectRateFamily(detected_rate);
    state.managers.eventDispatcher->publish(event);
}

static void publish_filter_switch_event(RuntimeState& state, const std::string& filterPath,
                                        PhaseType phaseType, bool reloadHeadroom) {
    daemon_core::api::FilterSwitchRequested event;
    event.filterPath = filterPath;
    event.phaseType = phaseType;
    event.reloadHeadroom = reloadHeadroom;
    if (state.managers.eventDispatcher) {
        state.managers.eventDispatcher->publish(event);
    }
}

static void initialize_streaming_cache_manager(RuntimeState& state);

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

static size_t get_playback_ready_threshold(RuntimeState& state, size_t period_size) {
    bool crossfeedActive = false;
    size_t crossfeedBlockSize = 0;
    size_t producerBlockSize = 0;

    if (state.crossfeed.enabled.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> cf_lock(state.crossfeed.crossfeedMutex);
        if (state.crossfeed.processor) {
            crossfeedActive = true;
            crossfeedBlockSize = state.crossfeed.processor->getStreamValidInputPerBlock();
        }
    }

    if (state.upsampler) {
        // StreamValidInputPerBlock() is in input frames. Multiply by upsample ratio to obtain the
        // number of samples the producer actually contributes to the playback ring per block so the
        // ALSA thread can wake up as soon as a full GPU block finishes.
        size_t streamBlock = state.upsampler->getStreamValidInputPerBlock();
        int upsampleRatio = state.upsampler->getUpsampleRatio();
        if (streamBlock > 0 && upsampleRatio > 0) {
            producerBlockSize = streamBlock * static_cast<size_t>(upsampleRatio);
        }
    }

    return PlaybackBuffer::computeReadyThreshold(period_size, crossfeedActive, crossfeedBlockSize,
                                                 producerBlockSize);
}

// Fallback manager (Issue #139)

// Crossfeed enable/disable safety (Issue #888)
// - Avoid mixing pre/post switch audio by clearing playback + streaming caches.
// - Do not touch SoftMute here; caller wraps this with a fade-out/in.
static bool reset_streaming_caches_for_switch(RuntimeState& state) {
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
    } pauseGuard(state.audioPipeline.get());

    if (!pauseGuard.ok) {
        return false;
    }

    playback_buffer(state).reset();
    playback_buffer(state).cv().notify_all();

    {
        std::lock_guard<std::mutex> lock(state.streaming.streamingMutex);
        if (!state.streaming.streamInputLeft.empty()) {
            std::fill(state.streaming.streamInputLeft.begin(),
                      state.streaming.streamInputLeft.end(), 0.0f);
        }
        if (!state.streaming.streamInputRight.empty()) {
            std::fill(state.streaming.streamInputRight.begin(),
                      state.streaming.streamInputRight.end(), 0.0f);
        }
        state.streaming.streamAccumulatedLeft = 0;
        state.streaming.streamAccumulatedRight = 0;
        state.streaming.upsamplerOutputLeft.clear();
        state.streaming.upsamplerOutputRight.clear();
        if (state.upsampler) {
            state.upsampler->resetStreaming();
        }
    }

    { state.crossfeed.resetRequested.store(true, std::memory_order_release); }

    return true;
}

static void initialize_streaming_cache_manager(RuntimeState& state) {
    streaming_cache::StreamingCacheDependencies deps;
    deps.flushAction = [&state](std::chrono::nanoseconds /*gap*/) -> bool {
        // DAC 切断/入力ギャップ後の復帰で「古いオーバーラップ状態」を引きずらないよう、
        // 制御系スレッドで RT を静止させた上でキャッシュをフラッシュする。
        if (state.audioPipeline) {
            state.audioPipeline->requestRtPause();
            if (!state.audioPipeline->waitForRtQuiescent(std::chrono::milliseconds(500))) {
                state.audioPipeline->resumeRtPause();
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
        } pauseRelease(state.audioPipeline.get());

        if (state.softMute.controller) {
            state.softMute.controller->startFadeOut();
        }

        playback_buffer(state).reset();

        {
            std::lock_guard<std::mutex> streamLock(state.streaming.streamingMutex);
            if (!state.streaming.streamInputLeft.empty()) {
                std::fill(state.streaming.streamInputLeft.begin(),
                          state.streaming.streamInputLeft.end(), 0.0f);
            }
            if (!state.streaming.streamInputRight.empty()) {
                std::fill(state.streaming.streamInputRight.begin(),
                          state.streaming.streamInputRight.end(), 0.0f);
            }
            state.streaming.streamAccumulatedLeft = 0;
            state.streaming.streamAccumulatedRight = 0;
            state.streaming.upsamplerOutputLeft.clear();
            state.streaming.upsamplerOutputRight.clear();
        }

        if (state.upsampler) {
            state.upsampler->resetStreaming();
        }

        state.crossfeed.resetRequested.store(true, std::memory_order_release);

        if (state.softMute.controller) {
            state.softMute.controller->startFadeIn();
        }
        return true;
    };
    state.managers.streamingCacheManager =
        std::make_unique<streaming_cache::StreamingCacheManager>(deps);
}

// ========== DAC Device Monitoring ==========

static inline int64_t get_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static dac::DacManager::Dependencies make_dac_dependencies(
    RuntimeState& state, std::function<void(const nlohmann::json&)> eventPublisher = {}) {
    dac::DacManager::Dependencies deps;
    deps.config = &state.config;
    deps.runningFlag = &state.flags.running;
    deps.timestampProvider = get_timestamp_ms;
    deps.eventPublisher = std::move(eventPublisher);
    deps.defaultDevice = DEFAULT_ALSA_DEVICE;
    return deps;
}

static runtime_stats::Dependencies build_runtime_stats_dependencies(RuntimeState& state) {
    runtime_stats::Dependencies deps;
    deps.config = &state.config;
    deps.upsampler = state.upsampler;
    deps.headroomCache = &state.headroomCache;
    deps.dacManager = state.managers.dacManager.get();
    deps.fallbackManager = state.fallbackManager;
    deps.fallbackActive = &state.fallbackActive;
    deps.inputSampleRate = &state.rates.inputSampleRate;
    deps.headroomGain = &state.gains.headroom;
    deps.outputGain = &state.gains.output;
    deps.limiterGain = &state.gains.limiter;
    deps.effectiveGain = &state.gains.effective;
    return deps;
}

// ========== Configuration ==========

static void print_config_summary(RuntimeState& state, const AppConfig& cfg) {
    int outputRate = state.rates.inputSampleRate * cfg.upsampleRatio;
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
    LOG_INFO("  Input rate:     {} Hz (auto-negotiated)", state.rates.inputSampleRate);
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
    gpu_upsampler::metrics::setAudioConfig(state.rates.inputSampleRate, outputRate,
                                           cfg.upsampleRatio);
}

static void reset_runtime_state(RuntimeState& state) {
    state.managers.streamingCacheManager.reset();

    playback_buffer(state).reset();
    state.streaming.streamInputLeft.clear();
    state.streaming.streamInputRight.clear();
    state.streaming.streamAccumulatedLeft = 0;
    state.streaming.streamAccumulatedRight = 0;
    state.streaming.upsamplerOutputLeft.clear();
    state.streaming.upsamplerOutputRight.clear();

    // Reset crossfeed streaming buffers
    daemon_audio::clearCrossfeedRuntimeBuffers(state.crossfeed);
}

static bool reinitialize_streaming_for_legacy_mode(RuntimeState& state) {
    if (!state.upsampler) {
        return false;
    }

    std::lock_guard<std::mutex> streamLock(state.streaming.streamingMutex);
    state.upsampler->resetStreaming();
    playback_buffer(state).reset();

    state.streaming.streamInputLeft.clear();
    state.streaming.streamInputRight.clear();
    state.streaming.streamAccumulatedLeft = 0;
    state.streaming.streamAccumulatedRight = 0;
    state.streaming.upsamplerOutputLeft.clear();
    state.streaming.upsamplerOutputRight.clear();

    // Rebuild legacy streams so the buffers match the full FFT (avoids invalid cudaMemset after
    // disabling partitions).
    if (!state.upsampler->initializeStreaming()) {
        std::cerr << "[Partition] Failed to initialize legacy streaming buffers" << '\n';
        return false;
    }

    size_t buffer_capacity =
        compute_stream_buffer_capacity(state, state.upsampler->getStreamValidInputPerBlock());
    state.streaming.streamInputLeft.resize(buffer_capacity, 0.0f);
    state.streaming.streamInputRight.resize(buffer_capacity, 0.0f);
    state.streaming.streamAccumulatedLeft = 0;
    state.streaming.streamAccumulatedRight = 0;

    size_t upsampler_output_capacity =
        buffer_capacity * static_cast<size_t>(state.upsampler->getUpsampleRatio());
    state.streaming.upsamplerOutputLeft.reserve(upsampler_output_capacity);
    state.streaming.upsamplerOutputRight.reserve(upsampler_output_capacity);

    return true;
}

static bool handle_rate_switch(RuntimeState& state, int newInputRate) {
    if (!state.upsampler || !state.upsampler->isMultiRateEnabled()) {
        std::cerr << "[Rate] Multi-rate mode not enabled" << '\n';
        return false;
    }

    int currentRate = state.upsampler->getCurrentInputRate();
    if (currentRate == newInputRate) {
        std::cout << "[Rate] Already at target rate: " << newInputRate << " Hz" << '\n';
        return true;
    }

    std::cout << "[Rate] Switching: " << currentRate << " Hz -> " << newInputRate << " Hz" << '\n';

    int savedRate = currentRate;
    ConvolutionEngine::RateFamily targetFamily = ConvolutionEngine::detectRateFamily(newInputRate);
    if (targetFamily == ConvolutionEngine::RateFamily::RATE_UNKNOWN) {
        targetFamily = ConvolutionEngine::RateFamily::RATE_44K;
    }

    if (state.softMute.controller) {
        state.softMute.controller->startFadeOut();
        auto startTime = std::chrono::steady_clock::now();
        while (state.softMute.controller->isTransitioning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto elapsed = std::chrono::steady_clock::now() - startTime;
            if (elapsed > std::chrono::milliseconds(200)) {
                std::cerr << "[Rate] Warning: Fade-out timeout" << '\n';
                break;
            }
        }
    }

    int newOutputRate = state.upsampler->getOutputSampleRate();
    int newUpsampleRatio = state.upsampler->getUpsampleRatio();
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
    } pauseGuard(state.audioPipeline.get());

    if (!pauseGuard.ok) {
        if (state.softMute.controller) {
            state.softMute.controller->setPlaying();
        }
        return false;
    }

    {
        std::lock_guard<std::mutex> streamLock(state.streaming.streamingMutex);

        state.upsampler->resetStreaming();

        playback_buffer(state).reset();
        state.streaming.streamInputLeft.clear();
        state.streaming.streamInputRight.clear();
        state.streaming.streamAccumulatedLeft = 0;
        state.streaming.streamAccumulatedRight = 0;

        if (!state.upsampler->switchToInputRate(newInputRate)) {
            std::cerr << "[Rate] Failed to switch rate, rolling back" << '\n';
            if (state.upsampler->switchToInputRate(savedRate)) {
                std::cout << "[Rate] Rollback successful: restored to " << savedRate << " Hz"
                          << '\n';
            } else {
                std::cerr << "[Rate] ERROR: Rollback failed!" << '\n';
            }
            if (state.softMute.controller) {
                state.softMute.controller->startFadeIn();
            }
            return false;
        }

        if (!state.upsampler->initializeStreaming()) {
            std::cerr << "[Rate] Failed to re-initialize streaming mode, rolling back" << '\n';
            if (state.upsampler->switchToInputRate(savedRate)) {
                if (state.upsampler->initializeStreaming()) {
                    std::cout << "[Rate] Rollback successful: restored to " << savedRate << " Hz"
                              << '\n';
                }
            }
            if (state.softMute.controller) {
                state.softMute.controller->startFadeIn();
            }
            return false;
        }

        state.rates.inputSampleRate = newInputRate;
        set_rate_family(state, targetFamily);
        newOutputRate = state.upsampler->getOutputSampleRate();
        newUpsampleRatio = state.upsampler->getUpsampleRatio();

        buffer_capacity =
            compute_stream_buffer_capacity(state, state.upsampler->getStreamValidInputPerBlock());
        state.streaming.streamInputLeft.resize(buffer_capacity, 0.0f);
        state.streaming.streamInputRight.resize(buffer_capacity, 0.0f);
        state.streaming.streamAccumulatedLeft = 0;
        state.streaming.streamAccumulatedRight = 0;
        size_t upsampler_output_capacity =
            buffer_capacity * static_cast<size_t>(state.upsampler->getUpsampleRatio());
        state.streaming.upsamplerOutputLeft.reserve(upsampler_output_capacity);
        state.streaming.upsamplerOutputRight.reserve(upsampler_output_capacity);

        if (state.softMute.controller) {
            delete state.softMute.controller;
        }
        state.softMute.controller = new SoftMute::Controller(50, newOutputRate);
    }

    if (state.crossfeed.processor) {
        std::lock_guard<std::mutex> cfLock(state.crossfeed.crossfeedMutex);
        auto status = daemon_audio::switchCrossfeedRateFamilyLocked(state.crossfeed, state.config,
                                                                    targetFamily);
        if (status == daemon_audio::CrossfeedSwitchStatus::Failed) {
            std::cerr << "[Rate] Warning: Failed to switch crossfeed HRTF rate family" << '\n';
        }
    }

    if (state.softMute.controller) {
        state.softMute.controller->startFadeIn();
    }

    std::cout << "[Rate] Switch complete: " << newInputRate << " Hz (" << newUpsampleRatio
              << "x -> " << newOutputRate << " Hz)" << '\n';
    std::cout << "[Rate] Streaming buffers re-initialized: " << buffer_capacity
              << " samples capacity" << '\n';

    return true;
}

static void load_runtime_config(RuntimeState& state, const std::string& configFilePath) {
    AppConfig loaded;
    bool found = loadAppConfig(configFilePath, loaded);
    state.config = loaded;
    ensure_output_config(state.config);

    if (state.config.alsaDevice.empty()) {
        set_preferred_output_device(state.config, DEFAULT_ALSA_DEVICE);
    }
    if (state.config.filterPath.empty()) {
        state.config.filterPath = DEFAULT_FILTER_PATH;
    }
    if (state.config.upsampleRatio <= 0) {
        state.config.upsampleRatio = DEFAULT_UPSAMPLE_RATIO;
    }
    if (state.config.blockSize <= 0) {
        state.config.blockSize = DEFAULT_BLOCK_SIZE;
    }
    if (state.config.bufferSize <= 0) {
        state.config.bufferSize = 262144;
    }
    if (state.config.periodSize <= 0) {
        state.config.periodSize = 32768;
    }
    if (state.config.loopback.device.empty()) {
        state.config.loopback.device = DEFAULT_LOOPBACK_DEVICE;
    }
    if (state.config.loopback.sampleRate == 0) {
        state.config.loopback.sampleRate = DEFAULT_INPUT_SAMPLE_RATE;
    }
    if (state.config.loopback.channels == 0) {
        state.config.loopback.channels = CHANNELS;
    }
    if (state.config.loopback.periodFrames == 0) {
        state.config.loopback.periodFrames = DEFAULT_LOOPBACK_PERIOD_FRAMES;
    }
    if (state.config.loopback.format.empty()) {
        state.config.loopback.format = "S16_LE";
    }
    if (state.config.loopback.enabled) {
        state.rates.inputSampleRate = static_cast<int>(state.config.loopback.sampleRate);
    }
    if (state.config.i2s.enabled) {
        // MVP: i2s.sampleRate can be 0 (= follow runtime/negotiated rate).
        // If explicitly specified, adopt it as the engine input sample rate.
        if (state.config.i2s.sampleRate != 0) {
            state.rates.inputSampleRate = static_cast<int>(state.config.i2s.sampleRate);
        }
    }
    if (state.rates.inputSampleRate != 44100 && state.rates.inputSampleRate != 48000) {
        state.rates.inputSampleRate = DEFAULT_INPUT_SAMPLE_RATE;
    }

    if (!found) {
        std::cout << "Config: Using defaults (no config.json found)" << '\n';
    }

    enforce_phase_partition_constraints(state.config);

    print_config_summary(state, state.config);
    if (!state.managers.headroomController) {
        state.managers.headroomController = std::make_unique<audio_pipeline::HeadroomController>(
            make_headroom_dependencies(state, nullptr));
    }
    if (state.managers.headroomController) {
        state.managers.headroomController->setTargetPeak(state.config.headroomTarget);
        state.managers.headroomController->resetEffectiveGain(
            "config load (pending filter headroom)");
    }
    float initialOutput = state.gains.output.load(std::memory_order_relaxed);
    state.gains.limiter.store(1.0f, std::memory_order_relaxed);
    state.gains.effective.store(initialOutput, std::memory_order_relaxed);
}

static snd_pcm_format_t parse_i2s_format(const std::string& formatStr) {
    // Same set as loopback (MVP)
    return daemon_input::parseLoopbackFormat(formatStr);
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

static void i2s_capture_thread(RuntimeState& state, const std::string& device,
                               snd_pcm_format_t format, unsigned int requested_rate,
                               unsigned int channels, snd_pcm_uframes_t period_frames) {
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

    state.i2s.captureRunning.store(true, std::memory_order_release);

    while (state.flags.running.load(std::memory_order_acquire)) {
        snd_pcm_uframes_t negotiated_period = configured_period_frames;
        unsigned int actual_rate = requested_rate;
        snd_pcm_t* handle = open_i2s_capture(device, format, requested_rate, channels,
                                             negotiated_period, actual_rate);
        {
            std::lock_guard<std::mutex> lock(state.i2s.handleMutex);
            state.i2s.handle = handle;
        }

        if (!handle) {
            // Device not ready / unplugged. Retry with backoff.
            state.i2s.captureReady.store(false, std::memory_order_release);
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
            int current = state.rates.inputSampleRate;
            if (current != static_cast<int>(actual_rate)) {
                LOG_WARN("[I2S] Detected input rate {} Hz (engine {} Hz). Scheduling rate follow.",
                         actual_rate, current);
                state.rates.pendingRateChange.store(static_cast<int>(actual_rate),
                                                    std::memory_order_release);
            }
        }

        state.i2s.captureReady.store(true, std::memory_order_release);

        bool needReopen = false;
        while (state.flags.running.load(std::memory_order_acquire)) {
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
                daemon_input::waitForCaptureReady(handle);
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
                daemon_input::waitForCaptureReady(handle);
                continue;
            }

            if (!daemon_input::convertPcmToFloat(rawBuffer, format, static_cast<size_t>(frames),
                                                 channels, floatBuffer)) {
                LOG_ERROR("[I2S] Unsupported format during conversion");
                needReopen = true;
                break;
            }

            if (state.audioPipeline) {
                state.audioPipeline->process(floatBuffer.data(), static_cast<uint32_t>(frames));
            }
        }

        snd_pcm_drop(handle);
        snd_pcm_close(handle);
        {
            std::lock_guard<std::mutex> lock(state.i2s.handleMutex);
            state.i2s.handle = nullptr;
        }
        state.i2s.captureReady.store(false, std::memory_order_release);

        if (!needReopen) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    state.i2s.captureRunning.store(false, std::memory_order_release);
    state.i2s.captureReady.store(false, std::memory_order_release);
    LOG_INFO("[I2S] Capture thread terminated");
}

// ALSA output thread (705.6kHz direct to DAC)
static void alsa_output_thread(RuntimeState& state) {
    elevate_realtime_priority("ALSA output");

    daemon_output::AlsaPcmController pcmController(daemon_output::AlsaPcmControllerDependencies{
        .config = &state.config,
        .dacManager = state.managers.dacManager.get(),
        .streamingCacheManager = state.managers.streamingCacheManager.get(),
        .fallbackManager = state.fallbackManager,
        .running = &state.flags.running,
        .outputReady = &state.flags.outputReady,
        .currentOutputRate =
            [&state]() { return state.rates.currentOutputRate.load(std::memory_order_acquire); },
    });

    if (!pcmController.openSelected()) {
        return;
    }

    auto period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
    if (period_size == 0) {
        period_size = static_cast<snd_pcm_uframes_t>(
            (state.config.periodSize > 0) ? state.config.periodSize : 32768);
    }
    std::vector<int32_t> interleaved_buffer(period_size * CHANNELS);
    std::vector<float> float_buffer(period_size * CHANNELS);  // for soft mute processing
    auto& bufferManager = playback_buffer(state);

    // Main playback loop
    while (state.flags.running) {
        // Heartbeat check every few hundred loops
        static int alive_counter = 0;
        if (++alive_counter > 200) {  // ~200 iterations ~ a few seconds depending on buffer wait
            alive_counter = 0;
            if (!pcmController.alive()) {
                LOG_EVERY_N(WARN, 5, "[ALSA] PCM disconnected/suspended, attempting reopen...");
                pcmController.close();
                while (state.flags.running && !pcmController.openSelected()) {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                }
                period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                if (period_size == 0) {
                    period_size = static_cast<snd_pcm_uframes_t>(
                        (state.config.periodSize > 0) ? state.config.periodSize : 32768);
                }
                interleaved_buffer.resize(period_size * CHANNELS);
                float_buffer.resize(period_size * CHANNELS);
                continue;
            }
        }

        // Issue #219: Check for pending ALSA reconfiguration (output rate changed)
        if (state.rates.alsaReconfigureNeeded.exchange(false, std::memory_order_acquire)) {
            int new_output_rate = state.rates.alsaNewOutputRate.load(std::memory_order_acquire);
            if (new_output_rate > 0) {
                LOG_INFO("[Main] Reconfiguring ALSA for new output rate {} Hz", new_output_rate);

                // Reconfigure ALSA with new rate
                if (pcmController.reconfigure(new_output_rate)) {
                    period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                    if (period_size == 0) {
                        period_size = static_cast<snd_pcm_uframes_t>(
                            (state.config.periodSize > 0) ? state.config.periodSize : 32768);
                    }
                    interleaved_buffer.resize(period_size * CHANNELS);
                    float_buffer.resize(period_size * CHANNELS);

                    // Update soft mute sample rate
                    if (state.softMute.controller) {
                        state.softMute.controller->setSampleRate(new_output_rate);
                    }

                    LOG_INFO("[Main] ALSA reconfiguration successful");
                } else {
                    // Failed to reconfigure - try to reopen with old rate
                    LOG_ERROR("[Main] ALSA reconfiguration failed, attempting recovery...");
                    int old_rate = state.rates.currentOutputRate.load(std::memory_order_acquire);
                    if (!pcmController.reconfigure(old_rate)) {
                        LOG_ERROR("[Main] ALSA recovery failed, waiting for reconnect...");
                    }
                }
            }
        }

        if (auto pendingDevice = state.managers.dacManager->consumePendingChange()) {
            std::string nextDevice = *pendingDevice;
            if (!nextDevice.empty() && nextDevice != pcmController.device()) {
                LOG_INFO("[ALSA] Switching output to {}", nextDevice);
                if (!pcmController.switchDevice(nextDevice)) {
                    continue;
                }
                period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                if (period_size == 0) {
                    period_size = static_cast<snd_pcm_uframes_t>(
                        (state.config.periodSize > 0) ? state.config.periodSize : 32768);
                }
                interleaved_buffer.resize(period_size * CHANNELS);
                float_buffer.resize(period_size * CHANNELS);
            }
        }

        // Wait for GPU processed data (dynamic threshold to avoid underflow with crossfeed)
        std::unique_lock<std::mutex> lock(bufferManager.mutex());
        size_t ready_threshold =
            get_playback_ready_threshold(state, static_cast<size_t>(period_size));
        bufferManager.cv().wait_for(
            lock, std::chrono::milliseconds(200), [&bufferManager, ready_threshold, &state] {
                return bufferManager.queuedFramesLocked() >= ready_threshold ||
                       !state.flags.running;
            });

        if (!state.flags.running) {
            break;
        }

        lock.unlock();

        audio_pipeline::RenderResult renderResult;
        if (state.audioPipeline) {
            renderResult = state.audioPipeline->renderOutput(static_cast<size_t>(period_size),
                                                             interleaved_buffer, float_buffer,
                                                             state.softMute.controller);
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
        maybe_restore_soft_mute_params(state);

        // Write to ALSA device
        long frames_written = pcmController.writeInterleaved(interleaved_buffer.data(),
                                                             static_cast<size_t>(period_size));
        if (frames_written < 0) {
            // Device may be gone; attempt reopen
            LOG_EVERY_N(ERROR, 10, "[ALSA] Write error: {}, retrying reopen...",
                        snd_strerror(frames_written));
            pcmController.close();
            while (state.flags.running && !pcmController.openSelected()) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
            if (!pcmController.isOpen()) {
                continue;
            }
            period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
            if (period_size == 0) {
                period_size = static_cast<snd_pcm_uframes_t>(
                    (state.config.periodSize > 0) ? state.config.periodSize : 32768);
            }
            interleaved_buffer.resize(period_size * CHANNELS);
            float_buffer.resize(period_size * CHANNELS);
        }
    }

    // Cleanup
    pcmController.close();
    LOG_INFO("[ALSA] Output thread terminated");
}

int App::run(const AppOptions& options, int argc, char* argv[]) {
    RuntimeState& state = *state_;
    const std::string configPath =
        options.configFilePath.empty() ? std::string(DEFAULT_CONFIG_FILE) : options.configFilePath;
    const std::string statsPath =
        options.statsFilePath.empty() ? std::string(STATS_FILE_PATH) : options.statsFilePath;

    shutdown_manager::ShutdownManager::Dependencies shutdownDeps{
        &state.softMute.controller, &state.flags.running, &state.flags.reloadRequested, nullptr};
    shutdown_manager::ShutdownManager shutdownManager(shutdownDeps);
    shutdownManager.installSignalHandlers();

    int exitCode = 0;

    do {
        shutdownManager.reset();
        state.flags.running = true;
        state.flags.reloadRequested = false;
        state.flags.zmqBindFailed.store(false);
        reset_runtime_state(state);

        // Load configuration from config.json (if exists)
        load_runtime_config(state, configPath);

        // Environment variable overrides config.json
        if (const char* env_dev = std::getenv("ALSA_DEVICE")) {
            set_preferred_output_device(state.config, env_dev);
            std::cout << "Config: ALSA_DEVICE env override: " << env_dev << '\n';
        }

        // Command line argument overrides filter path
        if (argc > 1) {
            state.config.filterPath = argv[1];
            std::cout << "Config: CLI filter path override: " << argv[1] << '\n';
        }

        initialize_event_modules(state);

        PartitionRuntime::RuntimeRequest partitionRequest{
            state.config.partitionedConvolution.enabled, state.config.eqEnabled,
            state.config.crossfeed.enabled};

        auto buildResult = daemon_audio::buildUpsampler(state.config, state.rates.inputSampleRate,
                                                        partitionRequest, state.flags.running);
        if (buildResult.status == daemon_audio::UpsamplerBuildStatus::Failure) {
            exitCode = 1;
            break;
        }
        if (buildResult.status == daemon_audio::UpsamplerBuildStatus::Interrupted) {
            break;
        }

        state.upsampler = buildResult.upsampler.release();
        ConvolutionEngine::RateFamily initialFamily = buildResult.initialRateFamily;

        if (state.config.multiRateEnabled) {
            state.rates.currentInputRate.store(buildResult.currentInputRate,
                                               std::memory_order_release);
            state.rates.currentOutputRate.store(buildResult.currentOutputRate,
                                                std::memory_order_release);
            set_rate_family(state,
                            ConvolutionEngine::detectRateFamily(state.rates.inputSampleRate));
        }

        // Set state.rates.activeRateFamily and state.rates.activePhaseType for headroom
        // tracking
        if (state.config.multiRateEnabled) {
            // Rate family already set during initializeMultiRate()
            // state.rates.activeRateFamily is set via set_rate_family() above
        } else {
            state.rates.activeRateFamily = initialFamily;
        }

        state.rates.activePhaseType = state.config.phaseType;
        std::string filterPath;
        if (state.managers.headroomController) {
            filterPath = state.managers.headroomController->currentFilterPath();
        }
        publish_filter_switch_event(state, filterPath, state.rates.activePhaseType, true);

        // Pre-allocate streaming input buffers (avoid RT reallocations)
        size_t buffer_capacity =
            compute_stream_buffer_capacity(state, state.upsampler->getStreamValidInputPerBlock());
        state.streaming.streamInputLeft.resize(buffer_capacity, 0.0f);
        state.streaming.streamInputRight.resize(buffer_capacity, 0.0f);
        std::cout << "Streaming buffer capacity: " << buffer_capacity
                  << " samples (pre-sized for RT path)" << '\n';
        state.streaming.streamAccumulatedLeft = 0;
        state.streaming.streamAccumulatedRight = 0;
        size_t upsampler_output_capacity =
            buffer_capacity * static_cast<size_t>(state.config.upsampleRatio);
        state.streaming.upsamplerOutputLeft.reserve(upsampler_output_capacity);
        state.streaming.upsamplerOutputRight.reserve(upsampler_output_capacity);
        initialize_streaming_cache_manager(state);

        (void)daemon_audio::initializeCrossfeed(state, state.config.partitionedConvolution.enabled);

        std::cout << '\n';

        if (!state.audioPipeline && state.upsampler) {
            audio_pipeline::Dependencies pipelineDeps{};
            pipelineDeps.config = &state.config;
            pipelineDeps.upsampler.available = true;
            pipelineDeps.upsampler.streamLeft = state.upsampler->streamLeft_;
            pipelineDeps.upsampler.streamRight = state.upsampler->streamRight_;
            pipelineDeps.output.outputGain = &state.gains.output;
            pipelineDeps.output.limiterGain = &state.gains.limiter;
            pipelineDeps.output.effectiveGain = &state.gains.effective;
            pipelineDeps.upsampler.process =
                [&state](const float* data, size_t frames,
                         ConvolutionEngine::StreamFloatVector& output, cudaStream_t stream,
                         ConvolutionEngine::StreamFloatVector& streamInput,
                         size_t& streamAccumulated) {
                    if (!state.upsampler) {
                        return false;
                    }
                    return state.upsampler->processStreamBlock(data, frames, output, stream,
                                                               streamInput, streamAccumulated);
                };
            pipelineDeps.fallbackActive = &state.fallbackActive;
            pipelineDeps.outputReady = &state.flags.outputReady;
            pipelineDeps.streamingCacheManager = state.managers.streamingCacheManager.get();
            pipelineDeps.streamInputLeft = &state.streaming.streamInputLeft;
            pipelineDeps.streamInputRight = &state.streaming.streamInputRight;
            pipelineDeps.streamAccumulatedLeft = &state.streaming.streamAccumulatedLeft;
            pipelineDeps.streamAccumulatedRight = &state.streaming.streamAccumulatedRight;
            pipelineDeps.upsamplerOutputLeft = &state.streaming.upsamplerOutputLeft;
            pipelineDeps.upsamplerOutputRight = &state.streaming.upsamplerOutputRight;
            pipelineDeps.cfStreamInputLeft = &state.crossfeed.cfStreamInputLeft;
            pipelineDeps.cfStreamInputRight = &state.crossfeed.cfStreamInputRight;
            pipelineDeps.cfStreamAccumulatedLeft = &state.crossfeed.cfStreamAccumulatedLeft;
            pipelineDeps.cfStreamAccumulatedRight = &state.crossfeed.cfStreamAccumulatedRight;
            pipelineDeps.cfOutputLeft = &state.crossfeed.cfOutputLeft;
            pipelineDeps.cfOutputRight = &state.crossfeed.cfOutputRight;
            pipelineDeps.crossfeedEnabled = &state.crossfeed.enabled;
            pipelineDeps.crossfeedResetRequested = &state.crossfeed.resetRequested;
            pipelineDeps.crossfeedProcessor = state.crossfeed.processor;
            pipelineDeps.buffer.playbackBuffer = &playback_buffer(state);
            pipelineDeps.maxOutputBufferFrames = [&state]() {
                return get_max_output_buffer_frames(state);
            };
            pipelineDeps.currentOutputRate = [&state]() {
                return state.rates.currentOutputRate.load(std::memory_order_acquire);
            };
            state.audioPipeline =
                std::make_unique<audio_pipeline::AudioPipeline>(std::move(pipelineDeps));
            state.managers.audioPipelineRaw = state.audioPipeline.get();
        }

        // Check for early abort before starting threads
        if (!state.flags.running) {
            std::cout << "Startup interrupted by signal" << '\n';
            delete state.upsampler;
            state.upsampler = nullptr;
            break;
        }

        // Initialize soft mute controller with output sample rate
        int outputSampleRate = state.rates.inputSampleRate * state.config.upsampleRatio;
        state.softMute.controller =
            new SoftMute::Controller(DEFAULT_SOFT_MUTE_FADE_MS, outputSampleRate);
        std::cout << "Soft mute initialized (" << DEFAULT_SOFT_MUTE_FADE_MS << "ms fade at "
                  << outputSampleRate << "Hz)" << '\n';

        // Initialize fallback manager (Issue #139)
        if (state.config.fallback.enabled) {
            state.fallbackManager = new FallbackManager::Manager();
            FallbackManager::FallbackConfig fbConfig;
            fbConfig.gpuThreshold = state.config.fallback.gpuThreshold;
            fbConfig.gpuThresholdCount = state.config.fallback.gpuThresholdCount;
            fbConfig.gpuRecoveryThreshold = state.config.fallback.gpuRecoveryThreshold;
            fbConfig.gpuRecoveryCount = state.config.fallback.gpuRecoveryCount;
            fbConfig.xrunTriggersFallback = state.config.fallback.xrunTriggersFallback;
            fbConfig.monitorIntervalMs = state.config.fallback.monitorIntervalMs;

            // State change callback: update atomic flag and notify via ZeroMQ
            auto stateCallback = [&state](FallbackManager::FallbackState nextState) {
                bool isFallback = (nextState == FallbackManager::FallbackState::Fallback);
                state.fallbackActive.store(isFallback, std::memory_order_relaxed);

                // ZeroMQ notification is handled by STATS command response
                LOG_INFO("Fallback state changed: {}", isFallback ? "FALLBACK" : "NORMAL");
            };

            if (state.fallbackManager->initialize(fbConfig, stateCallback)) {
                std::cout << "Fallback manager initialized (GPU threshold: "
                          << fbConfig.gpuThreshold << "%, XRUN fallback: "
                          << (fbConfig.xrunTriggersFallback ? "enabled" : "disabled") << ")"
                          << '\n';
            } else {
                std::cerr << "Warning: Failed to initialize fallback manager" << '\n';
                delete state.fallbackManager;
                state.fallbackManager = nullptr;
                state.fallbackActive.store(false, std::memory_order_relaxed);
            }
        } else {
            std::cout << "Fallback manager disabled" << '\n';
            state.fallbackActive.store(false, std::memory_order_relaxed);
        }

        std::unique_ptr<daemon_control::ControlPlane> controlPlane;
        // Legacy RTP path was removed: ALSA/TCP-only configuration

        if (!state.managers.dacManager) {
            state.managers.dacManager =
                std::make_unique<dac::DacManager>(make_dac_dependencies(state));
        }
        if (!state.managers.dacManager) {
            std::cerr << "Failed to initialize DAC manager" << '\n';
            exitCode = 1;
            break;
        }
        state.managers.dacManagerRaw = state.managers.dacManager.get();

        daemon_control::ControlPlaneDependencies controlDeps{};
        controlDeps.config = &state.config;
        controlDeps.runningFlag = &state.flags.running;
        controlDeps.reloadRequested = &state.flags.reloadRequested;
        controlDeps.zmqBindFailed = &state.flags.zmqBindFailed;
        controlDeps.currentOutputRate = &state.rates.currentOutputRate;
        controlDeps.softMute = &state.softMute.controller;
        controlDeps.activePhaseType = &state.rates.activePhaseType;
        controlDeps.inputSampleRate = &state.rates.inputSampleRate;
        controlDeps.defaultAlsaDevice = DEFAULT_ALSA_DEVICE;
        controlDeps.dispatcher = state.managers.eventDispatcher.get();
        controlDeps.quitMainLoop = []() {};
        controlDeps.buildRuntimeStats = [&state]() {
            return build_runtime_stats_dependencies(state);
        };
        controlDeps.bufferCapacityFrames = [&state]() {
            return get_max_output_buffer_frames(state);
        };
        controlDeps.applySoftMuteForFilterSwitch = [&state](std::function<bool()> fn) {
            applySoftMuteForFilterSwitch(state, std::move(fn));
        };
        controlDeps.resetStreamingCachesForSwitch = [&state]() {
            return reset_streaming_caches_for_switch(state);
        };
        controlDeps.refreshHeadroom = [&state](const std::string& reason) {
            if (state.managers.headroomController) {
                state.managers.headroomController->refreshCurrentHeadroom(reason);
            }
        };
        controlDeps.reinitializeStreamingForLegacyMode = [&state]() {
            return reinitialize_streaming_for_legacy_mode(state);
        };
        controlDeps.setPreferredOutputDevice = [](AppConfig& cfg, const std::string& device) {
            set_preferred_output_device(cfg, device);
        };
        controlDeps.dacManager = state.managers.dacManager.get();
        controlDeps.upsampler = &state.upsampler;
        controlDeps.crossfeed.processor = &state.crossfeed.processor;
        controlDeps.crossfeed.enabledFlag = &state.crossfeed.enabled;
        controlDeps.crossfeed.mutex = &state.crossfeed.crossfeedMutex;
        controlDeps.crossfeed.resetStreamingState = [&state]() {
            state.crossfeed.resetRequested.store(true, std::memory_order_release);
        };
        controlDeps.statsFilePath = statsPath;

        controlPlane = std::make_unique<daemon_control::ControlPlane>(std::move(controlDeps));
        if (controlPlane && controlPlane->start() && state.managers.dacManager) {
            state.managers.dacManager->setEventPublisher(controlPlane->eventPublisher());
        }

        if (state.managers.dacManager) {
            state.managers.dacManager->initialize();
        }

        // Start ALSA output thread
        if (state.managers.dacManager) {
            state.managers.dacManager->start();
        }
        std::cout << "Starting ALSA output thread..." << '\n';
        std::thread alsa_thread(alsa_output_thread, std::ref(state));

        std::thread loopback_thread;
        std::thread i2s_thread;
        bool startupFailed = false;

        auto failStartup = [&](const std::string& reason) {
            LOG_ERROR("Startup failed: {}", reason);
            exitCode = 1;
            startupFailed = true;
            state.flags.running = false;
            playback_buffer(state).cv().notify_all();
        };

        if (state.config.i2s.enabled && state.config.loopback.enabled) {
            failStartup("Config error: i2s.enabled and loopback.enabled cannot both be true");
        }

        if (!startupFailed && state.config.i2s.enabled) {
            if (!validate_i2s_config(state.config)) {
                failStartup("Invalid I2S config");
            } else {
                snd_pcm_format_t i2s_format = parse_i2s_format(state.config.i2s.format);
                // i2s.sampleRate==0 means "follow engine/runtime negotiated input rate".
                // ALSA hw_params typically requires an explicit rate, so default to current engine
                // rate.
                unsigned int i2s_rate =
                    (state.config.i2s.sampleRate != 0)
                        ? static_cast<unsigned int>(state.config.i2s.sampleRate)
                        : static_cast<unsigned int>(state.rates.inputSampleRate);
                std::cout << "Starting I2S capture thread (" << state.config.i2s.device
                          << ", fmt=" << state.config.i2s.format << ", rate=" << i2s_rate
                          << ", period=" << state.config.i2s.periodFrames << ")" << '\n';
                i2s_thread =
                    std::thread(i2s_capture_thread, std::ref(state), state.config.i2s.device,
                                i2s_format, i2s_rate, state.config.i2s.channels,
                                static_cast<snd_pcm_uframes_t>(state.config.i2s.periodFrames));

                // Wait briefly for I2S thread to become ready
                bool i2s_ready = false;
                for (int i = 0; i < 40; ++i) {  // up to ~2s
                    if (state.i2s.captureReady.load(std::memory_order_acquire)) {
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
        if (!startupFailed && state.config.loopback.enabled) {
            if (!daemon_input::validateLoopbackConfig(state.config)) {
                failStartup("Invalid loopback config");
            } else {
                snd_pcm_format_t lb_format =
                    daemon_input::parseLoopbackFormat(state.config.loopback.format);
                std::cout << "Starting loopback capture thread (" << state.config.loopback.device
                          << ", fmt=" << state.config.loopback.format
                          << ", rate=" << state.config.loopback.sampleRate
                          << ", period=" << state.config.loopback.periodFrames << ")" << '\n';

                daemon_input::LoopbackCaptureDependencies loopbackDeps{
                    .playbackBuffer = &playback_buffer(state),
                    .running = &state.flags.running,
                    .currentOutputRate = &state.rates.currentOutputRate,
                    .audioPipeline = &state.managers.audioPipelineRaw,
                    .handleMutex = &state.loopback.handleMutex,
                    .handle = &state.loopback.handle,
                    .captureRunning = &state.loopback.captureRunning,
                    .captureReady = &state.loopback.captureReady,
                };

                loopback_thread = std::thread(
                    daemon_input::loopbackCaptureThread, state.config.loopback.device, lb_format,
                    state.config.loopback.sampleRate, state.config.loopback.channels,
                    static_cast<snd_pcm_uframes_t>(state.config.loopback.periodFrames),
                    loopbackDeps);

                // Wait briefly for loopback thread to become ready
                bool loopback_ready = false;
                for (int i = 0; i < 40; ++i) {  // up to ~2s
                    if (state.loopback.captureReady.load(std::memory_order_acquire)) {
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
                state.rates.inputSampleRate * state.config.upsampleRatio / 1000.0;
            std::cout << '\n';
            if (state.config.loopback.enabled) {
                std::cout << "System ready (Loopback capture mode). Audio routing configured:"
                          << '\n';
                std::cout << "  1. Loopback capture → GPU Upsampler (" << state.config.upsampleRatio
                          << "x upsampling)" << '\n';
                std::cout << "  2. GPU Upsampler → ALSA → SMSL DAC (" << outputRateKHz
                          << "kHz direct)" << '\n';
            } else if (state.config.i2s.enabled) {
                std::cout << "System ready (I2S capture mode). Audio routing configured:" << '\n';
                std::cout << "  1. I2S capture → GPU Upsampler (" << state.config.upsampleRatio
                          << "x upsampling)" << '\n';
                std::cout << "  2. GPU Upsampler → ALSA → DAC (" << outputRateKHz << "kHz direct)"
                          << '\n';
            } else {
                std::cout << "System ready (TCP/loopback input). Audio routing configured:" << '\n';
                std::cout << "  1. Network or loopback source → GPU Upsampler ("
                          << state.config.upsampleRatio << "x upsampling)" << '\n';
                std::cout << "  2. GPU Upsampler → ALSA → SMSL DAC (" << outputRateKHz
                          << "kHz direct)" << '\n';
            }
            std::cout << "Press Ctrl+C to stop." << '\n';
            std::cout << "========================================" << '\n';

            shutdownManager.notifyReady();

            auto runMainLoop = [&]() {
                while (state.flags.running.load() && !state.flags.reloadRequested.load() &&
                       !state.flags.zmqBindFailed.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    if (state.managers.streamingCacheManager) {
                        (void)state.managers.streamingCacheManager->drainFlushRequests();
                    }
                    // Input rate follow-up hook (Issue #906; network-driven detection is #824)
                    int pending =
                        state.rates.pendingRateChange.exchange(0, std::memory_order_acq_rel);
                    if (pending > 0 && (pending == 44100 || pending == 48000)) {
                        (void)handle_rate_switch(state, pending);
                    }
                    shutdownManager.tick();
                }
            };

            if (state.flags.zmqBindFailed.load()) {
                std::cerr << "Startup aborted due to ZeroMQ bind failure." << '\n';
            } else {
                runMainLoop();
            }
        }

        shutdownManager.runShutdownSequence();

        // Step 5: Signal worker threads to stop and wait for them
        std::cout << "  Step 5: Stopping worker threads..." << '\n';
        state.flags.running = false;
        playback_buffer(state).cv().notify_all();
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
        if (state.fallbackManager) {
            state.fallbackManager->shutdown();
            delete state.fallbackManager;
            state.fallbackManager = nullptr;
            state.fallbackActive.store(false, std::memory_order_relaxed);
        }
        delete state.softMute.controller;
        state.softMute.controller = nullptr;
        daemon_audio::shutdownCrossfeed(state.crossfeed);
        if (state.audioPipeline) {
            state.audioPipeline.reset();  // Tear down pipeline before deleting GPU upsampler
            state.managers.audioPipelineRaw = nullptr;
        }
        delete state.upsampler;
        state.upsampler = nullptr;
        if (state.managers.dacManager) {
            state.managers.dacManager->stop();
        }

        // Don't reload if ZMQ bind failed - exit completely
        if (state.flags.zmqBindFailed.load()) {
            std::cerr << "Exiting due to ZeroMQ initialization failure." << '\n';
            exitCode = 1;
            break;
        }

        if (state.flags.reloadRequested) {
            std::cout << "Reload requested. Restarting daemon with updated config..." << '\n';
        }
    } while (state.flags.reloadRequested);

    std::cout << "Goodbye!" << '\n';
    return exitCode;
}

}  // namespace daemon_app
