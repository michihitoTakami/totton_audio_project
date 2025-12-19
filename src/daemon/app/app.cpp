#include "daemon/app/app.h"

#include "audio/fallback_manager.h"
#include "audio/soft_mute.h"
#include "convolution_engine.h"
#include "core/config_loader.h"
#include "core/daemon_constants.h"
#include "core/partition_runtime_utils.h"
#include "daemon/api/events.h"
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
#include "daemon/output/playback_buffer_manager.h"
#include "daemon/pcm/dac_manager.h"
#include "daemon/shutdown_manager.h"
#include "io/playback_buffer.h"
#include "logging/logger.h"
#include "logging/metrics.h"

#include <algorithm>
#include <alsa/asoundlib.h>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

namespace daemon_app {
namespace {

constexpr const char* kDefaultAlsaDevice = "hw:USB";
constexpr const char* kDefaultLoopbackDevice = "hw:Loopback,1,0";
constexpr uint32_t kDefaultLoopbackPeriodFrames = 1024;
constexpr const char* kDefaultFilterPath = "data/coefficients/filter_44k_16x_2m_min_phase.bin";
static constexpr std::array<const char*, 1> kSupportedOutputModes = {"usb"};

using namespace DaemonConstants;

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
        resolved = kDefaultAlsaDevice;
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
            config.output.usb.preferredDevice = kDefaultAlsaDevice;
        }
    }

    if (config.alsaDevice.empty()) {
        config.alsaDevice = config.output.usb.preferredDevice;
    }
}

static void enforce_phase_partition_constraints(AppConfig& config) {
    if (config.partitionedConvolution.enabled && config.phaseType == PhaseType::Linear) {
        std::cout << "[Partition] Linear phase is incompatible with low-latency mode. "
                  << "Switching to minimum phase." << '\n';
        config.phaseType = PhaseType::Minimum;
    }
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
    return frames * 2;
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

static daemon_output::PlaybackBufferManager& playback_buffer(RuntimeState& state) {
    if (!state.playback.buffer) {
        state.playback.buffer = std::make_unique<daemon_output::PlaybackBufferManager>(
            [&state]() { return get_max_output_buffer_frames(state); });
    }
    return *state.playback.buffer;
}

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

static void applySoftMuteForFilterSwitch(RuntimeState& state,
                                         std::function<bool()> filterSwitchFunc) {
    using namespace DaemonConstants;

    if (!state.softMute.controller) {
        filterSwitchFunc();
        return;
    }

    std::lock_guard<std::mutex> lock(state.softMute.opMutex);

    state.softMute.restorePending.store(false, std::memory_order_release);

    int originalFadeDuration = state.softMute.controller->getFadeDuration();
    int outputSampleRate = state.softMute.controller->getSampleRate();

    state.softMute.controller->setFadeDuration(FILTER_SWITCH_FADE_MS);
    state.softMute.controller->setSampleRate(outputSampleRate);

    std::cout << "[Filter Switch] Starting fade-out (" << (FILTER_SWITCH_FADE_MS / 1000.0)
              << "s)..." << '\n';
    state.softMute.controller->startFadeOut();

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
        std::cout << "[Filter Switch] Starting fade-in (" << (FILTER_SWITCH_FADE_MS / 1000.0)
                  << "s)..." << '\n';
        state.softMute.controller->startFadeIn();

        state.softMute.restoreFadeMs.store(originalFadeDuration, std::memory_order_relaxed);
        state.softMute.restoreSampleRate.store(outputSampleRate, std::memory_order_relaxed);
        state.softMute.restorePending.store(true, std::memory_order_release);
    } else {
        std::cerr << "[Filter Switch] Switch failed, restoring audio state" << '\n';
        state.softMute.controller->setPlaying();
        state.softMute.controller->setFadeDuration(originalFadeDuration);
        state.softMute.controller->setSampleRate(outputSampleRate);
    }
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

    state.crossfeed.resetRequested.store(true, std::memory_order_release);

    return true;
}

static void initialize_streaming_cache_manager(RuntimeState& state) {
    streaming_cache::StreamingCacheDependencies deps;
    deps.flushAction = [&state](std::chrono::nanoseconds /*gap*/) -> bool {
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
    deps.defaultDevice = kDefaultAlsaDevice;
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

static void set_rate_family(RuntimeState& state, ConvolutionEngine::RateFamily family) {
    state.rates.currentRateFamilyInt.store(static_cast<int>(family), std::memory_order_release);
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
        set_preferred_output_device(state.config, kDefaultAlsaDevice);
    }
    if (state.config.filterPath.empty()) {
        state.config.filterPath = kDefaultFilterPath;
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
        state.config.loopback.device = kDefaultLoopbackDevice;
    }
    if (state.config.loopback.sampleRate == 0) {
        state.config.loopback.sampleRate = DEFAULT_INPUT_SAMPLE_RATE;
    }
    if (state.config.loopback.channels == 0) {
        state.config.loopback.channels = CHANNELS;
    }
    if (state.config.loopback.periodFrames == 0) {
        state.config.loopback.periodFrames = kDefaultLoopbackPeriodFrames;
    }
    if (state.config.loopback.format.empty()) {
        state.config.loopback.format = "S16_LE";
    }
    if (state.config.loopback.enabled) {
        state.rates.inputSampleRate = static_cast<int>(state.config.loopback.sampleRate);
    }
    if (state.config.i2s.enabled) {
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

        int err =
            snd_pcm_open(&handle, target_device.c_str(), SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK);
        if (err < 0) {
            LOG_ERROR("[I2S] Cannot open capture device {}: {}", target_device, snd_strerror(err));
            return nullptr;
        }

        snd_pcm_hw_params_t* hw_params;
        snd_pcm_hw_params_alloca(&hw_params);
        snd_pcm_hw_params_any(handle, hw_params);
        snd_pcm_hw_params_set_access(handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
        snd_pcm_hw_params_set_format(handle, hw_params, format);
        snd_pcm_hw_params_set_channels(handle, hw_params, channels);

        if (!auto_rate) {
            snd_pcm_hw_params_set_rate_near(handle, hw_params, &candidate_rate, nullptr);
        }

        snd_pcm_hw_params_set_period_size_near(handle, hw_params, &local_period, nullptr);

        if (snd_pcm_hw_params(handle, hw_params) < 0) {
            LOG_ERROR("[I2S] hw_params failed for device {}", target_device);
            snd_pcm_close(handle);
            return nullptr;
        }

        snd_pcm_hw_params_get_rate(hw_params, &actual_rate, nullptr);
        snd_pcm_hw_params_get_period_size(hw_params, &period_frames, nullptr);
        LOG_INFO("[I2S] Opened capture device {} (rate={}Hz, period={} frames)", target_device,
                 (auto_rate ? actual_rate : candidate_rate), period_frames);
        if (!auto_rate && actual_rate != candidate_rate) {
            LOG_WARN("[I2S] Requested {} Hz, got {} Hz", candidate_rate, actual_rate);
        }

        if (target_device != device) {
            LOG_WARN("[I2S] Using ALSA device fallback '{}' (configured was '{}')", target_device,
                     device);
        }

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
                return handle;
            }
        }
    }

    return nullptr;
}

static void elevate_realtime_priority(const char* name, int priority = 65) {
#ifdef __linux__
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
        size_t streamBlock = state.upsampler->getStreamValidInputPerBlock();
        int upsampleRatio = state.upsampler->getUpsampleRatio();
        if (streamBlock > 0 && upsampleRatio > 0) {
            producerBlockSize = streamBlock * static_cast<size_t>(upsampleRatio);
        }
    }

    return PlaybackBuffer::computeReadyThreshold(period_size, crossfeedActive, crossfeedBlockSize,
                                                 producerBlockSize);
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
                if (snd_pcm_recover(handle, frames, 1) < 0) {
                    snd_pcm_prepare(handle);
                }
                continue;
            }
            if (frames < 0) {
                LOG_WARN("[I2S] Read error: {}", snd_strerror(frames));
                if (snd_pcm_recover(handle, frames, 1) < 0) {
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
    std::vector<float> float_buffer(period_size * CHANNELS);
    auto& bufferManager = playback_buffer(state);

    while (state.flags.running) {
        static int alive_counter = 0;
        if (++alive_counter > 200) {
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

        if (state.rates.alsaReconfigureNeeded.exchange(false, std::memory_order_acquire)) {
            int new_output_rate = state.rates.alsaNewOutputRate.load(std::memory_order_acquire);
            if (new_output_rate > 0) {
                LOG_INFO("[Main] Reconfiguring ALSA for new output rate {} Hz", new_output_rate);

                if (pcmController.reconfigure(new_output_rate)) {
                    period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                    if (period_size == 0) {
                        period_size = static_cast<snd_pcm_uframes_t>(
                            (state.config.periodSize > 0) ? state.config.periodSize : 32768);
                    }
                    interleaved_buffer.resize(period_size * CHANNELS);
                    float_buffer.resize(period_size * CHANNELS);

                    if (state.softMute.controller) {
                        state.softMute.controller->setSampleRate(new_output_rate);
                    }

                    LOG_INFO("[Main] ALSA reconfiguration successful");
                } else {
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

        bufferManager.cv().notify_all();

        maybe_restore_soft_mute_params(state);

        long frames_written = pcmController.writeInterleaved(interleaved_buffer.data(),
                                                             static_cast<size_t>(period_size));
        if (frames_written < 0) {
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

    pcmController.close();
    LOG_INFO("[ALSA] Output thread terminated");
}

}  // namespace

App::App(RuntimeState& state, std::string configFilePath, std::string statsFilePath)
    : state_(state),
      configFilePath_(std::move(configFilePath)),
      statsFilePath_(std::move(statsFilePath)) {}

int App::run(const AppOverrides& overrides) {
    shutdown_manager::ShutdownManager::Dependencies shutdownDeps{
        &state_.softMute.controller, &state_.flags.running, &state_.flags.reloadRequested, nullptr};
    shutdown_manager::ShutdownManager shutdownManager(shutdownDeps);
    shutdownManager.installSignalHandlers();

    int exitCode = 0;

    do {
        shutdownManager.reset();
        state_.flags.running = true;
        state_.flags.reloadRequested = false;
        state_.flags.zmqBindFailed.store(false);
        reset_runtime_state(state_);

        load_runtime_config(state_, configFilePath_);

        if (overrides.alsaDevice) {
            set_preferred_output_device(state_.config, *overrides.alsaDevice);
        }
        if (overrides.filterPath) {
            state_.config.filterPath = *overrides.filterPath;
        }

        initialize_event_modules(state_);

        PartitionRuntime::RuntimeRequest partitionRequest{
            state_.config.partitionedConvolution.enabled, state_.config.eqEnabled,
            state_.config.crossfeed.enabled};

        auto buildResult = daemon_audio::buildUpsampler(state_.config, state_.rates.inputSampleRate,
                                                        partitionRequest, state_.flags.running);
        if (buildResult.status == daemon_audio::UpsamplerBuildStatus::Failure) {
            exitCode = 1;
            break;
        }
        if (buildResult.status == daemon_audio::UpsamplerBuildStatus::Interrupted) {
            break;
        }

        state_.upsampler = buildResult.upsampler.release();
        ConvolutionEngine::RateFamily initialFamily = buildResult.initialRateFamily;

        if (state_.config.multiRateEnabled) {
            state_.rates.currentInputRate.store(buildResult.currentInputRate,
                                                std::memory_order_release);
            state_.rates.currentOutputRate.store(buildResult.currentOutputRate,
                                                 std::memory_order_release);
            set_rate_family(state_,
                            ConvolutionEngine::detectRateFamily(state_.rates.inputSampleRate));
        }

        if (state_.config.multiRateEnabled) {
        } else {
            state_.rates.activeRateFamily = initialFamily;
        }

        state_.rates.activePhaseType = state_.config.phaseType;
        std::string filterPath;
        if (state_.managers.headroomController) {
            filterPath = state_.managers.headroomController->currentFilterPath();
        }
        publish_filter_switch_event(state_, filterPath, state_.rates.activePhaseType, true);

        size_t buffer_capacity =
            compute_stream_buffer_capacity(state_, state_.upsampler->getStreamValidInputPerBlock());
        state_.streaming.streamInputLeft.resize(buffer_capacity, 0.0f);
        state_.streaming.streamInputRight.resize(buffer_capacity, 0.0f);
        std::cout << "Streaming buffer capacity: " << buffer_capacity
                  << " samples (pre-sized for RT path)" << '\n';
        state_.streaming.streamAccumulatedLeft = 0;
        state_.streaming.streamAccumulatedRight = 0;
        size_t upsampler_output_capacity =
            buffer_capacity * static_cast<size_t>(state_.config.upsampleRatio);
        state_.streaming.upsamplerOutputLeft.reserve(upsampler_output_capacity);
        state_.streaming.upsamplerOutputRight.reserve(upsampler_output_capacity);
        initialize_streaming_cache_manager(state_);

        (void)daemon_audio::initializeCrossfeed(state_,
                                                state_.config.partitionedConvolution.enabled);

        std::cout << '\n';

        if (!state_.audioPipeline && state_.upsampler) {
            audio_pipeline::Dependencies pipelineDeps{};
            pipelineDeps.config = &state_.config;
            pipelineDeps.upsampler.available = true;
            pipelineDeps.upsampler.streamLeft = state_.upsampler->streamLeft_;
            pipelineDeps.upsampler.streamRight = state_.upsampler->streamRight_;
            pipelineDeps.output.outputGain = &state_.gains.output;
            pipelineDeps.output.limiterGain = &state_.gains.limiter;
            pipelineDeps.output.effectiveGain = &state_.gains.effective;
            pipelineDeps.upsampler.process =
                [&state = state_](const float* data, size_t frames,
                                  ConvolutionEngine::StreamFloatVector& output, cudaStream_t stream,
                                  ConvolutionEngine::StreamFloatVector& streamInput,
                                  size_t& streamAccumulated) {
                    if (!state.upsampler) {
                        return false;
                    }
                    return state.upsampler->processStreamBlock(data, frames, output, stream,
                                                               streamInput, streamAccumulated);
                };
            pipelineDeps.fallbackActive = &state_.fallbackActive;
            pipelineDeps.outputReady = &state_.flags.outputReady;
            pipelineDeps.streamingCacheManager = state_.managers.streamingCacheManager.get();
            pipelineDeps.streamInputLeft = &state_.streaming.streamInputLeft;
            pipelineDeps.streamInputRight = &state_.streaming.streamInputRight;
            pipelineDeps.streamAccumulatedLeft = &state_.streaming.streamAccumulatedLeft;
            pipelineDeps.streamAccumulatedRight = &state_.streaming.streamAccumulatedRight;
            pipelineDeps.upsamplerOutputLeft = &state_.streaming.upsamplerOutputLeft;
            pipelineDeps.upsamplerOutputRight = &state_.streaming.upsamplerOutputRight;
            pipelineDeps.cfStreamInputLeft = &state_.crossfeed.cfStreamInputLeft;
            pipelineDeps.cfStreamInputRight = &state_.crossfeed.cfStreamInputRight;
            pipelineDeps.cfStreamAccumulatedLeft = &state_.crossfeed.cfStreamAccumulatedLeft;
            pipelineDeps.cfStreamAccumulatedRight = &state_.crossfeed.cfStreamAccumulatedRight;
            pipelineDeps.cfOutputLeft = &state_.crossfeed.cfOutputLeft;
            pipelineDeps.cfOutputRight = &state_.crossfeed.cfOutputRight;
            pipelineDeps.crossfeedEnabled = &state_.crossfeed.enabled;
            pipelineDeps.crossfeedResetRequested = &state_.crossfeed.resetRequested;
            pipelineDeps.crossfeedProcessor = state_.crossfeed.processor;
            pipelineDeps.buffer.playbackBuffer = &playback_buffer(state_);
            pipelineDeps.maxOutputBufferFrames = [&state = state_]() {
                return get_max_output_buffer_frames(state);
            };
            pipelineDeps.currentOutputRate = [&state = state_]() {
                return state.rates.currentOutputRate.load(std::memory_order_acquire);
            };
            state_.audioPipeline =
                std::make_unique<audio_pipeline::AudioPipeline>(std::move(pipelineDeps));
            state_.managers.audioPipelineRaw = state_.audioPipeline.get();
        }

        if (!state_.flags.running) {
            std::cout << "Startup interrupted by signal" << '\n';
            delete state_.upsampler;
            state_.upsampler = nullptr;
            break;
        }

        using namespace DaemonConstants;
        int outputSampleRate = state_.rates.inputSampleRate * state_.config.upsampleRatio;
        state_.softMute.controller =
            new SoftMute::Controller(DEFAULT_SOFT_MUTE_FADE_MS, outputSampleRate);
        std::cout << "Soft mute initialized (" << DEFAULT_SOFT_MUTE_FADE_MS << "ms fade at "
                  << outputSampleRate << "Hz)" << '\n';

        if (state_.config.fallback.enabled) {
            state_.fallbackManager = new FallbackManager::Manager();
            FallbackManager::FallbackConfig fbConfig;
            fbConfig.gpuThreshold = state_.config.fallback.gpuThreshold;
            fbConfig.gpuThresholdCount = state_.config.fallback.gpuThresholdCount;
            fbConfig.gpuRecoveryThreshold = state_.config.fallback.gpuRecoveryThreshold;
            fbConfig.gpuRecoveryCount = state_.config.fallback.gpuRecoveryCount;
            fbConfig.xrunTriggersFallback = state_.config.fallback.xrunTriggersFallback;
            fbConfig.monitorIntervalMs = state_.config.fallback.monitorIntervalMs;

            auto stateCallback = [&state = state_](FallbackManager::FallbackState stateValue) {
                bool isFallback = (stateValue == FallbackManager::FallbackState::Fallback);
                state.fallbackActive.store(isFallback, std::memory_order_relaxed);
                LOG_INFO("Fallback state changed: {}", isFallback ? "FALLBACK" : "NORMAL");
            };

            if (state_.fallbackManager->initialize(fbConfig, stateCallback)) {
                std::cout << "Fallback manager initialized (GPU threshold: "
                          << fbConfig.gpuThreshold << "%, XRUN fallback: "
                          << (fbConfig.xrunTriggersFallback ? "enabled" : "disabled") << ")"
                          << '\n';
            } else {
                std::cerr << "Warning: Failed to initialize fallback manager" << '\n';
                delete state_.fallbackManager;
                state_.fallbackManager = nullptr;
                state_.fallbackActive.store(false, std::memory_order_relaxed);
            }
        } else {
            std::cout << "Fallback manager disabled" << '\n';
            state_.fallbackActive.store(false, std::memory_order_relaxed);
        }

        std::unique_ptr<daemon_control::ControlPlane> controlPlane;

        if (!state_.managers.dacManager) {
            state_.managers.dacManager =
                std::make_unique<dac::DacManager>(make_dac_dependencies(state_));
        }
        if (!state_.managers.dacManager) {
            std::cerr << "Failed to initialize DAC manager" << '\n';
            exitCode = 1;
            break;
        }
        state_.managers.dacManagerRaw = state_.managers.dacManager.get();

        daemon_control::ControlPlaneDependencies controlDeps{};
        controlDeps.config = &state_.config;
        controlDeps.runningFlag = &state_.flags.running;
        controlDeps.reloadRequested = &state_.flags.reloadRequested;
        controlDeps.zmqBindFailed = &state_.flags.zmqBindFailed;
        controlDeps.currentOutputRate = &state_.rates.currentOutputRate;
        controlDeps.softMute = &state_.softMute.controller;
        controlDeps.activePhaseType = &state_.rates.activePhaseType;
        controlDeps.inputSampleRate = &state_.rates.inputSampleRate;
        controlDeps.defaultAlsaDevice = kDefaultAlsaDevice;
        controlDeps.dispatcher = state_.managers.eventDispatcher.get();
        controlDeps.quitMainLoop = []() {};
        controlDeps.buildRuntimeStats = [&state = state_]() {
            return build_runtime_stats_dependencies(state);
        };
        controlDeps.bufferCapacityFrames = [&state = state_]() {
            return get_max_output_buffer_frames(state);
        };
        controlDeps.applySoftMuteForFilterSwitch = [&state = state_](std::function<bool()> fn) {
            applySoftMuteForFilterSwitch(state, std::move(fn));
        };
        controlDeps.resetStreamingCachesForSwitch = [&state = state_]() {
            return reset_streaming_caches_for_switch(state);
        };
        controlDeps.refreshHeadroom = [&state = state_](const std::string& reason) {
            if (state.managers.headroomController) {
                state.managers.headroomController->refreshCurrentHeadroom(reason);
            }
        };
        controlDeps.reinitializeStreamingForLegacyMode = [&state = state_]() {
            return reinitialize_streaming_for_legacy_mode(state);
        };
        controlDeps.setPreferredOutputDevice = [](AppConfig& cfg, const std::string& device) {
            set_preferred_output_device(cfg, device);
        };
        controlDeps.dacManager = state_.managers.dacManager.get();
        controlDeps.upsampler = &state_.upsampler;
        controlDeps.crossfeed.processor = &state_.crossfeed.processor;
        controlDeps.crossfeed.enabledFlag = &state_.crossfeed.enabled;
        controlDeps.crossfeed.mutex = &state_.crossfeed.crossfeedMutex;
        controlDeps.crossfeed.resetStreamingState = [&state = state_]() {
            state.crossfeed.resetRequested.store(true, std::memory_order_release);
        };
        controlDeps.statsFilePath = statsFilePath_.c_str();

        controlPlane = std::make_unique<daemon_control::ControlPlane>(std::move(controlDeps));
        if (controlPlane && controlPlane->start() && state_.managers.dacManager) {
            state_.managers.dacManager->setEventPublisher(controlPlane->eventPublisher());
        }

        if (state_.managers.dacManager) {
            state_.managers.dacManager->initialize();
        }

        if (state_.managers.dacManager) {
            state_.managers.dacManager->start();
        }
        std::cout << "Starting ALSA output thread..." << '\n';
        std::thread alsa_thread(alsa_output_thread, std::ref(state_));

        std::thread loopback_thread;
        std::thread i2s_thread;
        bool startupFailed = false;

        auto failStartup = [&](const std::string& reason) {
            LOG_ERROR("Startup failed: {}", reason);
            exitCode = 1;
            startupFailed = true;
            state_.flags.running = false;
            playback_buffer(state_).cv().notify_all();
        };

        if (state_.config.i2s.enabled && state_.config.loopback.enabled) {
            failStartup("Config error: i2s.enabled and loopback.enabled cannot both be true");
        }

        if (!startupFailed && state_.config.i2s.enabled) {
            if (!validate_i2s_config(state_.config)) {
                failStartup("Invalid I2S config");
            } else {
                snd_pcm_format_t i2s_format = parse_i2s_format(state_.config.i2s.format);
                unsigned int i2s_rate =
                    (state_.config.i2s.sampleRate != 0)
                        ? static_cast<unsigned int>(state_.config.i2s.sampleRate)
                        : static_cast<unsigned int>(state_.rates.inputSampleRate);
                std::cout << "Starting I2S capture thread (" << state_.config.i2s.device
                          << ", fmt=" << state_.config.i2s.format << ", rate=" << i2s_rate
                          << ", period=" << state_.config.i2s.periodFrames << ")" << '\n';
                i2s_thread =
                    std::thread(i2s_capture_thread, std::ref(state_), state_.config.i2s.device,
                                i2s_format, i2s_rate, state_.config.i2s.channels,
                                static_cast<snd_pcm_uframes_t>(state_.config.i2s.periodFrames));

                bool i2s_ready = false;
                for (int i = 0; i < 40; ++i) {
                    if (state_.i2s.captureReady.load(std::memory_order_acquire)) {
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
        if (!startupFailed && state_.config.loopback.enabled) {
            if (!daemon_input::validateLoopbackConfig(state_.config)) {
                failStartup("Invalid loopback config");
            } else {
                snd_pcm_format_t lb_format =
                    daemon_input::parseLoopbackFormat(state_.config.loopback.format);
                std::cout << "Starting loopback capture thread (" << state_.config.loopback.device
                          << ", fmt=" << state_.config.loopback.format
                          << ", rate=" << state_.config.loopback.sampleRate
                          << ", period=" << state_.config.loopback.periodFrames << ")" << '\n';

                daemon_input::LoopbackCaptureDependencies loopbackDeps{
                    .playbackBuffer = &playback_buffer(state_),
                    .running = &state_.flags.running,
                    .currentOutputRate = &state_.rates.currentOutputRate,
                    .audioPipeline = &state_.managers.audioPipelineRaw,
                    .handleMutex = &state_.loopback.handleMutex,
                    .handle = &state_.loopback.handle,
                    .captureRunning = &state_.loopback.captureRunning,
                    .captureReady = &state_.loopback.captureReady,
                };

                loopback_thread = std::thread(
                    daemon_input::loopbackCaptureThread, state_.config.loopback.device, lb_format,
                    state_.config.loopback.sampleRate, state_.config.loopback.channels,
                    static_cast<snd_pcm_uframes_t>(state_.config.loopback.periodFrames),
                    loopbackDeps);

                bool loopback_ready = false;
                for (int i = 0; i < 40; ++i) {
                    if (state_.loopback.captureReady.load(std::memory_order_acquire)) {
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
                state_.rates.inputSampleRate * state_.config.upsampleRatio / 1000.0;
            std::cout << '\n';
            if (state_.config.loopback.enabled) {
                std::cout << "System ready (Loopback capture mode). Audio routing configured:"
                          << '\n';
                std::cout << "  1. Loopback capture -> GPU Upsampler ("
                          << state_.config.upsampleRatio << "x upsampling)" << '\n';
                std::cout << "  2. GPU Upsampler -> ALSA -> SMSL DAC (" << outputRateKHz
                          << "kHz direct)" << '\n';
            } else if (state_.config.i2s.enabled) {
                std::cout << "System ready (I2S capture mode). Audio routing configured:" << '\n';
                std::cout << "  1. I2S capture -> GPU Upsampler (" << state_.config.upsampleRatio
                          << "x upsampling)" << '\n';
                std::cout << "  2. GPU Upsampler -> ALSA -> DAC (" << outputRateKHz << "kHz direct)"
                          << '\n';
            } else {
                std::cout << "System ready (TCP/loopback input). Audio routing configured:" << '\n';
                std::cout << "  1. Network or loopback source -> GPU Upsampler ("
                          << state_.config.upsampleRatio << "x upsampling)" << '\n';
                std::cout << "  2. GPU Upsampler -> ALSA -> SMSL DAC (" << outputRateKHz
                          << "kHz direct)" << '\n';
            }
            std::cout << "Press Ctrl+C to stop." << '\n';
            std::cout << "========================================" << '\n';

            shutdownManager.notifyReady();

            auto runMainLoop = [&]() {
                while (state_.flags.running.load() && !state_.flags.reloadRequested.load() &&
                       !state_.flags.zmqBindFailed.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    if (state_.managers.streamingCacheManager) {
                        (void)state_.managers.streamingCacheManager->drainFlushRequests();
                    }
                    int pending =
                        state_.rates.pendingRateChange.exchange(0, std::memory_order_acq_rel);
                    if (pending > 0 && (pending == 44100 || pending == 48000)) {
                        (void)handle_rate_switch(state_, pending);
                    }
                    shutdownManager.tick();
                }
            };

            if (state_.flags.zmqBindFailed.load()) {
                std::cerr << "Startup aborted due to ZeroMQ bind failure." << '\n';
            } else {
                runMainLoop();
            }
        }

        shutdownManager.runShutdownSequence();

        std::cout << "  Step 5: Stopping worker threads..." << '\n';
        state_.flags.running = false;
        playback_buffer(state_).cv().notify_all();
        if (controlPlane) {
            controlPlane->stop();
        }
        alsa_thread.join();
        if (i2s_thread.joinable()) {
            i2s_thread.join();
        }
        if (loopback_thread.joinable()) {
            loopback_thread.join();
        }

        std::cout << "  Step 6: Releasing resources..." << '\n';
        if (state_.fallbackManager) {
            state_.fallbackManager->shutdown();
            delete state_.fallbackManager;
            state_.fallbackManager = nullptr;
            state_.fallbackActive.store(false, std::memory_order_relaxed);
        }
        delete state_.softMute.controller;
        state_.softMute.controller = nullptr;
        daemon_audio::shutdownCrossfeed(state_.crossfeed);
        if (state_.audioPipeline) {
            state_.audioPipeline.reset();
            state_.managers.audioPipelineRaw = nullptr;
        }
        delete state_.upsampler;
        state_.upsampler = nullptr;
        if (state_.managers.dacManager) {
            state_.managers.dacManager->stop();
        }

        if (state_.flags.zmqBindFailed.load()) {
            std::cerr << "Exiting due to ZeroMQ initialization failure." << '\n';
            exitCode = 1;
            break;
        }

        if (state_.flags.reloadRequested) {
            std::cout << "Reload requested. Restarting daemon with updated config..." << '\n';
        }
    } while (state_.flags.reloadRequested);

    std::cout << "Goodbye!" << '\n';
    return exitCode;
}

}  // namespace daemon_app
