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
#include "crossfeed_engine.h"
#include "daemon/api/dependencies.h"
#include "daemon/api/events.h"
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
static AppConfig g_config;
static std::atomic<float> g_output_gain{1.0f};
static std::atomic<float> g_headroom_gain{1.0f};
static FilterHeadroomCache g_headroom_cache;
static ConvolutionEngine::RateFamily g_active_rate_family = ConvolutionEngine::RateFamily::RATE_44K;
static PhaseType g_active_phase_type = PhaseType::Minimum;
static std::atomic<float> g_limiter_gain{1.0f};
static std::atomic<float> g_effective_gain{1.0f};

// Global state
static std::atomic<bool> g_running{true};
static std::atomic<bool> g_reload_requested{false};
static std::atomic<bool> g_zmq_bind_failed{false};  // True if ZeroMQ bind failed
static std::atomic<bool> g_output_ready{false};  // Indicates ALSA DAC is ready for input processing
static ConvolutionEngine::GPUUpsampler* g_upsampler = nullptr;
static std::mutex g_input_process_mutex;

static std::unique_ptr<audio_pipeline::AudioPipeline> g_audio_pipeline;

// Runtime state: Input sample rate (auto-negotiated, not from config)
// Detected from the active input path (TCP/loopback) or defaulted to 44.1 kHz
// Issue #219: Changed to atomic for thread-safe multi-rate switching
static std::atomic<int> g_current_input_rate{DEFAULT_INPUT_SAMPLE_RATE};
static std::atomic<int> g_current_output_rate{DEFAULT_OUTPUT_SAMPLE_RATE};
static std::atomic<int> g_current_rate_family_int{
    static_cast<int>(ConvolutionEngine::RateFamily::RATE_44K)};

static size_t get_max_output_buffer_frames() {
    using namespace DaemonConstants;
    auto seconds = static_cast<double>(MAX_OUTPUT_BUFFER_SECONDS);
    if (seconds <= 0.0) {
        return DEFAULT_MAX_OUTPUT_BUFFER_FRAMES;
    }

    int outputRate = g_current_output_rate.load(std::memory_order_acquire);
    if (outputRate <= 0) {
        outputRate = DEFAULT_OUTPUT_SAMPLE_RATE;
    }

    double frames = seconds * static_cast<double>(outputRate);
    if (frames <= 0.0) {
        return DEFAULT_MAX_OUTPUT_BUFFER_FRAMES;
    }
    return static_cast<size_t>(frames);
}

// Pending rate change (set by input event handlers, processed in main loop)
// Value: 0 = no change pending, >0 = detected input sample rate
static std::atomic<int> g_pending_rate_change{0};

// ALSA reconfiguration flags (set when output rate changes)
static std::atomic<bool> g_alsa_reconfigure_needed{false};
static std::atomic<int> g_alsa_new_output_rate{0};

// Helper to get/set rate family atomically
inline ConvolutionEngine::RateFamily g_get_rate_family() {
    return static_cast<ConvolutionEngine::RateFamily>(
        g_current_rate_family_int.load(std::memory_order_acquire));
}
inline void g_set_rate_family(ConvolutionEngine::RateFamily family) {
    g_current_rate_family_int.store(static_cast<int>(family), std::memory_order_release);
}

// Legacy alias (for backward compatibility with existing code)
static int g_input_sample_rate = DEFAULT_INPUT_SAMPLE_RATE;

// Soft Mute controller for glitch-free shutdown (50ms fade at output sample rate)
static SoftMute::Controller* g_soft_mute = nullptr;
static std::mutex g_soft_mute_op_mutex;
static std::atomic<bool> g_soft_mute_restore_pending{false};
static std::atomic<int> g_soft_mute_restore_fade_ms{0};
static std::atomic<int> g_soft_mute_restore_sample_rate{0};

static void maybe_restore_soft_mute_params() {
    if (!g_soft_mute) {
        return;
    }
    if (!g_soft_mute_restore_pending.load(std::memory_order_acquire)) {
        return;
    }
    SoftMute::MuteState st = g_soft_mute->getState();
    if (st != SoftMute::MuteState::PLAYING && st != SoftMute::MuteState::MUTED) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_soft_mute_op_mutex);
    if (!g_soft_mute) {
        return;
    }
    int fadeMs = g_soft_mute_restore_fade_ms.load(std::memory_order_relaxed);
    int sr = g_soft_mute_restore_sample_rate.load(std::memory_order_relaxed);
    g_soft_mute->setFadeDuration(fadeMs);
    g_soft_mute->setSampleRate(sr);
    g_soft_mute_restore_pending.store(false, std::memory_order_release);
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

    if (!g_soft_mute) {
        // If soft mute not initialized, perform switch without mute
        filterSwitchFunc();
        return;
    }

    std::lock_guard<std::mutex> lock(g_soft_mute_op_mutex);

    // Cancel any stale pending restore (new switch supersedes)
    g_soft_mute_restore_pending.store(false, std::memory_order_release);

    // Save current fade duration for restoration
    int originalFadeDuration = g_soft_mute->getFadeDuration();
    int outputSampleRate = g_soft_mute->getSampleRate();

    // Update fade duration for filter switching
    // Note: This is called from command thread, but audio thread may be processing.
    // The fade calculation will use the new duration from the next audio frame.
    g_soft_mute->setFadeDuration(FILTER_SWITCH_FADE_MS);
    g_soft_mute->setSampleRate(outputSampleRate);

    std::cout << "[Filter Switch] Starting fade-out (" << (FILTER_SWITCH_FADE_MS / 1000.0)
              << "s)..." << '\n';
    g_soft_mute->startFadeOut();

    // Wait briefly until mostly muted (or timeout) to avoid pops, but keep command responsive
    auto fade_start = std::chrono::steady_clock::now();
    const auto quick_timeout = std::chrono::milliseconds(250);
    while (true) {
        SoftMute::MuteState st = g_soft_mute->getState();
        float gain = g_soft_mute->getCurrentGain();
        if (st == SoftMute::MuteState::MUTED || gain < 0.05f) {
            break;
        }
        if (std::chrono::steady_clock::now() - fade_start > quick_timeout) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Perform filter switch while fade-out is progressing
    bool switch_success = filterSwitchFunc();

    if (switch_success) {
        // Start fade-in after filter switch
        std::cout << "[Filter Switch] Starting fade-in (" << (FILTER_SWITCH_FADE_MS / 1000.0)
                  << "s)..." << '\n';
        g_soft_mute->startFadeIn();

        // Mark pending restoration to be applied once transition completes
        g_soft_mute_restore_fade_ms.store(originalFadeDuration, std::memory_order_relaxed);
        g_soft_mute_restore_sample_rate.store(outputSampleRate, std::memory_order_relaxed);
        g_soft_mute_restore_pending.store(true, std::memory_order_release);
    } else {
        // If switch failed, restore original state immediately
        std::cerr << "[Filter Switch] Switch failed, restoring audio state" << '\n';
        g_soft_mute->setPlaying();
        g_soft_mute->setFadeDuration(originalFadeDuration);
        g_soft_mute->setSampleRate(outputSampleRate);
    }
}

// Audio buffer for thread communication (managed component)
static std::unique_ptr<daemon_output::PlaybackBufferManager> g_playback_buffer;

static daemon_output::PlaybackBufferManager& playback_buffer() {
    if (!g_playback_buffer) {
        g_playback_buffer = std::make_unique<daemon_output::PlaybackBufferManager>(
            []() { return get_max_output_buffer_frames(); });
    }
    return *g_playback_buffer;
}

// Streaming input accumulation buffers
static StreamFloatVector g_stream_input_left;
static StreamFloatVector g_stream_input_right;
static size_t g_stream_accumulated_left = 0;
static size_t g_stream_accumulated_right = 0;
static StreamFloatVector g_upsampler_output_left;
static StreamFloatVector g_upsampler_output_right;

// Crossfeed (HRTF) processor
static CrossfeedEngine::HRTFProcessor* g_hrtf_processor = nullptr;
static std::atomic<bool> g_crossfeed_enabled{false};  // Atomic for quick check
static std::vector<float> g_cf_stream_input_left;
static std::vector<float> g_cf_stream_input_right;
static size_t g_cf_stream_accumulated_left = 0;
static size_t g_cf_stream_accumulated_right = 0;
static std::vector<float> g_cf_output_left;
static std::vector<float> g_cf_output_right;
static std::mutex g_crossfeed_mutex;  // Protects HRTFProcessor and CF buffers
static std::mutex g_streaming_mutex;  // Protects streaming buffer state during rate switch
static std::vector<float> g_cf_output_buffer_left;
static std::vector<float> g_cf_output_buffer_right;

static std::unique_ptr<streaming_cache::StreamingCacheManager> g_streaming_cache_manager;

static std::unique_ptr<daemon_core::api::EventDispatcher> g_event_dispatcher;
static daemon_core::api::DaemonDependencies g_daemon_dependencies{
    .config = &g_config,
    .running = &g_running,
    .outputReady = &g_output_ready,
    .crossfeedEnabled = &g_crossfeed_enabled,
    .currentInputRate = &g_current_input_rate,
    .currentOutputRate = &g_current_output_rate,
    .softMute = &g_soft_mute,
    .upsampler = &g_upsampler,
    .audioPipeline = nullptr,
    .dacManager = nullptr,
    .streamingMutex = &g_streaming_mutex,
    .refreshHeadroom = refresh_current_headroom,
};

static audio_pipeline::AudioPipeline* g_audio_pipeline_raw = nullptr;
static dac::DacManager* g_dac_manager_raw = nullptr;

static std::unique_ptr<audio_pipeline::RateSwitcher> g_rate_switcher;
static std::unique_ptr<audio_pipeline::FilterManager> g_filter_manager;
static std::unique_ptr<audio_pipeline::SoftMuteRunner> g_soft_mute_runner;
static std::unique_ptr<daemon_output::AlsaOutput> g_alsa_output_interface;
static std::unique_ptr<daemon_control::handlers::HandlerRegistry> g_handler_registry;
static std::mutex g_loopback_handle_mutex;
static snd_pcm_t* g_loopback_handle = nullptr;
static std::atomic<bool> g_loopback_capture_running{false};
static std::atomic<bool> g_loopback_capture_ready{false};

static void update_daemon_dependencies() {
    g_daemon_dependencies.config = &g_config;
    g_daemon_dependencies.running = &g_running;
    g_daemon_dependencies.outputReady = &g_output_ready;
    g_daemon_dependencies.crossfeedEnabled = &g_crossfeed_enabled;
    g_daemon_dependencies.currentInputRate = &g_current_input_rate;
    g_daemon_dependencies.currentOutputRate = &g_current_output_rate;
    g_daemon_dependencies.softMute = &g_soft_mute;
    g_daemon_dependencies.upsampler = &g_upsampler;
    g_daemon_dependencies.audioPipeline = &g_audio_pipeline_raw;
    g_daemon_dependencies.dacManager = &g_dac_manager_raw;
    g_daemon_dependencies.streamingMutex = &g_streaming_mutex;
    g_daemon_dependencies.refreshHeadroom = [](const std::string& reason) {
        refresh_current_headroom(reason);
    };
}

static void initialize_event_modules() {
    g_event_dispatcher = std::make_unique<daemon_core::api::EventDispatcher>();
    update_daemon_dependencies();

    g_rate_switcher = std::make_unique<audio_pipeline::RateSwitcher>(
        audio_pipeline::RateSwitcherDependencies{.dispatcher = g_event_dispatcher.get(),
                                                 .deps = g_daemon_dependencies,
                                                 .pendingRate = &g_pending_rate_change});
    g_filter_manager =
        std::make_unique<audio_pipeline::FilterManager>(audio_pipeline::FilterManagerDependencies{
            .dispatcher = g_event_dispatcher.get(), .deps = g_daemon_dependencies});
    g_soft_mute_runner =
        std::make_unique<audio_pipeline::SoftMuteRunner>(audio_pipeline::SoftMuteRunnerDependencies{
            .dispatcher = g_event_dispatcher.get(), .deps = g_daemon_dependencies});
    g_alsa_output_interface =
        std::make_unique<daemon_output::AlsaOutput>(daemon_output::AlsaOutputDependencies{
            .dispatcher = g_event_dispatcher.get(), .deps = g_daemon_dependencies});
    g_handler_registry = std::make_unique<daemon_control::handlers::HandlerRegistry>(
        daemon_control::handlers::HandlerRegistryDependencies{.dispatcher =
                                                                  g_event_dispatcher.get()});

    g_rate_switcher->start();
    g_filter_manager->start();
    g_soft_mute_runner->start();
    g_alsa_output_interface->start();
    g_handler_registry->registerDefaults();
}

static void publish_rate_change_event(int detected_rate) {
    if (!g_rate_switcher || !g_event_dispatcher) {
        return;
    }
    daemon_core::api::RateChangeRequested event;
    event.detectedInputRate = detected_rate;
    event.rateFamily = ConvolutionEngine::detectRateFamily(detected_rate);
    g_event_dispatcher->publish(event);
}

static void publish_filter_switch_event(const std::string& filterPath, PhaseType phaseType,
                                        bool reloadHeadroom) {
    daemon_core::api::FilterSwitchRequested event;
    event.filterPath = filterPath;
    event.phaseType = phaseType;
    event.reloadHeadroom = reloadHeadroom;
    if (g_event_dispatcher) {
        g_event_dispatcher->publish(event);
    }
}

static void initialize_streaming_cache_manager();

static void elevate_realtime_priority(const char* name, int priority = 65) {
#ifdef __linux__
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

    if (g_crossfeed_enabled.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        if (g_hrtf_processor && g_hrtf_processor->isEnabled()) {
            crossfeedActive = true;
            crossfeedBlockSize = g_hrtf_processor->getStreamValidInputPerBlock();
        }
    }

    if (g_upsampler) {
        // StreamValidInputPerBlock() is in input frames. Multiply by upsample ratio to obtain the
        // number of samples the producer actually contributes to the playback ring per block so the
        // ALSA thread can wake up as soon as a full GPU block finishes.
        size_t streamBlock = g_upsampler->getStreamValidInputPerBlock();
        int upsampleRatio = g_upsampler->getUpsampleRatio();
        if (streamBlock > 0 && upsampleRatio > 0) {
            producerBlockSize = streamBlock * static_cast<size_t>(upsampleRatio);
        }
    }

    return PlaybackBuffer::computeReadyThreshold(period_size, crossfeedActive, crossfeedBlockSize,
                                                 producerBlockSize);
}

// Fallback manager (Issue #139)
static FallbackManager::Manager* g_fallback_manager = nullptr;
static std::atomic<bool> g_fallback_active{false};  // Atomic for quick check in audio thread

// Resets crossfeed streaming buffers and GPU overlap state.
// Caller must hold g_crossfeed_mutex.
static void reset_crossfeed_stream_state_locked() {
    g_cf_stream_input_left.clear();
    g_cf_stream_input_right.clear();
    g_cf_stream_accumulated_left = 0;
    g_cf_stream_accumulated_right = 0;
    g_cf_output_left.clear();
    g_cf_output_right.clear();
    if (g_hrtf_processor) {
        g_hrtf_processor->resetStreaming();
    }
}

static void initialize_streaming_cache_manager() {
    streaming_cache::StreamingCacheDependencies deps;
    deps.inputMutex = &g_input_process_mutex;
    deps.resetPlaybackBuffer = []() { playback_buffer().reset(); };

    deps.streamInputLeft = &g_stream_input_left;
    deps.streamInputRight = &g_stream_input_right;
    deps.streamAccumulatedLeft = &g_stream_accumulated_left;
    deps.streamAccumulatedRight = &g_stream_accumulated_right;
    deps.streamingMutex = &g_streaming_mutex;

    deps.upsamplerPtr = &g_upsampler;
    deps.softMute = &g_soft_mute;
    deps.onCrossfeedReset = []() {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        reset_crossfeed_stream_state_locked();
    };
    g_streaming_cache_manager = std::make_unique<streaming_cache::StreamingCacheManager>(deps);
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
    deps.config = &g_config;
    deps.runningFlag = &g_running;
    deps.timestampProvider = get_timestamp_ms;
    deps.eventPublisher = std::move(eventPublisher);
    deps.defaultDevice = DEFAULT_ALSA_DEVICE;
    return deps;
}

static std::unique_ptr<dac::DacManager> g_dac_manager;

static runtime_stats::Dependencies build_runtime_stats_dependencies() {
    runtime_stats::Dependencies deps;
    deps.config = &g_config;
    deps.upsampler = g_upsampler;
    deps.headroomCache = &g_headroom_cache;
    deps.dacManager = g_dac_manager.get();
    deps.fallbackManager = g_fallback_manager;
    deps.fallbackActive = &g_fallback_active;
    deps.inputSampleRate = &g_input_sample_rate;
    deps.headroomGain = &g_headroom_gain;
    deps.outputGain = &g_output_gain;
    deps.limiterGain = &g_limiter_gain;
    deps.effectiveGain = &g_effective_gain;
    return deps;
}

// ========== PID File Lock (flock-based) ==========

// File descriptor for the PID lock file (kept open while running)
static int g_pid_lock_fd = -1;

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
    g_pid_lock_fd = open(PID_FILE_PATH, O_RDWR | O_CREAT, 0644);
    if (g_pid_lock_fd < 0) {
        LOG_ERROR("Cannot open PID file: {} ({})", PID_FILE_PATH, strerror(errno));
        return false;
    }

    // Try to acquire exclusive lock (non-blocking)
    if (flock(g_pid_lock_fd, LOCK_EX | LOCK_NB) < 0) {
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
        close(g_pid_lock_fd);
        g_pid_lock_fd = -1;
        return false;
    }

    // Lock acquired - write our PID to the file
    if (ftruncate(g_pid_lock_fd, 0) < 0) {
        LOG_WARN("Cannot truncate PID file");
    }
    dprintf(g_pid_lock_fd, "%d\n", getpid());
    fsync(g_pid_lock_fd);  // Ensure PID is written to disk

    return true;
}

// Release PID lock and remove file
// Note: Lock is automatically released when process exits (even on crash)
static void release_pid_lock() {
    if (g_pid_lock_fd >= 0) {
        flock(g_pid_lock_fd, LOCK_UN);
        close(g_pid_lock_fd);
        g_pid_lock_fd = -1;
    }
    // Remove the PID file on clean shutdown
    unlink(PID_FILE_PATH);
    // Remove the stats file on clean shutdown
    unlink(STATS_FILE_PATH);
}

// ========== Configuration ==========

static void print_config_summary(const AppConfig& cfg) {
    int outputRate = g_input_sample_rate * cfg.upsampleRatio;
    LOG_INFO("Config:");
    LOG_INFO("  ALSA device:    {}", cfg.alsaDevice);
    LOG_INFO("  Output mode:    {} (preferred USB device: {})", outputModeToString(cfg.output.mode),
             cfg.output.usb.preferredDevice);
    LOG_INFO("  Loopback:       {} device={} rate={} ch={} fmt={} period={}",
             (cfg.loopback.enabled ? "enabled" : "disabled"), cfg.loopback.device,
             cfg.loopback.sampleRate, cfg.loopback.channels, cfg.loopback.format,
             cfg.loopback.periodFrames);
    LOG_INFO("  Input rate:     {} Hz (auto-negotiated)", g_input_sample_rate);
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
    gpu_upsampler::metrics::setAudioConfig(g_input_sample_rate, outputRate, cfg.upsampleRatio);
}

static std::string resolve_filter_path_for(ConvolutionEngine::RateFamily family, PhaseType phase) {
    if (phase == PhaseType::Linear) {
        return (family == ConvolutionEngine::RateFamily::RATE_44K) ? g_config.filterPath44kLinear
                                                                   : g_config.filterPath48kLinear;
    }
    return (family == ConvolutionEngine::RateFamily::RATE_44K) ? g_config.filterPath44kMin
                                                               : g_config.filterPath48kMin;
}

static std::string current_filter_path() {
    return resolve_filter_path_for(g_active_rate_family, g_active_phase_type);
}

static void update_effective_gain(float headroomGain, const std::string& reason) {
    g_headroom_gain.store(headroomGain, std::memory_order_relaxed);
    float effective = g_config.gain * headroomGain;
    g_output_gain.store(effective, std::memory_order_relaxed);
    LOG_INFO("Gain [{}]: user {:.4f} * headroom {:.4f} = {:.4f}", reason, g_config.gain,
             headroomGain, effective);
}

static void apply_headroom_for_path(const std::string& path, const std::string& reason) {
    if (path.empty()) {
        LOG_WARN("Headroom [{}]: empty filter path, falling back to unity gain", reason);
        update_effective_gain(1.0f, reason);
        return;
    }

    FilterHeadroomInfo info = g_headroom_cache.get(path);
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
    g_streaming_cache_manager.reset();

    playback_buffer().reset();
    g_stream_input_left.clear();
    g_stream_input_right.clear();
    g_stream_accumulated_left = 0;
    g_stream_accumulated_right = 0;
    g_upsampler_output_left.clear();
    g_upsampler_output_right.clear();

    // Reset crossfeed streaming buffers
    g_cf_stream_input_left.clear();
    g_cf_stream_input_right.clear();
    g_cf_stream_accumulated_left = 0;
    g_cf_stream_accumulated_right = 0;
    g_cf_output_buffer_left.clear();
    g_cf_output_buffer_right.clear();
}

static bool reinitialize_streaming_for_legacy_mode() {
    if (!g_upsampler) {
        return false;
    }

    std::lock_guard<std::mutex> streamLock(g_streaming_mutex);
    g_upsampler->resetStreaming();
    playback_buffer().reset();

    g_stream_input_left.clear();
    g_stream_input_right.clear();
    g_stream_accumulated_left = 0;
    g_stream_accumulated_right = 0;
    g_upsampler_output_left.clear();
    g_upsampler_output_right.clear();

    // Rebuild legacy streams so the buffers match the full FFT (avoids invalid cudaMemset after
    // disabling partitions).
    if (!g_upsampler->initializeStreaming()) {
        std::cerr << "[Partition] Failed to initialize legacy streaming buffers" << '\n';
        return false;
    }

    size_t buffer_capacity = g_upsampler->getStreamValidInputPerBlock() * 2;
    if (buffer_capacity > 0) {
        g_stream_input_left.resize(buffer_capacity, 0.0f);
        g_stream_input_right.resize(buffer_capacity, 0.0f);
    } else {
        g_stream_input_left.clear();
        g_stream_input_right.clear();
    }
    g_stream_accumulated_left = 0;
    g_stream_accumulated_right = 0;

    size_t upsampler_output_capacity =
        g_upsampler->getStreamValidInputPerBlock() * g_upsampler->getUpsampleRatio() * 2;
    g_upsampler_output_left.reserve(upsampler_output_capacity);
    g_upsampler_output_right.reserve(upsampler_output_capacity);

    return true;
}

static bool handle_rate_switch(int newInputRate) {
    if (!g_upsampler || !g_upsampler->isMultiRateEnabled()) {
        std::cerr << "[Rate] Multi-rate mode not enabled" << '\n';
        return false;
    }

    int currentRate = g_upsampler->getCurrentInputRate();
    if (currentRate == newInputRate) {
        std::cout << "[Rate] Already at target rate: " << newInputRate << " Hz" << '\n';
        return true;
    }

    std::cout << "[Rate] Switching: " << currentRate << " Hz -> " << newInputRate << " Hz" << '\n';

    int savedRate = currentRate;

    if (g_soft_mute) {
        g_soft_mute->startFadeOut();
        auto startTime = std::chrono::steady_clock::now();
        while (g_soft_mute->isTransitioning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto elapsed = std::chrono::steady_clock::now() - startTime;
            if (elapsed > std::chrono::milliseconds(200)) {
                std::cerr << "[Rate] Warning: Fade-out timeout" << '\n';
                break;
            }
        }
    }

    int newOutputRate = g_upsampler->getOutputSampleRate();
    int newUpsampleRatio = g_upsampler->getUpsampleRatio();
    size_t buffer_capacity = 0;

    {
        std::lock_guard<std::mutex> streamLock(g_streaming_mutex);

        g_upsampler->resetStreaming();

        playback_buffer().reset();
        g_stream_input_left.clear();
        g_stream_input_right.clear();
        g_stream_accumulated_left = 0;
        g_stream_accumulated_right = 0;

        if (!g_upsampler->switchToInputRate(newInputRate)) {
            std::cerr << "[Rate] Failed to switch rate, rolling back" << '\n';
            if (g_upsampler->switchToInputRate(savedRate)) {
                std::cout << "[Rate] Rollback successful: restored to " << savedRate << " Hz"
                          << '\n';
            } else {
                std::cerr << "[Rate] ERROR: Rollback failed!" << '\n';
            }
            if (g_soft_mute) {
                g_soft_mute->startFadeIn();
            }
            return false;
        }

        if (!g_upsampler->initializeStreaming()) {
            std::cerr << "[Rate] Failed to re-initialize streaming mode, rolling back" << '\n';
            if (g_upsampler->switchToInputRate(savedRate)) {
                if (g_upsampler->initializeStreaming()) {
                    std::cout << "[Rate] Rollback successful: restored to " << savedRate << " Hz"
                              << '\n';
                }
            }
            if (g_soft_mute) {
                g_soft_mute->startFadeIn();
            }
            return false;
        }

        g_input_sample_rate = newInputRate;
        newOutputRate = g_upsampler->getOutputSampleRate();
        newUpsampleRatio = g_upsampler->getUpsampleRatio();

        buffer_capacity = g_upsampler->getStreamValidInputPerBlock() * 2;
        g_stream_input_left.resize(buffer_capacity, 0.0f);
        g_stream_input_right.resize(buffer_capacity, 0.0f);
        g_stream_accumulated_left = 0;
        g_stream_accumulated_right = 0;

        if (g_soft_mute) {
            delete g_soft_mute;
        }
        g_soft_mute = new SoftMute::Controller(50, newOutputRate);
    }

    if (g_soft_mute) {
        g_soft_mute->startFadeIn();
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
    g_config = loaded;
    ensure_output_config(g_config);

    if (g_config.alsaDevice.empty()) {
        set_preferred_output_device(g_config, DEFAULT_ALSA_DEVICE);
    }
    if (g_config.filterPath.empty()) {
        g_config.filterPath = DEFAULT_FILTER_PATH;
    }
    if (g_config.upsampleRatio <= 0) {
        g_config.upsampleRatio = DEFAULT_UPSAMPLE_RATIO;
    }
    if (g_config.blockSize <= 0) {
        g_config.blockSize = DEFAULT_BLOCK_SIZE;
    }
    if (g_config.bufferSize <= 0) {
        g_config.bufferSize = 262144;
    }
    if (g_config.periodSize <= 0) {
        g_config.periodSize = 32768;
    }
    if (g_config.loopback.device.empty()) {
        g_config.loopback.device = DEFAULT_LOOPBACK_DEVICE;
    }
    if (g_config.loopback.sampleRate == 0) {
        g_config.loopback.sampleRate = DEFAULT_INPUT_SAMPLE_RATE;
    }
    if (g_config.loopback.channels == 0) {
        g_config.loopback.channels = CHANNELS;
    }
    if (g_config.loopback.periodFrames == 0) {
        g_config.loopback.periodFrames = DEFAULT_LOOPBACK_PERIOD_FRAMES;
    }
    if (g_config.loopback.format.empty()) {
        g_config.loopback.format = "S16_LE";
    }
    if (g_config.loopback.enabled) {
        g_input_sample_rate = static_cast<int>(g_config.loopback.sampleRate);
    }
    if (g_input_sample_rate != 44100 && g_input_sample_rate != 48000) {
        g_input_sample_rate = DEFAULT_INPUT_SAMPLE_RATE;
    }

    if (!found) {
        std::cout << "Config: Using defaults (no config.json found)" << '\n';
    }

    enforce_phase_partition_constraints(g_config);

    print_config_summary(g_config);
    g_headroom_cache.setTargetPeak(g_config.headroomTarget);
    update_effective_gain(1.0f, "config load (pending filter headroom)");
    float initialOutput = g_output_gain.load(std::memory_order_relaxed);
    g_limiter_gain.store(1.0f, std::memory_order_relaxed);
    g_effective_gain.store(initialOutput, std::memory_order_relaxed);
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

static snd_pcm_t* open_loopback_capture(const std::string& device, snd_pcm_format_t format,
                                        unsigned int rate, unsigned int channels,
                                        snd_pcm_uframes_t period_frames) {
    snd_pcm_t* handle = nullptr;
    int err = snd_pcm_open(&handle, device.c_str(), SND_PCM_STREAM_CAPTURE, 0);
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

    snd_pcm_uframes_t buffer_frames =
        static_cast<snd_pcm_uframes_t>(std::max<uint32_t>(period_frames * 4, period_frames));
    snd_pcm_hw_params_set_period_size_near(handle, hw_params, &period_frames, nullptr);
    snd_pcm_hw_params_set_buffer_size_near(handle, hw_params, &buffer_frames);

    if ((err = snd_pcm_hw_params(handle, hw_params)) < 0) {
        LOG_ERROR("[Loopback] Cannot apply hardware parameters: {}", snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }

    if ((err = snd_pcm_prepare(handle)) < 0) {
        LOG_ERROR("[Loopback] Cannot prepare capture device: {}", snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }

    LOG_INFO("[Loopback] Capture device {} configured ({} Hz, {} ch, period {} frames)", device,
             rate_near, channels, period_frames);
    return handle;
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

static void loopback_capture_thread(const std::string& device, snd_pcm_format_t format,
                                    unsigned int rate, unsigned int channels,
                                    snd_pcm_uframes_t period_frames) {
    if (channels != static_cast<unsigned int>(CHANNELS)) {
        LOG_ERROR("[Loopback] Unsupported channel count {} (expected {})", channels, CHANNELS);
        return;
    }

    snd_pcm_t* handle = open_loopback_capture(device, format, rate, channels, period_frames);
    {
        std::lock_guard<std::mutex> lock(g_loopback_handle_mutex);
        g_loopback_handle = handle;
    }

    if (!handle) {
        return;
    }

    g_loopback_capture_running.store(true, std::memory_order_release);
    g_loopback_capture_ready.store(true, std::memory_order_release);

    std::vector<int16_t> buffer16;
    std::vector<int32_t> buffer32;
    std::vector<uint8_t> buffer24;
    if (format == SND_PCM_FORMAT_S16_LE) {
        buffer16.resize(period_frames * channels);
    } else if (format == SND_PCM_FORMAT_S32_LE) {
        buffer32.resize(period_frames * channels);
    } else if (format == SND_PCM_FORMAT_S24_3LE) {
        buffer24.resize(static_cast<size_t>(period_frames) * channels * 3);
    }
    std::vector<float> floatBuffer;

    while (g_running.load(std::memory_order_acquire)) {
        // Apply backpressure before pulling more input.
        // This prevents software loopback / bursty upstream from running ahead of ALSA playback.
        playback_buffer().throttleProducerIfFull(
            g_running, []() { return g_current_output_rate.load(std::memory_order_acquire); });

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

        snd_pcm_sframes_t frames = snd_pcm_readi(handle, rawBuffer, period_frames);
        if (frames == -EAGAIN) {
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
            continue;
        }

        const void* src = rawBuffer;
        if (!convert_pcm_to_float(src, format, static_cast<size_t>(frames), channels,
                                  floatBuffer)) {
            LOG_ERROR("[Loopback] Unsupported format during conversion");
            break;
        }

        if (g_audio_pipeline) {
            g_audio_pipeline->process(floatBuffer.data(), static_cast<uint32_t>(frames));
        }
    }

    snd_pcm_drop(handle);
    snd_pcm_close(handle);
    {
        std::lock_guard<std::mutex> lock(g_loopback_handle_mutex);
        g_loopback_handle = nullptr;
    }
    g_loopback_capture_running.store(false, std::memory_order_release);
    g_loopback_capture_ready.store(false, std::memory_order_release);
    LOG_INFO("[Loopback] Capture thread terminated");
}

// ALSA output thread (705.6kHz direct to DAC)
void alsa_output_thread() {
    elevate_realtime_priority("ALSA output");

    daemon_output::AlsaPcmController pcmController(daemon_output::AlsaPcmControllerDependencies{
        .config = &g_config,
        .dacManager = g_dac_manager.get(),
        .streamingCacheManager = g_streaming_cache_manager.get(),
        .fallbackManager = g_fallback_manager,
        .running = &g_running,
        .outputReady = &g_output_ready,
        .currentOutputRate = []() { return g_current_output_rate.load(std::memory_order_acquire); },
    });

    if (!pcmController.openSelected()) {
        return;
    }

    auto period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
    if (period_size == 0) {
        period_size =
            static_cast<snd_pcm_uframes_t>((g_config.periodSize > 0) ? g_config.periodSize : 32768);
    }
    std::vector<int32_t> interleaved_buffer(period_size * CHANNELS);
    std::vector<float> float_buffer(period_size * CHANNELS);  // for soft mute processing
    auto& bufferManager = playback_buffer();

    // Main playback loop
    while (g_running) {
        // Heartbeat check every few hundred loops
        static int alive_counter = 0;
        if (++alive_counter > 200) {  // ~200 iterations ~ a few seconds depending on buffer wait
            alive_counter = 0;
            if (!pcmController.alive()) {
                std::cerr << "ALSA: PCM disconnected/suspended, attempting reopen..." << '\n';
                pcmController.close();
                while (g_running && !pcmController.openSelected()) {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                }
                period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                if (period_size == 0) {
                    period_size = static_cast<snd_pcm_uframes_t>(
                        (g_config.periodSize > 0) ? g_config.periodSize : 32768);
                }
                interleaved_buffer.resize(period_size * CHANNELS);
                float_buffer.resize(period_size * CHANNELS);
                continue;
            }
        }

        // Issue #219: Check for pending ALSA reconfiguration (output rate changed)
        if (g_alsa_reconfigure_needed.exchange(false, std::memory_order_acquire)) {
            int new_output_rate = g_alsa_new_output_rate.load(std::memory_order_acquire);
            if (new_output_rate > 0) {
                LOG_INFO("[Main] Reconfiguring ALSA for new output rate {} Hz", new_output_rate);

                // Reconfigure ALSA with new rate
                if (pcmController.reconfigure(new_output_rate)) {
                    period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                    if (period_size == 0) {
                        period_size = static_cast<snd_pcm_uframes_t>(
                            (g_config.periodSize > 0) ? g_config.periodSize : 32768);
                    }
                    interleaved_buffer.resize(period_size * CHANNELS);
                    float_buffer.resize(period_size * CHANNELS);

                    // Update soft mute sample rate
                    if (g_soft_mute) {
                        g_soft_mute->setSampleRate(new_output_rate);
                    }

                    LOG_INFO("[Main] ALSA reconfiguration successful");
                } else {
                    // Failed to reconfigure - try to reopen with old rate
                    LOG_ERROR("[Main] ALSA reconfiguration failed, attempting recovery...");
                    int old_rate = g_current_output_rate.load(std::memory_order_acquire);
                    if (!pcmController.reconfigure(old_rate)) {
                        LOG_ERROR("[Main] ALSA recovery failed, waiting for reconnect...");
                    }
                }
            }
        }

        if (auto pendingDevice = g_dac_manager->consumePendingChange()) {
            std::string nextDevice = *pendingDevice;
            if (!nextDevice.empty() && nextDevice != pcmController.device()) {
                std::cout << "ALSA: Switching output to " << nextDevice << '\n';
                if (!pcmController.switchDevice(nextDevice)) {
                    continue;
                }
                period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                if (period_size == 0) {
                    period_size = static_cast<snd_pcm_uframes_t>(
                        (g_config.periodSize > 0) ? g_config.periodSize : 32768);
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
                return bufferManager.queuedFramesLocked() >= ready_threshold || !g_running;
            });

        if (!g_running) {
            break;
        }

        lock.unlock();

        audio_pipeline::RenderResult renderResult;
        if (g_audio_pipeline) {
            renderResult = g_audio_pipeline->renderOutput(
                static_cast<size_t>(period_size), interleaved_buffer, float_buffer, g_soft_mute);
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

        static auto last_stats_write = std::chrono::steady_clock::now();
        if (!renderResult.wroteSilence && renderResult.framesRendered > 0) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_stats_write >= std::chrono::seconds(1)) {
                runtime_stats::writeStatsFile(build_runtime_stats_dependencies(),
                                              get_max_output_buffer_frames(), STATS_FILE_PATH);
                last_stats_write = now;
            }
            size_t total = runtime_stats::totalSamples();
            size_t clips = runtime_stats::clipCount();
            if (total % (static_cast<size_t>(period_size) * 2 * 100) == 0 && clips > 0) {
                std::cout << "WARNING: Clipping detected - " << clips << " samples clipped out of "
                          << total << " (" << (100.0 * clips / total) << "%)" << '\n';
            }
        }

        // Write to ALSA device
        long frames_written = pcmController.writeInterleaved(interleaved_buffer.data(),
                                                             static_cast<size_t>(period_size));
        if (frames_written < 0) {
            // Device may be gone; attempt reopen
            std::cerr << "ALSA: Write error: " << snd_strerror(frames_written)
                      << ", retrying reopen..." << '\n';
            pcmController.close();
            while (g_running && !pcmController.openSelected()) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
            if (!pcmController.isOpen()) {
                continue;
            }
            period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
            if (period_size == 0) {
                period_size = static_cast<snd_pcm_uframes_t>(
                    (g_config.periodSize > 0) ? g_config.periodSize : 32768);
            }
            interleaved_buffer.resize(period_size * CHANNELS);
            float_buffer.resize(period_size * CHANNELS);
        }
    }

    // Cleanup
    pcmController.close();
    std::cout << "ALSA: Output thread terminated" << '\n';
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

    shutdown_manager::ShutdownManager::Dependencies shutdownDeps{&g_soft_mute, &g_running,
                                                                 &g_reload_requested, nullptr};
    shutdown_manager::ShutdownManager shutdownManager(shutdownDeps);
    shutdownManager.installSignalHandlers();

    int exitCode = 0;

    do {
        shutdownManager.reset();
        g_running = true;
        g_reload_requested = false;
        g_zmq_bind_failed.store(false);
        reset_runtime_state();

        // Load configuration from config.json (if exists)
        load_runtime_config();

        // Environment variable overrides config.json
        if (const char* env_dev = std::getenv("ALSA_DEVICE")) {
            set_preferred_output_device(g_config, env_dev);
            std::cout << "Config: ALSA_DEVICE env override: " << env_dev << '\n';
        }

        // Command line argument overrides filter path
        if (argc > 1) {
            g_config.filterPath = argv[1];
            std::cout << "Config: CLI filter path override: " << argv[1] << '\n';
        }

        initialize_event_modules();

        PartitionRuntime::RuntimeRequest partitionRequest{g_config.partitionedConvolution.enabled,
                                                          g_config.eqEnabled,
                                                          g_config.crossfeed.enabled};

        // Auto-select filter based on sample rate if configured filter doesn't exist
        // but a sample-rate-specific version does
        if (!std::filesystem::exists(g_config.filterPath)) {
            // Try to find sample-rate-specific filter
            std::string basePath = g_config.filterPath;
            size_t dotPos = basePath.rfind('.');
            if (dotPos != std::string::npos) {
                std::string rateSpecificPath = basePath.substr(0, dotPos) + "_" +
                                               std::to_string(g_input_sample_rate) +
                                               basePath.substr(dotPos);
                if (std::filesystem::exists(rateSpecificPath)) {
                    std::cout << "Config: Using sample-rate-specific filter: " << rateSpecificPath
                              << '\n';
                    g_config.filterPath = rateSpecificPath;
                }
            }
        }

        // Warn if using 44.1kHz filter with 48kHz input
        if (g_input_sample_rate == 48000 &&
            g_config.filterPath.find("44100") == std::string::npos &&
            g_config.filterPath.find("48000") == std::string::npos) {
            std::cout << "Warning: Using generic filter with 48kHz input. "
                      << "For optimal quality, generate a 48kHz-optimized filter." << '\n';
        }

        // Initialize GPU upsampler with configured values
        std::cout << "Initializing GPU upsampler..." << '\n';
        g_upsampler = new ConvolutionEngine::GPUUpsampler();
        g_upsampler->setPartitionedConvolutionConfig(g_config.partitionedConvolution);

        bool initSuccess = false;
        ConvolutionEngine::RateFamily initialFamily = ConvolutionEngine::RateFamily::RATE_44K;
        if (g_config.multiRateEnabled) {
            std::cout << "Multi-rate mode enabled" << '\n';
            std::cout << "  Coefficient directory: " << g_config.coefficientDir << '\n';

            if (!std::filesystem::exists(g_config.coefficientDir)) {
                std::cerr << "Config error: Coefficient directory not found: "
                          << g_config.coefficientDir << '\n';
                delete g_upsampler;
                exitCode = 1;
                break;
            }

            initSuccess = g_upsampler->initializeMultiRate(g_config.coefficientDir,
                                                           g_config.blockSize, g_input_sample_rate);

            if (initSuccess) {
                g_current_input_rate.store(g_upsampler->getInputSampleRate(),
                                           std::memory_order_release);
                g_current_output_rate.store(g_upsampler->getOutputSampleRate(),
                                            std::memory_order_release);
                g_set_rate_family(ConvolutionEngine::detectRateFamily(g_input_sample_rate));
            }
        } else {
            std::cout << "Quad-phase mode enabled" << '\n';

            bool allFilesExist = true;
            for (const auto& path : {g_config.filterPath44kMin, g_config.filterPath48kMin,
                                     g_config.filterPath44kLinear, g_config.filterPath48kLinear}) {
                if (!std::filesystem::exists(path)) {
                    std::cerr << "Config error: Filter file not found: " << path << '\n';
                    allFilesExist = false;
                }
            }
            if (!allFilesExist) {
                delete g_upsampler;
                exitCode = 1;
                break;
            }

            initialFamily = ConvolutionEngine::detectRateFamily(g_input_sample_rate);
            if (initialFamily == ConvolutionEngine::RateFamily::RATE_UNKNOWN) {
                initialFamily = ConvolutionEngine::RateFamily::RATE_44K;
            }

            initSuccess = g_upsampler->initializeQuadPhase(
                g_config.filterPath44kMin, g_config.filterPath48kMin, g_config.filterPath44kLinear,
                g_config.filterPath48kLinear, g_config.upsampleRatio, g_config.blockSize,
                initialFamily, g_config.phaseType);
        }

        if (!initSuccess) {
            std::cerr << "Failed to initialize GPU upsampler" << '\n';
            delete g_upsampler;
            exitCode = 1;
            break;
        }

        // Check for early abort (signal received during GPU initialization)
        if (!g_running) {
            std::cout << "Startup interrupted by signal" << '\n';
            delete g_upsampler;
            g_upsampler = nullptr;
            break;
        }

        if (g_config.multiRateEnabled) {
            std::cout << "GPU upsampler ready (multi-rate mode, " << g_config.blockSize
                      << " samples/block)" << '\n';
            std::cout << "  Current input rate: " << g_upsampler->getCurrentInputRate() << " Hz"
                      << '\n';
            std::cout << "  Upsample ratio: " << g_upsampler->getUpsampleRatio() << "x" << '\n';
            std::cout << "  Output rate: " << g_upsampler->getOutputSampleRate() << " Hz" << '\n';
        } else {
            std::cout << "GPU upsampler ready (" << g_config.upsampleRatio << "x upsampling, "
                      << g_config.blockSize << " samples/block)" << '\n';
        }

        // Set g_active_rate_family and g_active_phase_type for headroom tracking
        if (g_config.multiRateEnabled) {
            // Rate family already set during initializeMultiRate()
            // g_active_rate_family is set via g_set_rate_family() above
        } else {
            g_active_rate_family = initialFamily;
        }

        g_active_phase_type = g_config.phaseType;
        publish_filter_switch_event(current_filter_path(), g_active_phase_type, true);

        std::cout << "Input sample rate: " << g_upsampler->getInputSampleRate() << " Hz -> "
                  << g_upsampler->getOutputSampleRate() << " Hz output" << '\n';
        if (!g_config.multiRateEnabled) {
            std::cout << "Phase type: " << phaseTypeToString(g_config.phaseType) << '\n';
        }

        // Log latency warning for linear phase
        if (g_config.phaseType == PhaseType::Linear) {
            double latencySec = g_upsampler->getLatencySeconds();
            std::cout << "  WARNING: Linear phase latency: " << latencySec << " seconds ("
                      << g_upsampler->getLatencySamples() << " samples)" << '\n';
        }

        // Initialize streaming mode to preserve overlap buffers across input callbacks
        if (!g_upsampler->initializeStreaming()) {
            std::cerr << "Failed to initialize streaming mode" << '\n';
            delete g_upsampler;
            exitCode = 1;
            break;
        }
        PartitionRuntime::applyPartitionPolicy(partitionRequest, *g_upsampler, g_config, "ALSA");

        // Check for early abort
        if (!g_running) {
            std::cout << "Startup interrupted by signal" << '\n';
            delete g_upsampler;
            g_upsampler = nullptr;
            break;
        }

        // Apply EQ profile if enabled
        if (g_config.eqEnabled && !g_config.eqProfilePath.empty()) {
            std::cout << "Loading EQ profile: " << g_config.eqProfilePath << '\n';
            EQ::EqProfile eqProfile;
            if (EQ::parseEqFile(g_config.eqProfilePath, eqProfile)) {
                std::cout << "  EQ: " << eqProfile.name << " (" << eqProfile.bands.size()
                          << " bands, preamp " << eqProfile.preampDb << " dB)" << '\n';

                // Compute EQ magnitude response and apply with minimum phase reconstruction
                size_t filterFftSize = g_upsampler->getFilterFftSize();  // N/2+1 (R2C output)
                size_t fullFftSize = g_upsampler->getFullFftSize();      // N (full FFT)
                double outputSampleRate =
                    static_cast<double>(g_input_sample_rate) * g_config.upsampleRatio;
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

                if (g_upsampler->applyEqMagnitude(eqMagnitude)) {
                    // Log message depends on phase type (already logged by applyEqMagnitude)
                } else {
                    std::cerr << "  EQ: Failed to apply frequency response" << '\n';
                }
            } else {
                std::cerr << "  EQ: Failed to parse profile: " << g_config.eqProfilePath << '\n';
            }
        }

        // Pre-allocate streaming input buffers (based on streamValidInputPerBlock_)
        // Use 2x safety margin to handle timing variations
        size_t buffer_capacity = g_upsampler->getStreamValidInputPerBlock() * 2;
        g_stream_input_left.resize(buffer_capacity, 0.0f);
        g_stream_input_right.resize(buffer_capacity, 0.0f);
        std::cout << "Streaming buffer capacity: " << buffer_capacity
                  << " samples (2x streamValidInputPerBlock)" << '\n';
        g_stream_accumulated_left = 0;
        g_stream_accumulated_right = 0;
        size_t upsampler_output_capacity =
            g_upsampler->getStreamValidInputPerBlock() * g_config.upsampleRatio * 2;
        g_upsampler_output_left.reserve(upsampler_output_capacity);
        g_upsampler_output_right.reserve(upsampler_output_capacity);
        initialize_streaming_cache_manager();

        if (!g_config.partitionedConvolution.enabled) {
            // Initialize HRTF processor for crossfeed (optional feature)
            // Crossfeed is disabled by default until enabled via ZeroMQ command
            std::string hrtfDir = "data/crossfeed/hrtf";
            if (std::filesystem::exists(hrtfDir)) {
                std::cout << "Initializing HRTF processor for crossfeed..." << '\n';
                g_hrtf_processor = new CrossfeedEngine::HRTFProcessor();

                // Determine rate family based on input sample rate
                CrossfeedEngine::RateFamily rateFamily =
                    (g_input_sample_rate == 48000) ? CrossfeedEngine::RateFamily::RATE_48K
                                                   : CrossfeedEngine::RateFamily::RATE_44K;

                if (g_hrtf_processor->initialize(hrtfDir, g_config.blockSize,
                                                 CrossfeedEngine::HeadSize::M, rateFamily)) {
                    if (g_hrtf_processor->initializeStreaming()) {
                        std::cout << "  HRTF processor ready (head size: M, rate family: "
                                  << (rateFamily == CrossfeedEngine::RateFamily::RATE_44K ? "44k"
                                                                                          : "48k")
                                  << ")" << '\n';

                        // Pre-allocate crossfeed streaming buffers
                        size_t cf_buffer_capacity =
                            g_hrtf_processor->getStreamValidInputPerBlock() * 2;
                        g_cf_stream_input_left.resize(cf_buffer_capacity, 0.0f);
                        g_cf_stream_input_right.resize(cf_buffer_capacity, 0.0f);
                        g_cf_stream_accumulated_left = 0;
                        g_cf_stream_accumulated_right = 0;
                        g_cf_output_buffer_left.reserve(cf_buffer_capacity);
                        g_cf_output_buffer_right.reserve(cf_buffer_capacity);
                        std::cout << "  Crossfeed buffer capacity: " << cf_buffer_capacity
                                  << " samples" << '\n';

                        // Crossfeed is initialized but disabled by default
                        g_crossfeed_enabled.store(false);
                        g_hrtf_processor->setEnabled(false);
                        std::cout << "  Crossfeed: initialized (disabled by default)" << '\n';
                    } else {
                        std::cerr << "  HRTF: Failed to initialize streaming mode" << '\n';
                        delete g_hrtf_processor;
                        g_hrtf_processor = nullptr;
                    }
                } else {
                    std::cerr << "  HRTF: Failed to initialize processor" << '\n';
                    std::cerr << "  Hint: Run 'uv run python scripts/filters/generate_hrtf.py' to "
                                 "generate HRTF "
                                 "filters"
                              << '\n';
                    delete g_hrtf_processor;
                    g_hrtf_processor = nullptr;
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

        if (!g_audio_pipeline && g_upsampler) {
            audio_pipeline::Dependencies pipelineDeps{};
            pipelineDeps.config = &g_config;
            pipelineDeps.upsampler.available = true;
            pipelineDeps.upsampler.streamLeft = g_upsampler->streamLeft_;
            pipelineDeps.upsampler.streamRight = g_upsampler->streamRight_;
            pipelineDeps.output.outputGain = &g_output_gain;
            pipelineDeps.output.limiterGain = &g_limiter_gain;
            pipelineDeps.output.effectiveGain = &g_effective_gain;
            pipelineDeps.upsampler.process =
                [](const float* data, size_t frames, ConvolutionEngine::StreamFloatVector& output,
                   cudaStream_t stream, ConvolutionEngine::StreamFloatVector& streamInput,
                   size_t& streamAccumulated) {
                    if (!g_upsampler) {
                        return false;
                    }
                    return g_upsampler->processStreamBlock(data, frames, output, stream,
                                                           streamInput, streamAccumulated);
                };
            pipelineDeps.fallbackActive = &g_fallback_active;
            pipelineDeps.outputReady = &g_output_ready;
            pipelineDeps.inputMutex = &g_input_process_mutex;
            pipelineDeps.streamingCacheManager = g_streaming_cache_manager.get();
            pipelineDeps.streamInputLeft = &g_stream_input_left;
            pipelineDeps.streamInputRight = &g_stream_input_right;
            pipelineDeps.streamAccumulatedLeft = &g_stream_accumulated_left;
            pipelineDeps.streamAccumulatedRight = &g_stream_accumulated_right;
            pipelineDeps.upsamplerOutputLeft = &g_upsampler_output_left;
            pipelineDeps.upsamplerOutputRight = &g_upsampler_output_right;
            pipelineDeps.cfStreamInputLeft = &g_cf_stream_input_left;
            pipelineDeps.cfStreamInputRight = &g_cf_stream_input_right;
            pipelineDeps.cfStreamAccumulatedLeft = &g_cf_stream_accumulated_left;
            pipelineDeps.cfStreamAccumulatedRight = &g_cf_stream_accumulated_right;
            pipelineDeps.cfOutputLeft = &g_cf_output_left;
            pipelineDeps.cfOutputRight = &g_cf_output_right;
            pipelineDeps.crossfeedEnabled = &g_crossfeed_enabled;
            pipelineDeps.crossfeedProcessor = g_hrtf_processor;
            pipelineDeps.crossfeedMutex = &g_crossfeed_mutex;
            pipelineDeps.buffer.playbackBuffer = &playback_buffer();
            pipelineDeps.maxOutputBufferFrames = []() { return get_max_output_buffer_frames(); };
            pipelineDeps.currentOutputRate = []() {
                return g_current_output_rate.load(std::memory_order_acquire);
            };
            g_audio_pipeline =
                std::make_unique<audio_pipeline::AudioPipeline>(std::move(pipelineDeps));
            g_audio_pipeline_raw = g_audio_pipeline.get();
        }

        // Check for early abort before starting threads
        if (!g_running) {
            std::cout << "Startup interrupted by signal" << '\n';
            delete g_upsampler;
            g_upsampler = nullptr;
            break;
        }

        // Initialize soft mute controller with output sample rate
        using namespace DaemonConstants;
        int outputSampleRate = g_input_sample_rate * g_config.upsampleRatio;
        g_soft_mute = new SoftMute::Controller(DEFAULT_SOFT_MUTE_FADE_MS, outputSampleRate);
        std::cout << "Soft mute initialized (" << DEFAULT_SOFT_MUTE_FADE_MS << "ms fade at "
                  << outputSampleRate << "Hz)" << '\n';

        // Initialize fallback manager (Issue #139)
        if (g_config.fallback.enabled) {
            g_fallback_manager = new FallbackManager::Manager();
            FallbackManager::FallbackConfig fbConfig;
            fbConfig.gpuThreshold = g_config.fallback.gpuThreshold;
            fbConfig.gpuThresholdCount = g_config.fallback.gpuThresholdCount;
            fbConfig.gpuRecoveryThreshold = g_config.fallback.gpuRecoveryThreshold;
            fbConfig.gpuRecoveryCount = g_config.fallback.gpuRecoveryCount;
            fbConfig.xrunTriggersFallback = g_config.fallback.xrunTriggersFallback;
            fbConfig.monitorIntervalMs = g_config.fallback.monitorIntervalMs;

            // State change callback: update atomic flag and notify via ZeroMQ
            auto stateCallback = [](FallbackManager::FallbackState state) {
                bool isFallback = (state == FallbackManager::FallbackState::Fallback);
                g_fallback_active.store(isFallback, std::memory_order_relaxed);

                // ZeroMQ notification is handled by STATS command response
                LOG_INFO("Fallback state changed: {}", isFallback ? "FALLBACK" : "NORMAL");
            };

            if (g_fallback_manager->initialize(fbConfig, stateCallback)) {
                std::cout << "Fallback manager initialized (GPU threshold: "
                          << fbConfig.gpuThreshold << "%, XRUN fallback: "
                          << (fbConfig.xrunTriggersFallback ? "enabled" : "disabled") << ")"
                          << '\n';
            } else {
                std::cerr << "Warning: Failed to initialize fallback manager" << '\n';
                delete g_fallback_manager;
                g_fallback_manager = nullptr;
                g_fallback_active.store(false, std::memory_order_relaxed);
            }
        } else {
            std::cout << "Fallback manager disabled" << '\n';
            g_fallback_active.store(false, std::memory_order_relaxed);
        }

        std::unique_ptr<daemon_control::ControlPlane> controlPlane;
        // Legacy RTP path was removed: ALSA/TCP-only configuration

        if (!g_dac_manager) {
            g_dac_manager = std::make_unique<dac::DacManager>(make_dac_dependencies());
        }
        if (!g_dac_manager) {
            std::cerr << "Failed to initialize DAC manager" << '\n';
            exitCode = 1;
            break;
        }
        g_dac_manager_raw = g_dac_manager.get();

        daemon_control::ControlPlaneDependencies controlDeps{};
        controlDeps.config = &g_config;
        controlDeps.runningFlag = &g_running;
        controlDeps.reloadRequested = &g_reload_requested;
        controlDeps.zmqBindFailed = &g_zmq_bind_failed;
        controlDeps.currentOutputRate = &g_current_output_rate;
        controlDeps.softMute = &g_soft_mute;
        controlDeps.activePhaseType = &g_active_phase_type;
        controlDeps.inputSampleRate = &g_input_sample_rate;
        controlDeps.defaultAlsaDevice = DEFAULT_ALSA_DEVICE;
        controlDeps.dispatcher = g_event_dispatcher.get();
        controlDeps.quitMainLoop = []() {};
        controlDeps.buildRuntimeStats = []() { return build_runtime_stats_dependencies(); };
        controlDeps.bufferCapacityFrames = []() { return get_max_output_buffer_frames(); };
        controlDeps.applySoftMuteForFilterSwitch = [](std::function<bool()> fn) {
            applySoftMuteForFilterSwitch(std::move(fn));
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
        controlDeps.dacManager = g_dac_manager.get();
        controlDeps.upsampler = &g_upsampler;
        controlDeps.crossfeed.processor = &g_hrtf_processor;
        controlDeps.crossfeed.enabledFlag = &g_crossfeed_enabled;
        controlDeps.crossfeed.mutex = &g_crossfeed_mutex;
        controlDeps.crossfeed.resetStreamingState = []() { reset_crossfeed_stream_state_locked(); };
        controlDeps.statsFilePath = STATS_FILE_PATH;

        controlPlane = std::make_unique<daemon_control::ControlPlane>(std::move(controlDeps));
        if (controlPlane && controlPlane->start() && g_dac_manager) {
            g_dac_manager->setEventPublisher(controlPlane->eventPublisher());
        }

        if (g_dac_manager) {
            g_dac_manager->initialize();
        }

        // Start ALSA output thread
        if (g_dac_manager) {
            g_dac_manager->start();
        }
        std::cout << "Starting ALSA output thread..." << '\n';
        std::thread alsa_thread(alsa_output_thread);

        std::thread loopback_thread;
        if (g_config.loopback.enabled) {
            if (!validate_loopback_config(g_config)) {
                exitCode = 1;
                g_running = false;
                alsa_thread.join();
                break;
            }
            snd_pcm_format_t lb_format = parse_loopback_format(g_config.loopback.format);
            std::cout << "Starting loopback capture thread (" << g_config.loopback.device
                      << ", fmt=" << g_config.loopback.format
                      << ", rate=" << g_config.loopback.sampleRate
                      << ", period=" << g_config.loopback.periodFrames << ")" << '\n';
            loopback_thread =
                std::thread(loopback_capture_thread, g_config.loopback.device, lb_format,
                            g_config.loopback.sampleRate, g_config.loopback.channels,
                            static_cast<snd_pcm_uframes_t>(g_config.loopback.periodFrames));

            // Wait briefly for loopback thread to become ready
            bool loopback_ready = false;
            for (int i = 0; i < 40; ++i) {  // up to ~2s
                if (g_loopback_capture_ready.load(std::memory_order_acquire)) {
                    loopback_ready = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (!loopback_ready) {
                LOG_ERROR("[Loopback] Failed to start capture thread (not ready)");
                g_running = false;
                playback_buffer().cv().notify_all();
                if (loopback_thread.joinable()) {
                    loopback_thread.join();
                }
                alsa_thread.join();
                exitCode = 1;
                break;
            }
        }

        double outputRateKHz = g_input_sample_rate * g_config.upsampleRatio / 1000.0;
        std::cout << '\n';
        if (g_config.loopback.enabled) {
            std::cout << "System ready (Loopback capture mode). Audio routing configured:" << '\n';
            std::cout << "  1. Loopback capture → GPU Upsampler (" << g_config.upsampleRatio
                      << "x upsampling)" << '\n';
            std::cout << "  2. GPU Upsampler → ALSA → SMSL DAC (" << outputRateKHz << "kHz direct)"
                      << '\n';
        } else {
            std::cout << "System ready (TCP/loopback input). Audio routing configured:" << '\n';
            std::cout << "  1. Network or loopback source → GPU Upsampler ("
                      << g_config.upsampleRatio << "x upsampling)" << '\n';
            std::cout << "  2. GPU Upsampler → ALSA → SMSL DAC (" << outputRateKHz << "kHz direct)"
                      << '\n';
        }
        std::cout << "Press Ctrl+C to stop." << '\n';
        std::cout << "========================================" << '\n';

        shutdownManager.notifyReady();

        auto runMainLoop = [&]() {
            while (g_running.load() && !g_reload_requested.load() && !g_zmq_bind_failed.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                shutdownManager.tick();
            }
        };

        if (g_zmq_bind_failed.load()) {
            std::cerr << "Startup aborted due to ZeroMQ bind failure." << '\n';
        } else {
            runMainLoop();
        }

        shutdownManager.runShutdownSequence();

        // Step 5: Signal worker threads to stop and wait for them
        std::cout << "  Step 5: Stopping worker threads..." << '\n';
        g_running = false;
        playback_buffer().cv().notify_all();
        if (controlPlane) {
            controlPlane->stop();
        }
        alsa_thread.join();  // ALSA thread will call snd_pcm_drain() before exit
        if (loopback_thread.joinable()) {
            loopback_thread.join();
        }

        // Step 6: Release audio processing resources
        std::cout << "  Step 6: Releasing resources..." << '\n';
        if (g_fallback_manager) {
            g_fallback_manager->shutdown();
            delete g_fallback_manager;
            g_fallback_manager = nullptr;
            g_fallback_active.store(false, std::memory_order_relaxed);
        }
        delete g_soft_mute;
        g_soft_mute = nullptr;
        delete g_hrtf_processor;
        g_hrtf_processor = nullptr;
        g_crossfeed_enabled.store(false);
        if (g_audio_pipeline) {
            g_audio_pipeline.reset();  // Tear down pipeline before deleting GPU upsampler
            g_audio_pipeline_raw = nullptr;
        }
        delete g_upsampler;
        g_upsampler = nullptr;
        if (g_dac_manager) {
            g_dac_manager->stop();
        }

        // Don't reload if ZMQ bind failed - exit completely
        if (g_zmq_bind_failed.load()) {
            std::cerr << "Exiting due to ZeroMQ initialization failure." << '\n';
            exitCode = 1;
            break;
        }

        if (g_reload_requested) {
            std::cout << "Reload requested. Restarting daemon with updated config..." << '\n';
        }
    } while (g_reload_requested);

    // Release PID lock and remove file on clean exit
    release_pid_lock();
    std::cout << "Goodbye!" << '\n';
    return exitCode;
}
