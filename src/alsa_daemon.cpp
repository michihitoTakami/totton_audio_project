#include "audio_utils.h"
#include "base64.h"
#include "config_loader.h"
#include "convolution_engine.h"
#include "crossfeed_engine.h"
#include "dac_capability.h"
#include "daemon/dac_manager.h"
#include "daemon/rtp_engine_coordinator.h"
#include "daemon/runtime_stats.h"
#include "daemon/shutdown_manager.h"
#include "daemon/streaming_cache_manager.h"
#include "daemon/zmq_server.h"
#include "daemon_constants.h"
#include "eq_parser.h"
#include "eq_to_fir.h"
#include "fallback_manager.h"
#include "filter_headroom.h"
#include "logging/logger.h"
#include "logging/metrics.h"
#include "partition_runtime_utils.h"
#include "playback_buffer.h"
#include "rtp_session_manager.h"
#include "soft_mute.h"

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
#include <pipewire/pipewire.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <spa/param/audio/format-utils.h>
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

static void enforce_phase_partition_constraints(AppConfig& config) {
    if (config.partitionedConvolution.enabled && config.phaseType == PhaseType::Linear) {
        std::cout << "[Partition] Linear phase is incompatible with low-latency mode. "
                  << "Switching to minimum phase." << std::endl;
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
        std::cout << "Config: Unsupported output mode '" << current << "', forcing 'usb'"
                  << std::endl;
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
static std::atomic<bool> g_main_loop_running{false};  // True when pw_main_loop_run() is active
static std::atomic<bool> g_zmq_bind_failed{false};    // True if ZeroMQ bind failed
static std::atomic<bool> g_output_ready{false};  // Indicates ALSA DAC is ready for input processing
static ConvolutionEngine::GPUUpsampler* g_upsampler = nullptr;
static struct pw_main_loop* g_pw_loop = nullptr;  // For signal check timer
static std::unique_ptr<rtp_engine::RtpEngineCoordinator> g_rtp_coordinator;
static std::mutex g_input_process_mutex;

static void process_interleaved_block(const float* input_samples, uint32_t n_frames);

static std::unique_ptr<daemon_ipc::ZmqCommandServer> g_zmq_server;

static inline float compute_stereo_peak(const float* left, const float* right, size_t frames) {
    if (!left || !right || frames == 0) {
        return 0.0f;
    }
    float peak = 0.0f;
    for (size_t i = 0; i < frames; ++i) {
        float l = std::fabs(left[i]);
        float r = std::fabs(right[i]);
        if (l > peak)
            peak = l;
        if (r > peak)
            peak = r;
    }
    return peak;
}

static float apply_output_limiter(float* interleaved, size_t frames) {
    constexpr float kEpsilon = 1e-6f;
    if (!interleaved || frames == 0) {
        g_limiter_gain.store(1.0f, std::memory_order_relaxed);
        float baseGain = g_output_gain.load(std::memory_order_relaxed);
        g_effective_gain.store(baseGain, std::memory_order_relaxed);
        return 0.0f;
    }
    float peak = AudioUtils::computeInterleavedPeak(interleaved, frames);
    float limiterGain = 1.0f;
    const float target = g_config.headroomTarget;
    if (target > 0.0f && peak > target) {
        limiterGain = target / (peak + kEpsilon);
        AudioUtils::applyInterleavedGain(interleaved, frames, limiterGain);
        peak = target;
    }
    g_limiter_gain.store(limiterGain, std::memory_order_relaxed);
    float effective = g_output_gain.load(std::memory_order_relaxed) * limiterGain;
    g_effective_gain.store(effective, std::memory_order_relaxed);
    return peak;
}

// Runtime state: Input sample rate (auto-negotiated, not from config)
// This value is detected from PipeWire stream or set to default (44100 Hz)
// Issue #219: Changed to atomic for thread-safe multi-rate switching
static std::atomic<int> g_current_input_rate{DEFAULT_INPUT_SAMPLE_RATE};
static std::atomic<int> g_current_output_rate{DEFAULT_OUTPUT_SAMPLE_RATE};
static std::atomic<int> g_current_rate_family_int{
    static_cast<int>(ConvolutionEngine::RateFamily::RATE_44K)};

static size_t get_max_output_buffer_frames() {
    using namespace DaemonConstants;
    double seconds = static_cast<double>(MAX_OUTPUT_BUFFER_SECONDS);
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

// Pending rate change (set by PipeWire callback, processed in main loop)
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

// Helper function for soft mute during filter switching (Issue #266)
// Fade-out: 1.5 seconds, perform filter switch, fade-in: 1.5 seconds
// Forward declaration for handle_rate_change (Issue #219)
// Defined later in file, but called from ZeroMQ handler
static bool handle_rate_change(int detected_sample_rate);

// Thread safety: This function is called from ZeroMQ command thread.
// Fade parameter changes leverage the atomic configuration inside SoftMute::Controller, and
// we still wait for fade-out completion before manipulating filters to avoid artifacts.
static void applySoftMuteForFilterSwitch(std::function<bool()> filterSwitchFunc) {
    using namespace DaemonConstants;

    if (!g_soft_mute) {
        // If soft mute not initialized, perform switch without mute
        filterSwitchFunc();
        return;
    }

    // Save current fade duration for restoration
    int originalFadeDuration = g_soft_mute->getFadeDuration();
    int outputSampleRate = g_soft_mute->getSampleRate();

    // Update fade duration for filter switching
    // Note: This is called from command thread, but audio thread may be processing.
    // The fade calculation will use the new duration from the next audio frame.
    g_soft_mute->setFadeDuration(FILTER_SWITCH_FADE_MS);
    g_soft_mute->setSampleRate(outputSampleRate);

    std::cout << "[Filter Switch] Starting fade-out (" << (FILTER_SWITCH_FADE_MS / 1000.0)
              << "s)..." << std::endl;
    g_soft_mute->startFadeOut();

    // Wait for fade-out to complete (approximately 1.5 seconds)
    // Polling is necessary because fade is processed in audio thread
    auto fade_start = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::milliseconds(FILTER_SWITCH_FADE_TIMEOUT_MS);
    while (g_soft_mute->isTransitioning() &&
           g_soft_mute->getState() == SoftMute::MuteState::FADING_OUT) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto elapsed = std::chrono::steady_clock::now() - fade_start;
        if (elapsed > timeout) {
            std::cerr << "[Filter Switch] Warning: Fade-out timeout ("
                      << FILTER_SWITCH_FADE_TIMEOUT_MS << "ms), proceeding with switch"
                      << std::endl;
            break;
        }
    }

    // Ensure we're fully muted before switching
    if (g_soft_mute->getState() != SoftMute::MuteState::MUTED) {
        g_soft_mute->setMuted();
    }

    // Perform filter switch
    bool switch_success = filterSwitchFunc();

    if (switch_success) {
        // Start fade-in after filter switch
        std::cout << "[Filter Switch] Starting fade-in (" << (FILTER_SWITCH_FADE_MS / 1000.0)
                  << "s)..." << std::endl;
        g_soft_mute->startFadeIn();

        // Store original fade duration for reset in audio thread
        // The audio thread will reset fade duration when fade-in completes
        // (see alsa_output_thread() around line 1241)
    } else {
        // If switch failed, restore original state immediately
        std::cerr << "[Filter Switch] Switch failed, restoring audio state" << std::endl;
        g_soft_mute->setPlaying();
        g_soft_mute->setFadeDuration(originalFadeDuration);
        g_soft_mute->setSampleRate(outputSampleRate);
    }
}

// Audio buffer for thread communication
static std::mutex g_buffer_mutex;
static std::condition_variable g_buffer_cv;
static std::vector<float> g_output_buffer_left;
static std::vector<float> g_output_buffer_right;
static size_t g_output_read_pos = 0;

static void trim_output_buffer_locked(size_t minFramesToRemove) {
    if (g_output_read_pos == 0) {
        return;
    }

    size_t readable = g_output_buffer_left.size();
    if (readable == 0) {
        g_output_read_pos = 0;
        return;
    }

    if (g_output_read_pos > readable) {
        g_output_read_pos = readable;
    }

    if ((minFramesToRemove > 0 && g_output_read_pos >= minFramesToRemove) ||
        g_output_read_pos == readable) {
        auto eraseCount = static_cast<std::ptrdiff_t>(g_output_read_pos);
        g_output_buffer_left.erase(g_output_buffer_left.begin(),
                                   g_output_buffer_left.begin() + eraseCount);
        g_output_buffer_right.erase(g_output_buffer_right.begin(),
                                    g_output_buffer_right.begin() + eraseCount);
        g_output_read_pos = 0;
    }
}

template <typename Container>
static size_t enqueue_output_frames_locked(const Container& left, const Container& right) {
    size_t framesAvailable = std::min(left.size(), right.size());
    if (framesAvailable == 0) {
        return 0;
    }

    size_t capacityFrames = std::max<size_t>(1, get_max_output_buffer_frames());
    size_t bufferSize = g_output_buffer_left.size();
    size_t currentFrames = (bufferSize >= g_output_read_pos) ? (bufferSize - g_output_read_pos) : 0;
    auto decision =
        PlaybackBuffer::planCapacityEnforcement(currentFrames, framesAvailable, capacityFrames);

    size_t totalDropped = decision.dropFromExisting + decision.newDataOffset;
    if (totalDropped > 0) {
        int outputRate = g_current_output_rate.load(std::memory_order_acquire);
        if (outputRate <= 0) {
            outputRate = DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE;
        }
        float seconds = static_cast<float>(totalDropped) / static_cast<float>(outputRate);
        LOG_WARN(
            "Output buffer overflow: dropping {} frames ({:.3f}s) [queued={}, incoming={}, "
            "max={}]",
            totalDropped, seconds, currentFrames, framesAvailable, capacityFrames);
        runtime_stats::addDroppedFrames(totalDropped);
    }

    if (decision.dropFromExisting > 0) {
        g_output_read_pos += decision.dropFromExisting;
    }

    trim_output_buffer_locked(capacityFrames);

    if (decision.framesToStore == 0) {
        return 0;
    }

    size_t startIndex = framesAvailable - decision.framesToStore;
    auto startOffset = static_cast<std::ptrdiff_t>(startIndex);
    auto endOffset = static_cast<std::ptrdiff_t>(framesAvailable);
    g_output_buffer_left.insert(g_output_buffer_left.end(), left.begin() + startOffset,
                                left.begin() + endOffset);
    g_output_buffer_right.insert(g_output_buffer_right.end(), right.begin() + startOffset,
                                 right.begin() + endOffset);
    return decision.framesToStore;
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
    deps.outputBufferLeft = &g_output_buffer_left;
    deps.outputBufferRight = &g_output_buffer_right;
    deps.bufferMutex = &g_buffer_mutex;
    deps.outputReadPos = &g_output_read_pos;

    deps.streamInputLeft = &g_stream_input_left;
    deps.streamInputRight = &g_stream_input_right;
    deps.streamAccumulatedLeft = &g_stream_accumulated_left;
    deps.streamAccumulatedRight = &g_stream_accumulated_right;
    deps.streamingMutex = &g_streaming_mutex;

    deps.upsamplerPtr = &g_upsampler;
    deps.onCrossfeedReset = []() {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        reset_crossfeed_stream_state_locked();
    };
    g_streaming_cache_manager = std::make_unique<streaming_cache::StreamingCacheManager>(deps);
}

// ========== DAC Device Monitoring & ZeroMQ PUB ==========

static inline int64_t get_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static void send_zmq_payload(const nlohmann::json& payload) {
    if (!g_zmq_server) {
        return;
    }
    g_zmq_server->publish(payload.dump());
}

static dac::DacManager::Dependencies make_dac_dependencies() {
    dac::DacManager::Dependencies deps;
    deps.config = &g_config;
    deps.runningFlag = &g_running;
    deps.timestampProvider = get_timestamp_ms;
    deps.eventPublisher = send_zmq_payload;
    deps.defaultDevice = DEFAULT_ALSA_DEVICE;
    return deps;
}

static dac::DacManager g_dac_manager(make_dac_dependencies());

static runtime_stats::Dependencies build_runtime_stats_dependencies() {
    runtime_stats::Dependencies deps;
    deps.config = &g_config;
    deps.upsampler = g_upsampler;
    deps.headroomCache = &g_headroom_cache;
    deps.dacManager = &g_dac_manager;
    deps.fallbackManager = g_fallback_manager;
    deps.fallbackActive = &g_fallback_active;
    deps.inputSampleRate = &g_input_sample_rate;
    deps.headroomGain = &g_headroom_gain;
    deps.outputGain = &g_output_gain;
    deps.limiterGain = &g_limiter_gain;
    deps.effectiveGain = &g_effective_gain;
    return deps;
}

static void mark_dac_connected(
    const std::string& device,
    const char* log_message = "DAC connected - input processing enabled") {
    g_output_ready.store(true, std::memory_order_release);
    LOG_INFO(log_message);
    g_dac_manager.markActiveDevice(device, true);
}

static void mark_dac_disconnected(
    const std::string& device,
    const char* log_message = "DAC disconnected - stopping input processing") {
    g_output_ready.store(false, std::memory_order_release);
    LOG_INFO(log_message);
    g_dac_manager.markActiveDevice(device, false);
}

// ========== PID File Lock (flock-based) ==========

// File descriptor for the PID lock file (kept open while running)
static int g_pid_lock_fd = -1;

// Read PID from lock file (for display purposes)
static pid_t read_pid_from_lockfile() {
    std::ifstream pidfile(PID_FILE_PATH);
    if (!pidfile.is_open())
        return 0;
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

    g_output_buffer_left.clear();
    g_output_buffer_right.clear();
    g_output_read_pos = 0;
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

static bool handle_rate_switch(int newInputRate) {
    if (!g_upsampler || !g_upsampler->isMultiRateEnabled()) {
        std::cerr << "[Rate] Multi-rate mode not enabled" << std::endl;
        return false;
    }

    int currentRate = g_upsampler->getCurrentInputRate();
    if (currentRate == newInputRate) {
        std::cout << "[Rate] Already at target rate: " << newInputRate << " Hz" << std::endl;
        return true;
    }

    std::cout << "[Rate] Switching: " << currentRate << " Hz -> " << newInputRate << " Hz"
              << std::endl;

    int savedRate = currentRate;

    if (g_soft_mute) {
        g_soft_mute->startFadeOut();
        auto startTime = std::chrono::steady_clock::now();
        while (g_soft_mute->isTransitioning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto elapsed = std::chrono::steady_clock::now() - startTime;
            if (elapsed > std::chrono::milliseconds(200)) {
                std::cerr << "[Rate] Warning: Fade-out timeout" << std::endl;
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

        {
            std::lock_guard<std::mutex> bufferLock(g_buffer_mutex);
            g_output_buffer_left.clear();
            g_output_buffer_right.clear();
            g_output_read_pos = 0;
        }
        g_stream_input_left.clear();
        g_stream_input_right.clear();
        g_stream_accumulated_left = 0;
        g_stream_accumulated_right = 0;

        if (!g_upsampler->switchToInputRate(newInputRate)) {
            std::cerr << "[Rate] Failed to switch rate, rolling back" << std::endl;
            if (g_upsampler->switchToInputRate(savedRate)) {
                std::cout << "[Rate] Rollback successful: restored to " << savedRate << " Hz"
                          << std::endl;
            } else {
                std::cerr << "[Rate] ERROR: Rollback failed!" << std::endl;
            }
            if (g_soft_mute) {
                g_soft_mute->startFadeIn();
            }
            return false;
        }

        if (!g_upsampler->initializeStreaming()) {
            std::cerr << "[Rate] Failed to re-initialize streaming mode, rolling back" << std::endl;
            if (g_upsampler->switchToInputRate(savedRate)) {
                if (g_upsampler->initializeStreaming()) {
                    std::cout << "[Rate] Rollback successful: restored to " << savedRate << " Hz"
                              << std::endl;
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
              << "x -> " << newOutputRate << " Hz)" << std::endl;
    std::cout << "[Rate] Streaming buffers re-initialized: " << buffer_capacity
              << " samples capacity" << std::endl;

    return true;
}

static void load_runtime_config() {
    AppConfig loaded;
    bool found = loadAppConfig(CONFIG_FILE_PATH, loaded);
    g_config = loaded;
    ensure_output_config(g_config);

    if (g_config.alsaDevice.empty())
        set_preferred_output_device(g_config, DEFAULT_ALSA_DEVICE);
    if (g_config.filterPath.empty())
        g_config.filterPath = DEFAULT_FILTER_PATH;
    if (g_config.upsampleRatio <= 0)
        g_config.upsampleRatio = DEFAULT_UPSAMPLE_RATIO;
    if (g_config.blockSize <= 0)
        g_config.blockSize = DEFAULT_BLOCK_SIZE;
    if (g_config.bufferSize <= 0)
        g_config.bufferSize = 262144;
    if (g_config.periodSize <= 0)
        g_config.periodSize = 32768;
    if (g_input_sample_rate != 44100 && g_input_sample_rate != 48000) {
        g_input_sample_rate = DEFAULT_INPUT_SAMPLE_RATE;
    }

    if (!found) {
        std::cout << "Config: Using defaults (no config.json found)" << std::endl;
    }

    enforce_phase_partition_constraints(g_config);

    print_config_summary(g_config);
    g_headroom_cache.setTargetPeak(g_config.headroomTarget);
    update_effective_gain(1.0f, "config load (pending filter headroom)");
    float initialOutput = g_output_gain.load(std::memory_order_relaxed);
    g_limiter_gain.store(1.0f, std::memory_order_relaxed);
    g_effective_gain.store(initialOutput, std::memory_order_relaxed);
}

// ========== ZeroMQ Command Listener ==========

static std::string build_ok_response(const daemon_ipc::ZmqRequest& request,
                                     const std::string& message = "",
                                     const nlohmann::json& data = {}) {
    if (request.isJson) {
        nlohmann::json resp;
        resp["status"] = "ok";
        if (!message.empty()) {
            resp["message"] = message;
        }
        if (!data.is_null() && !data.empty()) {
            resp["data"] = data;
        }
        return resp.dump();
    }

    if (!data.is_null() && !data.empty()) {
        return "OK:" + data.dump();
    }
    if (!message.empty()) {
        return "OK:" + message;
    }
    return "OK";
}

static std::string build_error_response(const daemon_ipc::ZmqRequest& request,
                                        const std::string& code, const std::string& message) {
    if (request.isJson) {
        nlohmann::json resp;
        resp["status"] = "error";
        resp["error_code"] = code;
        resp["message"] = message;
        return resp.dump();
    }
    return "ERR:" + message;
}

static std::string handle_ping(const daemon_ipc::ZmqRequest& request) {
    return build_ok_response(request);
}

static std::string handle_reload(const daemon_ipc::ZmqRequest& request) {
    g_reload_requested = true;
    if (g_soft_mute) {
        g_soft_mute->startFadeOut();
    }
    if (g_main_loop_running.load() && g_pw_loop) {
        pw_main_loop_quit(g_pw_loop);
    }
    return build_ok_response(request);
}

static std::string handle_stats_command(const daemon_ipc::ZmqRequest& request) {
    auto stats =
        runtime_stats::collect(build_runtime_stats_dependencies(), get_max_output_buffer_frames());
    return build_ok_response(request, "", stats);
}

static std::string handle_crossfeed_enable(const daemon_ipc::ZmqRequest& request) {
    if (g_config.partitionedConvolution.enabled) {
        return build_error_response(request, "CROSSFEED_DISABLED",
                                    "Crossfeed not available in low-latency mode");
    }

    std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
    if (!g_hrtf_processor) {
        return build_error_response(request, "CROSSFEED_NOT_INITIALIZED",
                                    "HRTF processor not initialized");
    }

    reset_crossfeed_stream_state_locked();
    g_hrtf_processor->setEnabled(true);
    g_crossfeed_enabled.store(true);
    return build_ok_response(request, "Crossfeed enabled");
}

static std::string handle_crossfeed_disable(const daemon_ipc::ZmqRequest& request) {
    std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
    g_crossfeed_enabled.store(false);
    if (g_hrtf_processor) {
        g_hrtf_processor->setEnabled(false);
    }
    reset_crossfeed_stream_state_locked();
    return build_ok_response(request, "Crossfeed disabled");
}

static std::string build_crossfeed_status_response(const daemon_ipc::ZmqRequest& request,
                                                   bool includeHeadSize) {
    std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
    bool enabled = g_crossfeed_enabled.load();
    bool initialized = (g_hrtf_processor != nullptr);

    nlohmann::json data;
    data["enabled"] = enabled;
    data["initialized"] = initialized;

    if (includeHeadSize) {
        if (g_hrtf_processor != nullptr) {
            CrossfeedEngine::HeadSize currentSize = g_hrtf_processor->getCurrentHeadSize();
            data["head_size"] = CrossfeedEngine::headSizeToString(currentSize);
        } else {
            data["head_size"] = nullptr;
        }
    }

    return build_ok_response(request, "", data);
}

static std::string handle_crossfeed_status(const daemon_ipc::ZmqRequest& request) {
    return build_crossfeed_status_response(request, false);
}

static std::string handle_crossfeed_get_status(const daemon_ipc::ZmqRequest& request) {
    return build_crossfeed_status_response(request, true);
}

static bool validate_crossfeed_params(const nlohmann::json& params, std::string& rateFamily,
                                      std::string& combinedLL, std::string& combinedLR,
                                      std::string& combinedRL, std::string& combinedRR,
                                      std::string& errorMessage, std::string& errorCode) {
    rateFamily = params.value("rate_family", "");
    combinedLL = params.value("combined_ll", "");
    combinedLR = params.value("combined_lr", "");
    combinedRL = params.value("combined_rl", "");
    combinedRR = params.value("combined_rr", "");

    if (rateFamily.empty() || combinedLL.empty() || combinedLR.empty() || combinedRL.empty() ||
        combinedRR.empty()) {
        errorCode = "IPC_INVALID_PARAMS";
        errorMessage = "Missing required filter data";
        return false;
    }
    if (rateFamily != "44k" && rateFamily != "48k") {
        errorCode = "CROSSFEED_INVALID_RATE_FAMILY";
        errorMessage = "Invalid rate family: " + rateFamily + " (expected 44k or 48k)";
        return false;
    }
    return true;
}

static std::string handle_crossfeed_set_combined(const daemon_ipc::ZmqRequest& request) {
    if (!request.json || !request.json->contains("params")) {
        return build_error_response(request, "IPC_INVALID_PARAMS", "Missing params field");
    }

    bool processorReady = false;
    {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        processorReady = (g_hrtf_processor != nullptr);
    }
    if (!processorReady) {
        return build_error_response(request, "CROSSFEED_NOT_INITIALIZED",
                                    "HRTF processor not initialized");
    }

    auto params = (*request.json)["params"];
    std::string rateFamily;
    std::string combinedLL;
    std::string combinedLR;
    std::string combinedRL;
    std::string combinedRR;
    std::string errorMessage;
    std::string errorCode;

    if (!validate_crossfeed_params(params, rateFamily, combinedLL, combinedLR, combinedRL,
                                   combinedRR, errorMessage, errorCode)) {
        return build_error_response(request, errorCode, errorMessage);
    }

    auto decodedLL = Base64::decode(combinedLL);
    auto decodedLR = Base64::decode(combinedLR);
    auto decodedRL = Base64::decode(combinedRL);
    auto decodedRR = Base64::decode(combinedRR);

    constexpr size_t CUFFT_COMPLEX_SIZE = 8;
    constexpr size_t MAX_FILTER_BYTES = 256 * 1024;

    bool sizeValid = (decodedLL.size() % CUFFT_COMPLEX_SIZE == 0) &&
                     (decodedLR.size() % CUFFT_COMPLEX_SIZE == 0) &&
                     (decodedRL.size() % CUFFT_COMPLEX_SIZE == 0) &&
                     (decodedRR.size() % CUFFT_COMPLEX_SIZE == 0);

    bool sizesMatch = (decodedLL.size() == decodedLR.size()) &&
                      (decodedLL.size() == decodedRL.size()) &&
                      (decodedLL.size() == decodedRR.size());

    bool withinLimit = (decodedLL.size() <= MAX_FILTER_BYTES);

    if (!sizeValid) {
        return build_error_response(request, "CROSSFEED_INVALID_FILTER_SIZE",
                                    "Filter size must be multiple of 8 (cufftComplex)");
    }
    if (!sizesMatch) {
        return build_error_response(request, "CROSSFEED_INVALID_FILTER_SIZE",
                                    "All 4 channel filters must have same size");
    }
    if (!withinLimit) {
        return build_error_response(request, "CROSSFEED_INVALID_FILTER_SIZE",
                                    "Filter size exceeds maximum (256KB per channel)");
    }

    CrossfeedEngine::RateFamily family = (rateFamily == "44k")
                                             ? CrossfeedEngine::RateFamily::RATE_44K
                                             : CrossfeedEngine::RateFamily::RATE_48K;
    size_t complexCount = decodedLL.size() / CUFFT_COMPLEX_SIZE;
    const cufftComplex* filterLL = reinterpret_cast<const cufftComplex*>(decodedLL.data());
    const cufftComplex* filterLR = reinterpret_cast<const cufftComplex*>(decodedLR.data());
    const cufftComplex* filterRL = reinterpret_cast<const cufftComplex*>(decodedRL.data());
    const cufftComplex* filterRR = reinterpret_cast<const cufftComplex*>(decodedRR.data());

    bool applySuccess = false;
    applySoftMuteForFilterSwitch([&]() {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        if (!g_hrtf_processor) {
            return false;
        }
        applySuccess = g_hrtf_processor->setCombinedFilter(family, filterLL, filterLR, filterRL,
                                                           filterRR, complexCount);
        return applySuccess;
    });

    if (!applySuccess) {
        size_t expectedSize = 0;
        {
            std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
            if (g_hrtf_processor) {
                expectedSize = g_hrtf_processor->getFilterFftSize();
            }
        }
        nlohmann::json errorData;
        errorData["rate_family"] = rateFamily;
        errorData["complex_count"] = complexCount;
        errorData["expected_size"] = expectedSize;
        nlohmann::json resp;
        resp["status"] = "error";
        resp["error_code"] = "CROSSFEED_INVALID_FILTER_SIZE";
        resp["message"] = "Filter size mismatch or application failed";
        resp["data"] = errorData;
        return resp.dump();
    }

    {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        reset_crossfeed_stream_state_locked();
    }

    nlohmann::json data;
    data["rate_family"] = rateFamily;
    data["complex_count"] = complexCount;
    std::cout << "ZeroMQ: CROSSFEED_SET_COMBINED applied for " << rateFamily << " (" << complexCount
              << " complex values)" << std::endl;
    return build_ok_response(request, "Combined filter applied", data);
}

static std::string handle_crossfeed_generate(const daemon_ipc::ZmqRequest& request) {
    if (!request.json || !request.json->contains("params") ||
        !(*request.json)["params"].is_object()) {
        return build_error_response(request, "IPC_INVALID_PARAMS", "Missing params field");
    }

    bool processorReady = false;
    {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        processorReady = (g_hrtf_processor != nullptr);
    }
    if (!processorReady) {
        return build_error_response(request, "CROSSFEED_NOT_INITIALIZED",
                                    "HRTF processor not initialized");
    }

    auto params = (*request.json)["params"];
    std::string rateFamily = params.value("rate_family", "");
    double azimuth = params.value("azimuth_deg", 30.0);

    if (rateFamily != "44k" && rateFamily != "48k") {
        return build_error_response(request, "CROSSFEED_INVALID_RATE_FAMILY",
                                    "Invalid rate family: " + rateFamily);
    }

    HRTF::WoodworthParams modelParams;
    if (params.contains("model") && params["model"].is_object()) {
        auto model = params["model"];
        modelParams.headRadiusMeters = model.value("head_radius_m", modelParams.headRadiusMeters);
        modelParams.earSpacingMeters = model.value("ear_spacing_m", modelParams.earSpacingMeters);
        modelParams.farEarShadowDb = model.value("far_shadow_db", modelParams.farEarShadowDb);
        modelParams.diffuseFieldTiltDb =
            model.value("diffuse_tilt_db", modelParams.diffuseFieldTiltDb);
    }

    CrossfeedEngine::RateFamily family = (rateFamily == "44k")
                                             ? CrossfeedEngine::RateFamily::RATE_44K
                                             : CrossfeedEngine::RateFamily::RATE_48K;
    bool success = false;
    applySoftMuteForFilterSwitch([&]() {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        if (!g_hrtf_processor) {
            return false;
        }
        success = g_hrtf_processor->generateWoodworthProfile(family, static_cast<float>(azimuth),
                                                             modelParams);
        return success;
    });

    if (!success) {
        return build_error_response(request, "CROSSFEED_WOODWORTH_FAILED",
                                    "Failed to generate Woodworth profile");
    }

    {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        reset_crossfeed_stream_state_locked();
    }

    nlohmann::json data;
    data["rate_family"] = rateFamily;
    data["azimuth_deg"] = azimuth;
    data["head_radius_m"] = modelParams.headRadiusMeters;
    data["ear_spacing_m"] = modelParams.earSpacingMeters;
    data["far_shadow_db"] = modelParams.farEarShadowDb;
    data["diffuse_tilt_db"] = modelParams.diffuseFieldTiltDb;
    std::cout << "ZeroMQ: Generated Woodworth HRTF (" << rateFamily << ", az=" << azimuth << " deg)"
              << std::endl;
    return build_ok_response(request, "Woodworth profile generated", data);
}

static std::string handle_crossfeed_set_size(const daemon_ipc::ZmqRequest& request) {
    if (!request.json || !request.json->contains("params")) {
        return build_error_response(request, "IPC_INVALID_PARAMS", "Missing params field");
    }

    {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        if (!g_hrtf_processor) {
            return build_error_response(request, "CROSSFEED_NOT_INITIALIZED",
                                        "HRTF processor not initialized");
        }
    }

    auto params = (*request.json)["params"];
    std::string sizeStr = params.value("head_size", "");
    if (sizeStr.empty()) {
        return build_error_response(request, "IPC_INVALID_PARAMS", "Missing head_size parameter");
    }

    CrossfeedEngine::HeadSize targetSize = CrossfeedEngine::stringToHeadSize(sizeStr);
    bool switchSuccess = false;
    applySoftMuteForFilterSwitch([&]() {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        if (!g_hrtf_processor) {
            return false;
        }
        switchSuccess = g_hrtf_processor->switchHeadSize(targetSize);
        return switchSuccess;
    });

    if (!switchSuccess) {
        return build_error_response(request, "CROSSFEED_SIZE_SWITCH_FAILED",
                                    "Failed to switch head size");
    }

    {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        reset_crossfeed_stream_state_locked();
    }
    nlohmann::json data;
    data["head_size"] = CrossfeedEngine::headSizeToString(targetSize);
    return build_ok_response(request, "", data);
}

static std::string handle_rtp_command(const daemon_ipc::ZmqRequest& request) {
    if (!request.json) {
        return build_error_response(request, "IPC_INVALID_COMMAND",
                                    "RTP command requires JSON payload");
    }
    if (!g_rtp_coordinator) {
        return build_error_response(request, "IPC_INVALID_COMMAND",
                                    "RTP coordinator not initialized");
    }

    std::string response;
    if (g_rtp_coordinator->handleZeroMqCommand(request.command, *request.json, response)) {
        return response;
    }
    return build_error_response(request, "IPC_INVALID_COMMAND",
                                "Unknown JSON command: " + request.command);
}

static std::string handle_dac_list(const daemon_ipc::ZmqRequest& request) {
    return build_ok_response(request, "", g_dac_manager.buildDevicesJson());
}

static std::string handle_dac_status(const daemon_ipc::ZmqRequest& request) {
    nlohmann::json data = g_dac_manager.buildStatusJson();
    data["output_rate"] = g_current_output_rate.load(std::memory_order_acquire);
    return build_ok_response(request, "", data);
}

static std::string handle_dac_select(const daemon_ipc::ZmqRequest& request) {
    if (!request.json || !request.json->contains("params") ||
        !(*request.json)["params"].contains("device")) {
        return build_error_response(request, "IPC_INVALID_PARAMS", "Missing params.device field");
    }

    std::string targetDevice = (*request.json)["params"]["device"].get<std::string>();
    if (!g_dac_manager.isValidDeviceName(targetDevice)) {
        return build_error_response(request, "IPC_INVALID_PARAMS", "Invalid ALSA device name");
    }

    g_dac_manager.requestDevice(targetDevice);

    return build_ok_response(request, "Preferred ALSA device updated",
                             g_dac_manager.buildDevicesJson());
}

static std::string handle_dac_rescan(const daemon_ipc::ZmqRequest& request) {
    g_dac_manager.requestRescan();
    return build_ok_response(request, "DAC rescan scheduled", g_dac_manager.buildDevicesJson());
}

static nlohmann::json build_output_mode_json() {
    nlohmann::json data;
    data["mode"] = outputModeToString(g_config.output.mode);

    nlohmann::json modes = nlohmann::json::array();
    for (const auto* mode : kSupportedOutputModes) {
        modes.push_back(mode);
    }
    data["available_modes"] = modes;

    nlohmann::json options;
    options["usb"]["preferred_device"] = g_config.output.usb.preferredDevice;
    data["options"] = options;
    return data;
}

static std::string handle_output_mode_get(const daemon_ipc::ZmqRequest& request) {
    return build_ok_response(request, "", build_output_mode_json());
}

static std::string handle_output_mode_set(const daemon_ipc::ZmqRequest& request) {
    if (!request.json || !request.json->contains("params") ||
        !(*request.json)["params"].is_object()) {
        return build_error_response(request, "IPC_INVALID_PARAMS", "Missing params object");
    }

    const auto& params = (*request.json)["params"];
    std::string requestedMode = outputModeToString(g_config.output.mode);
    if (params.contains("mode") && params["mode"].is_string()) {
        requestedMode = params["mode"].get<std::string>();
    }
    std::string normalizedMode = normalize_output_mode(requestedMode);
    if (!is_supported_output_mode(normalizedMode)) {
        return build_error_response(request, "ERR_UNSUPPORTED_MODE",
                                    "Output mode '" + requestedMode + "' is not supported");
    }

    std::string preferredDevice = g_config.output.usb.preferredDevice;
    if (params.contains("options") && params["options"].is_object()) {
        const auto& options = params["options"];
        if (options.contains("usb") && options["usb"].is_object()) {
            const auto& usb = options["usb"];
            if (usb.contains("preferred_device") && usb["preferred_device"].is_string()) {
                preferredDevice = usb["preferred_device"].get<std::string>();
            } else if (usb.contains("preferredDevice") && usb["preferredDevice"].is_string()) {
                preferredDevice = usb["preferredDevice"].get<std::string>();
            }
        }
    }

    if (preferredDevice.empty()) {
        preferredDevice = DEFAULT_ALSA_DEVICE;
    }

    if (!g_dac_manager.isValidDeviceName(preferredDevice)) {
        return build_error_response(request, "IPC_INVALID_PARAMS", "Invalid ALSA device name");
    }

    set_preferred_output_device(g_config, preferredDevice);
    g_dac_manager.requestDevice(preferredDevice);

    return build_ok_response(request, "Output mode updated", build_output_mode_json());
}

static std::string handle_phase_type_get(const daemon_ipc::ZmqRequest& request) {
    if (!g_upsampler) {
        return build_error_response(request, "IPC_INVALID_COMMAND", "Upsampler not initialized");
    }
    PhaseType pt = g_upsampler->getPhaseType();
    std::string ptStr = (pt == PhaseType::Minimum) ? "minimum" : "linear";
    nlohmann::json data;
    data["phase_type"] = ptStr;
    return build_ok_response(request, "", data);
}

static std::string handle_phase_type_set(const daemon_ipc::ZmqRequest& request) {
    std::string phaseStr = request.payload;
    if (phaseStr.empty()) {
        return build_error_response(request, "IPC_INVALID_PARAMS",
                                    "Invalid phase type (use 'minimum' or 'linear')");
    }

    if (!g_upsampler) {
        return build_error_response(request, "IPC_INVALID_COMMAND", "Upsampler not initialized");
    }
    if (phaseStr != "minimum" && phaseStr != "linear") {
        return build_error_response(request, "IPC_INVALID_PARAMS",
                                    "Invalid phase type (use 'minimum' or 'linear')");
    }

    PhaseType newPhase = (phaseStr == "minimum") ? PhaseType::Minimum : PhaseType::Linear;
    PhaseType oldPhase = g_upsampler->getPhaseType();

    if (oldPhase == newPhase) {
        return build_ok_response(request, "Phase type already " + phaseStr);
    }

    bool switch_success = false;
    applySoftMuteForFilterSwitch([&]() {
        switch_success = g_upsampler->switchPhaseType(newPhase);
        if (switch_success) {
            g_active_phase_type = newPhase;
            refresh_current_headroom("phase switch");
            if (g_config.eqEnabled && !g_config.eqProfilePath.empty()) {
                EQ::EqProfile eqProfile;
                if (EQ::parseEqFile(g_config.eqProfilePath, eqProfile)) {
                    size_t filterFftSize = g_upsampler->getFilterFftSize();
                    size_t fullFftSize = g_upsampler->getFullFftSize();
                    double outputSampleRate =
                        static_cast<double>(g_input_sample_rate) * g_config.upsampleRatio;
                    auto eqMagnitude = EQ::computeEqMagnitudeForFft(filterFftSize, fullFftSize,
                                                                    outputSampleRate, eqProfile);
                    if (g_upsampler->applyEqMagnitude(eqMagnitude)) {
                        std::cout << "ZeroMQ: EQ re-applied with " << phaseStr << " phase"
                                  << std::endl;
                    } else {
                        std::cerr << "ZeroMQ: Warning - EQ re-apply failed" << std::endl;
                    }
                } else {
                    std::cerr << "ZeroMQ: Warning - Failed to parse EQ profile: "
                              << g_config.eqProfilePath << std::endl;
                }
            }
        }
        return switch_success;
    });

    if (!switch_success) {
        return build_error_response(request, "IPC_PROTOCOL_ERROR", "Failed to switch phase type");
    }

    if (newPhase == PhaseType::Linear && g_config.partitionedConvolution.enabled) {
        std::cout << "[Partition] Linear phase selected, disabling low-latency partitioned "
                     "convolution."
                  << std::endl;
        g_config.partitionedConvolution.enabled = false;
        g_upsampler->setPartitionedConvolutionConfig(g_config.partitionedConvolution);
    }
    return build_ok_response(request, "Phase type set to " + phaseStr);
}

static void register_zmq_handlers() {
    g_zmq_server->registerCommand("PING", handle_ping);
    g_zmq_server->registerCommand("RELOAD", handle_reload);
    g_zmq_server->registerCommand("STATS", handle_stats_command);
    g_zmq_server->registerCommand("CROSSFEED_ENABLE", handle_crossfeed_enable);
    g_zmq_server->registerCommand("CROSSFEED_DISABLE", handle_crossfeed_disable);
    g_zmq_server->registerCommand("CROSSFEED_STATUS", handle_crossfeed_status);
    g_zmq_server->registerCommand("CROSSFEED_GET_STATUS", handle_crossfeed_get_status);
    g_zmq_server->registerCommand("CROSSFEED_SET_COMBINED", handle_crossfeed_set_combined);
    g_zmq_server->registerCommand("CROSSFEED_GENERATE_WOODWORTH", handle_crossfeed_generate);
    g_zmq_server->registerCommand("CROSSFEED_SET_SIZE", handle_crossfeed_set_size);
    g_zmq_server->registerCommand("DAC_LIST", handle_dac_list);
    g_zmq_server->registerCommand("DAC_STATUS", handle_dac_status);
    g_zmq_server->registerCommand("DAC_SELECT", handle_dac_select);
    g_zmq_server->registerCommand("DAC_RESCAN", handle_dac_rescan);
    g_zmq_server->registerCommand("OUTPUT_MODE_GET", handle_output_mode_get);
    g_zmq_server->registerCommand("OUTPUT_MODE_SET", handle_output_mode_set);
    g_zmq_server->registerCommand("PHASE_TYPE_GET", handle_phase_type_get);
    g_zmq_server->registerCommand("PHASE_TYPE_SET", handle_phase_type_set);

    const std::vector<std::string> rtpCommands = {
        "RTP_START_SESSION",    "RTP_STOP_SESSION", "RTP_LIST_SESSIONS", "RTP_GET_SESSION",
        "RTP_DISCOVER_STREAMS", "StartSession",     "StopSession",       "ListSessions",
        "GetSession",           "DiscoverStreams"};
    for (const auto& cmd : rtpCommands) {
        g_zmq_server->registerCommand(cmd, handle_rtp_command);
    }
}

static bool start_zmq_server() {
    g_zmq_server = std::make_unique<daemon_ipc::ZmqCommandServer>();
    register_zmq_handlers();
    if (g_zmq_server->start()) {
        return true;
    }

    g_zmq_bind_failed.store(true, std::memory_order_release);
    g_running = false;
    if (g_pw_loop) {
        pw_main_loop_quit(g_pw_loop);
    }
    return false;
}

// PipeWire objects
// PipeWire objects
struct Data {
    struct pw_main_loop* loop;
    struct pw_stream* input_stream;
    struct spa_source* signal_check_timer;  // Timer for checking signal flags
    bool gpu_ready;
    shutdown_manager::ShutdownManager* shutdownManager;
};

// PipeWire timer callback to check for pending signals
// Called periodically (every 100ms) from the PipeWire main loop.
static void on_signal_check_timer(void* userdata, uint64_t expirations) {
    (void)expirations;

    auto* context = reinterpret_cast<Data*>(userdata);
    if (context && context->shutdownManager) {
        context->shutdownManager->tick(true);
    }
}

// Input stream process callback (44.1kHz audio from PipeWire)
static void process_interleaved_block(const float* input_samples, uint32_t n_frames) {
    if (!input_samples || n_frames == 0 || !g_upsampler) {
        return;
    }

    if (!g_output_ready.load(std::memory_order_acquire)) {
        static auto last_warn = std::chrono::steady_clock::now() -
                                std::chrono::seconds(6);  // allow immediate first log
        auto now = std::chrono::steady_clock::now();
        if (now - last_warn > std::chrono::seconds(5)) {
            LOG_DEBUG("Dropping input: DAC not ready");
            last_warn = now;
        }
        return;
    }

    if (g_streaming_cache_manager) {
        g_streaming_cache_manager->handleInputBlock();
    }

    std::lock_guard<std::mutex> inputLock(g_input_process_mutex);

    std::vector<float> left(n_frames);
    std::vector<float> right(n_frames);
    AudioUtils::deinterleaveStereo(input_samples, left.data(), right.data(), n_frames);
    float inputPeak = compute_stereo_peak(left.data(), right.data(), n_frames);
    runtime_stats::updateInputPeak(inputPeak);

    StreamFloatVector& output_left = g_upsampler_output_left;
    StreamFloatVector& output_right = g_upsampler_output_right;

    bool use_fallback = g_fallback_active.load(std::memory_order_relaxed);
    bool left_generated = false;
    bool right_generated = false;

    if (use_fallback) {
        size_t output_frames = static_cast<size_t>(n_frames) * g_config.upsampleRatio;
        output_left.assign(output_frames, 0.0f);
        output_right.assign(output_frames, 0.0f);
        for (size_t i = 0; i < n_frames; ++i) {
            output_left[i * g_config.upsampleRatio] = left[i];
            output_right[i * g_config.upsampleRatio] = right[i];
        }
        left_generated = true;
        right_generated = true;
    } else {
        left_generated = g_upsampler->processStreamBlock(
            left.data(), n_frames, output_left, g_upsampler->streamLeft_, g_stream_input_left,
            g_stream_accumulated_left);
        right_generated = g_upsampler->processStreamBlock(
            right.data(), n_frames, output_right, g_upsampler->streamRight_, g_stream_input_right,
            g_stream_accumulated_right);
    }

    if (!left_generated || !right_generated) {
        return;
    }

    size_t frames_generated = std::min(output_left.size(), output_right.size());
    if (frames_generated > 0) {
        float upsamplerPeak =
            compute_stereo_peak(output_left.data(), output_right.data(), frames_generated);
        runtime_stats::updateUpsamplerPeak(upsamplerPeak);
    }

    if (g_crossfeed_enabled.load()) {
        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
        if (g_hrtf_processor && g_hrtf_processor->isEnabled()) {
            bool cf_generated = g_hrtf_processor->processStreamBlock(
                output_left.data(), output_right.data(), output_left.size(), g_cf_output_left,
                g_cf_output_right, 0, g_cf_stream_input_left, g_cf_stream_input_right,
                g_cf_stream_accumulated_left, g_cf_stream_accumulated_right);

            if (cf_generated) {
                size_t cf_frames = std::min(g_cf_output_left.size(), g_cf_output_right.size());
                if (cf_frames > 0) {
                    float cfPeak = compute_stereo_peak(g_cf_output_left.data(),
                                                       g_cf_output_right.data(), cf_frames);
                    runtime_stats::updatePostCrossfeedPeak(cfPeak);
                }
                std::lock_guard<std::mutex> lock(g_buffer_mutex);
                enqueue_output_frames_locked(g_cf_output_left, g_cf_output_right);
                g_buffer_cv.notify_one();
                return;
            }
        }
        // Crossfeed disabled or not ready: fall through and store original upsampler output.
    }

    std::lock_guard<std::mutex> lock(g_buffer_mutex);
    enqueue_output_frames_locked(output_left, output_right);
    g_buffer_cv.notify_one();
    if (frames_generated > 0) {
        float postPeak =
            compute_stereo_peak(output_left.data(), output_right.data(), frames_generated);
        runtime_stats::updatePostCrossfeedPeak(postPeak);
    }
}

static void on_input_process(void* userdata) {
    Data* data = static_cast<Data*>(userdata);

    struct pw_buffer* buf = pw_stream_dequeue_buffer(data->input_stream);
    if (!buf) {
        return;
    }

    struct spa_buffer* spa_buf = buf->buffer;
    float* input_samples = static_cast<float*>(spa_buf->datas[0].data);
    uint32_t n_frames = spa_buf->datas[0].chunk->size / (sizeof(float) * CHANNELS);

    if (input_samples && n_frames > 0 && data->gpu_ready) {
        process_interleaved_block(input_samples, n_frames);
    }

    pw_stream_queue_buffer(data->input_stream, buf);
}

// Stream state changed callback
static void on_stream_state_changed(void* userdata, enum pw_stream_state old_state,
                                    enum pw_stream_state state, const char* error) {
    (void)userdata;
    (void)old_state;

    std::cout << "PipeWire input stream state: " << pw_stream_state_as_string(state);
    if (error) {
        std::cout << " (error: " << error << ")";
    }
    std::cout << std::endl;
}

// PipeWire format/param changed callback (Issue #219)
// Detects input sample rate changes from the source stream
static void on_param_changed(void* userdata, uint32_t id, const struct spa_pod* param) {
    (void)userdata;

    // Only process format changes
    if (id != SPA_PARAM_Format || param == nullptr) {
        return;
    }

    // Parse audio format info
    struct spa_audio_info_raw info;
    if (spa_format_audio_raw_parse(param, &info) < 0) {
        return;
    }

    int detected_rate = static_cast<int>(info.rate);
    int current_rate = g_current_input_rate.load(std::memory_order_acquire);

    // Check if rate actually changed
    if (detected_rate != current_rate && detected_rate > 0) {
        LOG_INFO("[PipeWire] Sample rate change detected: {} -> {} Hz", current_rate,
                 detected_rate);
        // Set pending rate change for main loop to process
        // (avoid blocking in real-time callback)
        g_pending_rate_change.store(detected_rate, std::memory_order_release);
    }
}

static const struct pw_stream_events input_stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
    .param_changed = on_param_changed,
    .process = on_input_process,
};

// Handle sample rate change detected by PipeWire (Issue #219)
// This function is called from the main loop when g_pending_rate_change is set
static bool handle_rate_change(int detected_sample_rate) {
    if (!g_upsampler) {
        return false;
    }

    // Multi-rate mode: use switchToInputRate() for dynamic rate switching
    if (!g_upsampler->isMultiRateEnabled()) {
        LOG_ERROR("[Rate] Multi-rate mode not enabled. Rate switching requires multi-rate mode.");
        return false;
    }

    // Save previous state for rollback
    int prev_input_rate = g_current_input_rate.load(std::memory_order_acquire);
    int prev_output_rate = g_current_output_rate.load(std::memory_order_acquire);

    bool switch_success = false;

    // Use soft mute for filter switching (fade-out, switch, fade-in)
    applySoftMuteForFilterSwitch([&]() {
        // Perform filter switch
        if (!g_upsampler->switchToInputRate(detected_sample_rate)) {
            LOG_ERROR("[Rate] Failed to switch to input rate: {} Hz", detected_sample_rate);
            return false;
        }

        g_current_input_rate.store(detected_sample_rate, std::memory_order_release);

        // Output rate is dynamically calculated (input_rate × upsample_ratio)
        int new_output_rate = g_upsampler->getOutputSampleRate();
        g_current_output_rate.store(new_output_rate, std::memory_order_release);

        // Update rate family
        g_set_rate_family(ConvolutionEngine::detectRateFamily(detected_sample_rate));

        // Re-initialize streaming after rate switch (Issue #219)
        if (!g_upsampler->initializeStreaming()) {
            LOG_ERROR("[Rate] Failed to reinitialize streaming, rolling back...");
            // Rollback: switch back to previous rate
            if (g_upsampler->switchToInputRate(prev_input_rate)) {
                g_upsampler->initializeStreaming();
            }
            g_current_input_rate.store(prev_input_rate, std::memory_order_release);
            g_current_output_rate.store(prev_output_rate, std::memory_order_release);
            return false;
        }

        // Resize streaming input buffers based on new streamValidInputPerBlock (Issue #219)
        size_t new_capacity = g_upsampler->getStreamValidInputPerBlock() * 2;
        g_stream_input_left.resize(new_capacity, 0.0f);
        g_stream_input_right.resize(new_capacity, 0.0f);

        // Clear accumulated samples (old rate data is invalid)
        g_stream_accumulated_left = 0;
        g_stream_accumulated_right = 0;

        LOG_INFO("[Rate] Streaming buffers resized to {} samples (streamValidInputPerBlock={})",
                 new_capacity, g_upsampler->getStreamValidInputPerBlock());

        // Clear output ring buffers to discard old rate samples (Issue #219)
        g_output_buffer_left.clear();
        g_output_buffer_right.clear();
        LOG_INFO("[Rate] Output ring buffers cleared");

        // Re-apply EQ if enabled (Issue #219: EQ state auto-restore)
        // EQ magnitude depends on output sample rate, so we must recalculate
        if (g_config.eqEnabled && !g_config.eqProfilePath.empty()) {
            EQ::EqProfile eqProfile;
            if (EQ::parseEqFile(g_config.eqProfilePath, eqProfile)) {
                size_t filterFftSize = g_upsampler->getFilterFftSize();
                size_t fullFftSize = g_upsampler->getFullFftSize();
                double outputSampleRate = static_cast<double>(new_output_rate);
                auto eqMagnitude = EQ::computeEqMagnitudeForFft(filterFftSize, fullFftSize,
                                                                outputSampleRate, eqProfile);

                if (g_upsampler->applyEqMagnitude(eqMagnitude)) {
                    LOG_INFO("[Rate] EQ re-applied for new output rate {} Hz", new_output_rate);
                } else {
                    LOG_WARN("[Rate] Failed to re-apply EQ after rate switch");
                }
            }
        }

        // Schedule ALSA reconfiguration if output rate changed
        if (new_output_rate != prev_output_rate) {
            g_alsa_reconfigure_needed.store(true, std::memory_order_release);
            g_alsa_new_output_rate.store(new_output_rate, std::memory_order_release);
            LOG_INFO("[Rate] ALSA reconfiguration scheduled for {} Hz ({} -> {}x upsampling)",
                     new_output_rate, detected_sample_rate, g_upsampler->getUpsampleRatio());
        } else {
            LOG_INFO("[Rate] Rate switched to {} Hz -> {} Hz ({}x upsampling)",
                     detected_sample_rate, new_output_rate, g_upsampler->getUpsampleRatio());
        }

        switch_success = true;
        return true;
    });

    return switch_success;
}

// Open and configure ALSA device. Returns nullptr on failure.
static snd_pcm_t* open_and_configure_pcm(const std::string& device) {
    snd_pcm_t* pcm_handle = nullptr;
    int err;

    if (device.empty()) {
        std::cerr << "ALSA: No output device selected yet" << std::endl;
        return nullptr;
    }

    err = snd_pcm_open(&pcm_handle, device.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        std::cerr << "ALSA: Cannot open device " << device << ": " << snd_strerror(err)
                  << std::endl;
        return nullptr;
    }

    // Set hardware parameters
    snd_pcm_hw_params_t* hw_params;
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(pcm_handle, hw_params);

    if ((err = snd_pcm_hw_params_set_access(pcm_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) <
            0 ||
        (err = snd_pcm_hw_params_set_format(pcm_handle, hw_params, SND_PCM_FORMAT_S32_LE)) < 0) {
        std::cerr << "ALSA: Cannot set access/format: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    int configuredOutputRate = g_current_output_rate.load(std::memory_order_acquire);
    if (configuredOutputRate <= 0) {
        configuredOutputRate = g_input_sample_rate * g_config.upsampleRatio;
    }
    unsigned int rate = static_cast<unsigned int>(configuredOutputRate);
    if ((err = snd_pcm_hw_params_set_rate_near(pcm_handle, hw_params, &rate, 0)) < 0) {
        std::cerr << "ALSA: Cannot set sample rate: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return nullptr;
    }
    if ((err = snd_pcm_hw_params_set_channels(pcm_handle, hw_params, CHANNELS)) < 0) {
        std::cerr << "ALSA: Cannot set channel count: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    snd_pcm_uframes_t buffer_size = static_cast<snd_pcm_uframes_t>(g_config.bufferSize);
    snd_pcm_uframes_t period_size = static_cast<snd_pcm_uframes_t>(g_config.periodSize);
    snd_pcm_hw_params_set_buffer_size_near(pcm_handle, hw_params, &buffer_size);
    snd_pcm_hw_params_set_period_size_near(pcm_handle, hw_params, &period_size, 0);

    if ((err = snd_pcm_hw_params(pcm_handle, hw_params)) < 0) {
        std::cerr << "ALSA: Cannot set hardware parameters: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    if ((err = snd_pcm_prepare(pcm_handle)) < 0) {
        std::cerr << "ALSA: Cannot prepare device: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    // Set software parameters for XRUN detection (Issue #139)
    snd_pcm_sw_params_t* sw_params;
    snd_pcm_sw_params_alloca(&sw_params);
    if (snd_pcm_sw_params_current(pcm_handle, sw_params) == 0) {
        // Enable XRUN detection
        snd_pcm_sw_params_set_start_threshold(pcm_handle, sw_params, buffer_size);
        snd_pcm_sw_params_set_avail_min(pcm_handle, sw_params, period_size);
        if (snd_pcm_sw_params(pcm_handle, sw_params) < 0) {
            std::cerr << "ALSA: Warning - Failed to set software parameters" << std::endl;
        }
    }

    std::cout << "ALSA: Output device " << device << " configured (" << rate
              << " Hz, 32-bit int, stereo)" << " buffer " << buffer_size << " frames, period "
              << period_size << " frames" << std::endl;
    return pcm_handle;
}

// Reconfigure ALSA PCM for new sample rate (Issue #219)
// Closes the old handle and opens a new one with the specified rate.
// Returns new handle on success, nullptr on failure (old handle is closed regardless).
static snd_pcm_t* reconfigure_alsa(snd_pcm_t* old_handle, const std::string& device,
                                   int new_sample_rate) {
    // Drop any pending samples and close old handle
    if (old_handle) {
        snd_pcm_drop(old_handle);
        snd_pcm_close(old_handle);
        LOG_INFO("[ALSA] Closed old PCM handle for reconfiguration");
    }

    snd_pcm_t* pcm_handle = nullptr;
    int err;

    if (device.empty()) {
        LOG_ERROR("[ALSA] Cannot reconfigure: no active device");
        return nullptr;
    }

    err = snd_pcm_open(&pcm_handle, device.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        LOG_ERROR("[ALSA] Cannot open device {} for reconfiguration: {}", device,
                  snd_strerror(err));
        return nullptr;
    }

    // Set hardware parameters
    snd_pcm_hw_params_t* hw_params;
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(pcm_handle, hw_params);

    if ((err = snd_pcm_hw_params_set_access(pcm_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) <
            0 ||
        (err = snd_pcm_hw_params_set_format(pcm_handle, hw_params, SND_PCM_FORMAT_S32_LE)) < 0) {
        LOG_ERROR("[ALSA] Cannot set access/format: {}", snd_strerror(err));
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    // Use the new sample rate
    unsigned int rate = static_cast<unsigned int>(new_sample_rate);
    if ((err = snd_pcm_hw_params_set_rate_near(pcm_handle, hw_params, &rate, 0)) < 0) {
        LOG_ERROR("[ALSA] Cannot set sample rate {} Hz: {}", new_sample_rate, snd_strerror(err));
        snd_pcm_close(pcm_handle);
        return nullptr;
    }
    if ((err = snd_pcm_hw_params_set_channels(pcm_handle, hw_params, CHANNELS)) < 0) {
        LOG_ERROR("[ALSA] Cannot set channel count: {}", snd_strerror(err));
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    snd_pcm_uframes_t buffer_size = static_cast<snd_pcm_uframes_t>(g_config.bufferSize);
    snd_pcm_uframes_t period_size = static_cast<snd_pcm_uframes_t>(g_config.periodSize);
    snd_pcm_hw_params_set_buffer_size_near(pcm_handle, hw_params, &buffer_size);
    snd_pcm_hw_params_set_period_size_near(pcm_handle, hw_params, &period_size, 0);

    if ((err = snd_pcm_hw_params(pcm_handle, hw_params)) < 0) {
        LOG_ERROR("[ALSA] Cannot set hardware parameters: {}", snd_strerror(err));
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    if ((err = snd_pcm_prepare(pcm_handle)) < 0) {
        LOG_ERROR("[ALSA] Cannot prepare device: {}", snd_strerror(err));
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    // Set software parameters for XRUN detection
    snd_pcm_sw_params_t* sw_params;
    snd_pcm_sw_params_alloca(&sw_params);
    if (snd_pcm_sw_params_current(pcm_handle, sw_params) == 0) {
        snd_pcm_sw_params_set_start_threshold(pcm_handle, sw_params, buffer_size);
        snd_pcm_sw_params_set_avail_min(pcm_handle, sw_params, period_size);
        if (snd_pcm_sw_params(pcm_handle, sw_params) < 0) {
            LOG_WARN("[ALSA] Failed to set software parameters");
        }
    }

    LOG_INFO(
        "[ALSA] Reconfigured for {} Hz (32-bit int, stereo), buffer {} frames, period {} frames",
        rate, buffer_size, period_size);
    return pcm_handle;
}

// Check current PCM state; return false if disconnected/suspended
static bool pcm_alive(snd_pcm_t* pcm_handle) {
    if (!pcm_handle)
        return false;
    snd_pcm_status_t* status;
    snd_pcm_status_alloca(&status);
    if (snd_pcm_status(pcm_handle, status) < 0) {
        return false;
    }
    snd_pcm_state_t st = snd_pcm_status_get_state(status);
    if (st == SND_PCM_STATE_DISCONNECTED || st == SND_PCM_STATE_SUSPENDED) {
        return false;
    }
    return true;
}

// ALSA output thread (705.6kHz direct to DAC)
void alsa_output_thread() {
    elevate_realtime_priority("ALSA output");

    std::string currentDevice = g_dac_manager.waitForSelection();
    snd_pcm_t* pcm_handle = nullptr;
    while (g_running && !pcm_handle) {
        if (currentDevice.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            currentDevice = g_dac_manager.waitForSelection();
            continue;
        }
        pcm_handle = open_and_configure_pcm(currentDevice);
        if (!pcm_handle) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            currentDevice = g_dac_manager.waitForSelection();
        } else {
            mark_dac_connected(currentDevice);
        }
    }
    std::vector<int32_t> interleaved_buffer(32768 * CHANNELS);  // resized after open
    std::vector<float> float_buffer(32768 * CHANNELS);          // for soft mute processing
    snd_pcm_uframes_t period_size = 32768;
    if (!pcm_handle) {
        return;
    }
    snd_pcm_hw_params_t* cur_params;
    snd_pcm_hw_params_alloca(&cur_params);
    if (snd_pcm_hw_params_current(pcm_handle, cur_params) == 0) {
        snd_pcm_uframes_t detected_period = 0;
        if (snd_pcm_hw_params_get_period_size(cur_params, &detected_period, nullptr) == 0 &&
            detected_period > 0) {
            period_size = detected_period;
            interleaved_buffer.resize(period_size * CHANNELS);
            float_buffer.resize(period_size * CHANNELS);
        }
    }

    // Main playback loop
    while (g_running) {
        // Heartbeat check every few hundred loops
        static int alive_counter = 0;
        if (++alive_counter > 200) {  // ~200 iterations ~ a few seconds depending on buffer wait
            alive_counter = 0;
            if (!pcm_alive(pcm_handle)) {
                std::cerr << "ALSA: PCM disconnected/suspended, attempting reopen..." << std::endl;
                if (pcm_handle) {
                    snd_pcm_close(pcm_handle);
                    pcm_handle = nullptr;
                }
                mark_dac_disconnected(currentDevice);
                while (g_running && !pcm_handle) {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    currentDevice = g_dac_manager.waitForSelection();
                    if (currentDevice.empty())
                        continue;
                    pcm_handle = open_and_configure_pcm(currentDevice);
                }
                if (pcm_handle) {
                    mark_dac_connected(currentDevice,
                                       "DAC reconnected - resuming input processing");
                }
                // Reset buffer positions to avoid backlog after long downtime
                {
                    std::lock_guard<std::mutex> lock(g_buffer_mutex);
                    g_output_buffer_left.clear();
                    g_output_buffer_right.clear();
                    g_output_read_pos = 0;
                }
                continue;
            }
        }

        // Issue #219: Check for pending rate change from PipeWire callback
        int pending_rate = g_pending_rate_change.exchange(0, std::memory_order_acquire);
        if (pending_rate > 0 && g_config.multiRateEnabled) {
            LOG_INFO("[Main] Processing pending rate change to {} Hz", pending_rate);
            if (!handle_rate_change(pending_rate)) {
                LOG_ERROR("[Main] Failed to handle rate change to {} Hz", pending_rate);
            }
        }

        // Issue #219: Check for pending ALSA reconfiguration (output rate changed)
        if (g_alsa_reconfigure_needed.exchange(false, std::memory_order_acquire)) {
            int new_output_rate = g_alsa_new_output_rate.load(std::memory_order_acquire);
            if (new_output_rate > 0) {
                LOG_INFO("[Main] Reconfiguring ALSA for new output rate {} Hz", new_output_rate);

                // Reconfigure ALSA with new rate
                snd_pcm_t* new_handle =
                    reconfigure_alsa(pcm_handle, currentDevice, new_output_rate);
                if (new_handle) {
                    pcm_handle = new_handle;

                    // Update soft mute sample rate
                    if (g_soft_mute) {
                        g_soft_mute->setSampleRate(new_output_rate);
                    }

                    LOG_INFO("[Main] ALSA reconfiguration successful");
                } else {
                    // Failed to reconfigure - try to reopen with old rate
                    LOG_ERROR("[Main] ALSA reconfiguration failed, attempting recovery...");
                    int old_rate = g_current_output_rate.load(std::memory_order_acquire);
                    pcm_handle = reconfigure_alsa(nullptr, currentDevice, old_rate);
                    if (!pcm_handle) {
                        LOG_ERROR("[Main] ALSA recovery failed, waiting for reconnect...");
                    }
                }
            }
        }

        if (auto pendingDevice = g_dac_manager.consumePendingChange()) {
            std::string nextDevice = *pendingDevice;
            if (!nextDevice.empty() && nextDevice != currentDevice) {
                std::cout << "ALSA: Switching output to " << nextDevice << std::endl;
                if (pcm_handle) {
                    snd_pcm_close(pcm_handle);
                    pcm_handle = nullptr;
                    mark_dac_disconnected(currentDevice);
                }
                currentDevice = nextDevice;
                while (g_running && !pcm_handle) {
                    pcm_handle = open_and_configure_pcm(currentDevice);
                    if (!pcm_handle) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        currentDevice = g_dac_manager.waitForSelection();
                        if (currentDevice.empty())
                            break;
                    }
                }
                if (!pcm_handle) {
                    continue;
                }
                mark_dac_connected(currentDevice, "DAC reconnected - resuming input processing");
            }
        }

        // Wait for GPU processed data (dynamic threshold to avoid underflow with crossfeed)
        std::unique_lock<std::mutex> lock(g_buffer_mutex);
        size_t ready_threshold = get_playback_ready_threshold(static_cast<size_t>(period_size));
        g_buffer_cv.wait_for(lock, std::chrono::milliseconds(200), [ready_threshold] {
            return (g_output_buffer_left.size() - g_output_read_pos) >= ready_threshold ||
                   !g_running;
        });

        if (!g_running)
            break;

        size_t available = g_output_buffer_left.size() - g_output_read_pos;
        if (available >= period_size) {
            // Step 1: Interleave L/R channels into float buffer with gain applied
            const float gain = g_output_gain.load(std::memory_order_relaxed);
            AudioUtils::interleaveStereoWithGain(g_output_buffer_left.data() + g_output_read_pos,
                                                 g_output_buffer_right.data() + g_output_read_pos,
                                                 float_buffer.data(), period_size, gain);

            // Step 2: Apply soft mute for glitch-free shutdown/transitions
            if (g_soft_mute) {
                g_soft_mute->process(float_buffer.data(), period_size);

                // Reset fade duration to default after filter switch fade-in completes
                // Check if we just completed a fade-in from filter switching
                // (fade duration > default indicates filter switch was in progress)
                using namespace DaemonConstants;
                if (g_soft_mute->getState() == SoftMute::MuteState::PLAYING &&
                    g_soft_mute->getFadeDuration() > DEFAULT_SOFT_MUTE_FADE_MS) {
                    g_soft_mute->setFadeDuration(DEFAULT_SOFT_MUTE_FADE_MS);
                }
            }

            // Apply limiter and track peak after gain & soft mute
            float postGainPeak = apply_output_limiter(float_buffer.data(), period_size);
            runtime_stats::updatePostGainPeak(postGainPeak);

            // Step 3: Clipping detection, clamping, and float→int32 conversion
            static auto last_stats_write = std::chrono::steady_clock::now();
            size_t current_clips = 0;

            for (size_t i = 0; i < period_size; ++i) {
                float left_sample = float_buffer[i * 2];
                float right_sample = float_buffer[i * 2 + 1];

                // Detect and count clipping (for diagnostics only)
                if (left_sample > 1.0f || left_sample < -1.0f || right_sample > 1.0f ||
                    right_sample < -1.0f) {
                    current_clips++;
                    runtime_stats::recordClip();
                }

                // Hard clipping as safety net only
                left_sample = std::clamp(left_sample, -1.0f, 1.0f);
                right_sample = std::clamp(right_sample, -1.0f, 1.0f);

                // Float to int32 conversion
                constexpr float INT32_MAX_FLOAT = 2147483647.0f;
                interleaved_buffer[i * 2] =
                    static_cast<int32_t>(std::lroundf(left_sample * INT32_MAX_FLOAT));
                interleaved_buffer[i * 2 + 1] =
                    static_cast<int32_t>(std::lroundf(right_sample * INT32_MAX_FLOAT));
            }

            runtime_stats::addSamples(period_size * 2);

            // Report clipping infrequently to avoid log spam
            size_t total = runtime_stats::totalSamples();
            size_t clips = runtime_stats::clipCount();
            if (total % (period_size * 2 * 100) == 0 && clips > 0) {
                std::cout << "WARNING: Clipping detected - " << clips << " samples clipped out of "
                          << total << " (" << (100.0 * clips / total) << "%)" << std::endl;
            }

            // Write stats file every second
            auto now = std::chrono::steady_clock::now();
            if (now - last_stats_write >= std::chrono::seconds(1)) {
                runtime_stats::writeStatsFile(build_runtime_stats_dependencies(),
                                              get_max_output_buffer_frames(), STATS_FILE_PATH);
                last_stats_write = now;
            }
            g_output_read_pos += period_size;

            size_t cleanupThreshold =
                std::max<size_t>(static_cast<size_t>(period_size) * 4, static_cast<size_t>(1));
            trim_output_buffer_locked(cleanupThreshold);

            lock.unlock();

            // Write to ALSA device
            snd_pcm_sframes_t frames_written =
                snd_pcm_writei(pcm_handle, interleaved_buffer.data(), period_size);
            if (frames_written < 0) {
                // Check for XRUN (Issue #139)
                if (frames_written == -EPIPE) {
                    // XRUN detected - notify fallback manager
                    if (g_fallback_manager) {
                        g_fallback_manager->notifyXrun();
                    }
                    gpu_upsampler::metrics::recordXrun();
                    LOG_WARN("ALSA: XRUN detected");
                }

                // Try standard recovery first
                snd_pcm_sframes_t rec = snd_pcm_recover(pcm_handle, frames_written, 0);
                if (rec < 0) {
                    // Device may be gone; attempt reopen
                    std::cerr << "ALSA: Write error: " << snd_strerror(frames_written)
                              << " (recover=" << snd_strerror(rec) << "), retrying reopen..."
                              << std::endl;
                    snd_pcm_close(pcm_handle);
                    pcm_handle = nullptr;
                    mark_dac_disconnected(currentDevice);
                    while (g_running && !pcm_handle) {
                        std::this_thread::sleep_for(std::chrono::seconds(5));
                        std::string next = g_dac_manager.getSelectedDevice();
                        if (!next.empty()) {
                            currentDevice = next;
                        }
                        if (currentDevice.empty())
                            continue;
                        pcm_handle = open_and_configure_pcm(currentDevice);
                    }
                    if (pcm_handle) {
                        mark_dac_connected(currentDevice,
                                           "DAC reconnected - resuming input processing");
                        // resize buffer to new period size if needed
                        snd_pcm_uframes_t new_period = 0;
                        snd_pcm_hw_params_t* hw_params;
                        snd_pcm_hw_params_alloca(&hw_params);
                        if (snd_pcm_hw_params_current(pcm_handle, hw_params) == 0 &&
                            snd_pcm_hw_params_get_period_size(hw_params, &new_period, nullptr) ==
                                0 &&
                            new_period != period_size) {
                            period_size = new_period;
                            interleaved_buffer.resize(period_size * CHANNELS);
                            float_buffer.resize(period_size * CHANNELS);
                        }
                        // Drop queued buffers on successful reopen to avoid burst
                        {
                            std::lock_guard<std::mutex> lock(g_buffer_mutex);
                            g_output_buffer_left.clear();
                            g_output_buffer_right.clear();
                            g_output_read_pos = 0;
                        }
                    } else {
                        // If still not available, continue loop to retry
                        continue;
                    }
                }
            }
        } else {
            // Not enough data → write silence to keep device fed and avoid xrun
            lock.unlock();

            // Apply soft mute even to silence (to advance fade position during shutdown)
            if (g_soft_mute && g_soft_mute->isTransitioning()) {
                std::fill(float_buffer.begin(), float_buffer.end(), 0.0f);
                g_soft_mute->process(float_buffer.data(), period_size);
            }

            std::fill(interleaved_buffer.begin(), interleaved_buffer.end(), 0);
            snd_pcm_sframes_t frames_written =
                snd_pcm_writei(pcm_handle, interleaved_buffer.data(), period_size);
            if (frames_written < 0) {
                snd_pcm_sframes_t rec = snd_pcm_recover(pcm_handle, frames_written, 0);
                if (rec < 0) {
                    std::cerr << "ALSA: Silence write error: " << snd_strerror(frames_written)
                              << " (recover=" << snd_strerror(rec) << "), retrying reopen..."
                              << std::endl;
                    if (pcm_handle) {
                        snd_pcm_close(pcm_handle);
                        pcm_handle = nullptr;
                    }
                    mark_dac_disconnected(currentDevice);
                    while (g_running && !pcm_handle) {
                        std::this_thread::sleep_for(std::chrono::seconds(5));
                        std::string next = g_dac_manager.getSelectedDevice();
                        if (!next.empty()) {
                            currentDevice = next;
                        }
                        if (currentDevice.empty())
                            continue;
                        pcm_handle = open_and_configure_pcm(currentDevice);
                    }
                    if (pcm_handle) {
                        mark_dac_connected(currentDevice,
                                           "DAC reconnected - resuming input processing");
                    }
                }
            }
        }
    }

    // Cleanup
    if (pcm_handle) {
        snd_pcm_drain(pcm_handle);
        snd_pcm_close(pcm_handle);
        mark_dac_disconnected(currentDevice);
    }
    std::cout << "ALSA: Output thread terminated" << std::endl;
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
        &g_soft_mute,
        &g_running,
        &g_reload_requested,
        &g_main_loop_running,
    };
    shutdown_manager::ShutdownManager shutdownManager(shutdownDeps);
    shutdownManager.installSignalHandlers();
    shutdownManager.setQuitLoopCallback([]() {
        if (g_main_loop_running.load() && g_pw_loop) {
            pw_main_loop_quit(g_pw_loop);
        }
    });

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
            std::cout << "Config: ALSA_DEVICE env override: " << env_dev << std::endl;
        }

        // Command line argument overrides filter path
        if (argc > 1) {
            g_config.filterPath = argv[1];
            std::cout << "Config: CLI filter path override: " << argv[1] << std::endl;
        }

        // Determine PipeWire mode (config + env overrides)
        bool pipewireEnabled = g_config.pipewireEnabled;
        std::string pipewireOverrideReason;
        auto toLower = [](std::string value) {
            std::transform(value.begin(), value.end(), value.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return value;
        };
        if (const char* env_mode = std::getenv("MAGICBOX_INPUT_MODE")) {
            std::string mode = toLower(env_mode);
            if (mode == "rtp") {
                pipewireEnabled = false;
                pipewireOverrideReason = "MAGICBOX_INPUT_MODE=rtp";
            } else if (mode == "pipewire") {
                pipewireEnabled = true;
                pipewireOverrideReason = "MAGICBOX_INPUT_MODE=pipewire";
            }
        }
        auto disableFromEnv = [&](const char* envName) {
            if (const char* value = std::getenv(envName)) {
                std::string flag = toLower(value);
                if (flag == "1" || flag == "true" || flag == "yes") {
                    pipewireEnabled = false;
                    pipewireOverrideReason = std::string(envName) + "=true";
                }
            }
        };
        disableFromEnv("PIPEWIRE_DISABLED");
        disableFromEnv("MAGICBOX_DISABLE_PIPEWIRE");
        auto enableFromEnv = [&](const char* envName) {
            if (const char* value = std::getenv(envName)) {
                std::string flag = toLower(value);
                if (flag == "1" || flag == "true" || flag == "yes") {
                    pipewireEnabled = true;
                    pipewireOverrideReason = std::string(envName) + "=true";
                }
            }
        };
        enableFromEnv("PIPEWIRE_ENABLED");
        enableFromEnv("MAGICBOX_ENABLE_PIPEWIRE");
        if (!pipewireEnabled) {
            std::cout << "Config: PipeWire input disabled ("
                      << (pipewireOverrideReason.empty() ? "config.json" : pipewireOverrideReason)
                      << "). Running in RTP-only mode." << std::endl;
        }

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
                              << std::endl;
                    g_config.filterPath = rateSpecificPath;
                }
            }
        }

        // Warn if using 44.1kHz filter with 48kHz input
        if (g_input_sample_rate == 48000 &&
            g_config.filterPath.find("44100") == std::string::npos &&
            g_config.filterPath.find("48000") == std::string::npos) {
            std::cout << "Warning: Using generic filter with 48kHz input. "
                      << "For optimal quality, generate a 48kHz-optimized filter." << std::endl;
        }

        // Initialize GPU upsampler with configured values
        std::cout << "Initializing GPU upsampler..." << std::endl;
        g_upsampler = new ConvolutionEngine::GPUUpsampler();
        g_upsampler->setPartitionedConvolutionConfig(g_config.partitionedConvolution);

        bool initSuccess = false;
        ConvolutionEngine::RateFamily initialFamily = ConvolutionEngine::RateFamily::RATE_44K;
        if (g_config.multiRateEnabled) {
            std::cout << "Multi-rate mode enabled" << std::endl;
            std::cout << "  Coefficient directory: " << g_config.coefficientDir << std::endl;

            if (!std::filesystem::exists(g_config.coefficientDir)) {
                std::cerr << "Config error: Coefficient directory not found: "
                          << g_config.coefficientDir << std::endl;
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
            std::cout << "Quad-phase mode enabled" << std::endl;

            bool allFilesExist = true;
            for (const auto& path : {g_config.filterPath44kMin, g_config.filterPath48kMin,
                                     g_config.filterPath44kLinear, g_config.filterPath48kLinear}) {
                if (!std::filesystem::exists(path)) {
                    std::cerr << "Config error: Filter file not found: " << path << std::endl;
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
            std::cerr << "Failed to initialize GPU upsampler" << std::endl;
            delete g_upsampler;
            exitCode = 1;
            break;
        }

        // Check for early abort (signal received during GPU initialization)
        if (!g_running) {
            std::cout << "Startup interrupted by signal" << std::endl;
            delete g_upsampler;
            g_upsampler = nullptr;
            break;
        }

        if (g_config.multiRateEnabled) {
            std::cout << "GPU upsampler ready (multi-rate mode, " << g_config.blockSize
                      << " samples/block)" << std::endl;
            std::cout << "  Current input rate: " << g_upsampler->getCurrentInputRate() << " Hz"
                      << std::endl;
            std::cout << "  Upsample ratio: " << g_upsampler->getUpsampleRatio() << "x"
                      << std::endl;
            std::cout << "  Output rate: " << g_upsampler->getOutputSampleRate() << " Hz"
                      << std::endl;
        } else {
            std::cout << "GPU upsampler ready (" << g_config.upsampleRatio << "x upsampling, "
                      << g_config.blockSize << " samples/block)" << std::endl;
        }

        // Set g_active_rate_family and g_active_phase_type for headroom tracking
        if (g_config.multiRateEnabled) {
            // Rate family already set during initializeMultiRate()
            // g_active_rate_family is set via g_set_rate_family() above
        } else {
            g_active_rate_family = initialFamily;
        }

        g_active_phase_type = g_config.phaseType;
        refresh_current_headroom("initial filter load");

        std::cout << "Input sample rate: " << g_upsampler->getInputSampleRate() << " Hz -> "
                  << g_upsampler->getOutputSampleRate() << " Hz output" << std::endl;
        if (!g_config.multiRateEnabled) {
            std::cout << "Phase type: " << phaseTypeToString(g_config.phaseType) << std::endl;
        }

        // Log latency warning for linear phase
        if (g_config.phaseType == PhaseType::Linear) {
            double latencySec = g_upsampler->getLatencySeconds();
            std::cout << "  WARNING: Linear phase latency: " << latencySec << " seconds ("
                      << g_upsampler->getLatencySamples() << " samples)" << std::endl;
        }

        // Initialize streaming mode to preserve overlap buffers across PipeWire callbacks
        if (!g_upsampler->initializeStreaming()) {
            std::cerr << "Failed to initialize streaming mode" << std::endl;
            delete g_upsampler;
            exitCode = 1;
            break;
        }
        PartitionRuntime::applyPartitionPolicy(partitionRequest, *g_upsampler, g_config, "ALSA");

        // Check for early abort
        if (!g_running) {
            std::cout << "Startup interrupted by signal" << std::endl;
            delete g_upsampler;
            g_upsampler = nullptr;
            break;
        }

        // Apply EQ profile if enabled
        if (g_config.eqEnabled && !g_config.eqProfilePath.empty()) {
            std::cout << "Loading EQ profile: " << g_config.eqProfilePath << std::endl;
            EQ::EqProfile eqProfile;
            if (EQ::parseEqFile(g_config.eqProfilePath, eqProfile)) {
                std::cout << "  EQ: " << eqProfile.name << " (" << eqProfile.bands.size()
                          << " bands, preamp " << eqProfile.preampDb << " dB)" << std::endl;

                // Compute EQ magnitude response and apply with minimum phase reconstruction
                size_t filterFftSize = g_upsampler->getFilterFftSize();  // N/2+1 (R2C output)
                size_t fullFftSize = g_upsampler->getFullFftSize();      // N (full FFT)
                double outputSampleRate =
                    static_cast<double>(g_input_sample_rate) * g_config.upsampleRatio;
                auto eqMagnitude = EQ::computeEqMagnitudeForFft(filterFftSize, fullFftSize,
                                                                outputSampleRate, eqProfile);

                if (g_upsampler->applyEqMagnitude(eqMagnitude)) {
                    // Log message depends on phase type (already logged by applyEqMagnitude)
                } else {
                    std::cerr << "  EQ: Failed to apply frequency response" << std::endl;
                }
            } else {
                std::cerr << "  EQ: Failed to parse profile: " << g_config.eqProfilePath
                          << std::endl;
            }
        }

        // Pre-allocate streaming input buffers (based on streamValidInputPerBlock_)
        // Use 2x safety margin to handle timing variations
        size_t buffer_capacity = g_upsampler->getStreamValidInputPerBlock() * 2;
        g_stream_input_left.resize(buffer_capacity, 0.0f);
        g_stream_input_right.resize(buffer_capacity, 0.0f);
        std::cout << "Streaming buffer capacity: " << buffer_capacity
                  << " samples (2x streamValidInputPerBlock)" << std::endl;
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
                std::cout << "Initializing HRTF processor for crossfeed..." << std::endl;
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
                                  << ")" << std::endl;

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
                                  << " samples" << std::endl;

                        // Crossfeed is initialized but disabled by default
                        g_crossfeed_enabled.store(false);
                        g_hrtf_processor->setEnabled(false);
                        std::cout << "  Crossfeed: initialized (disabled by default)" << std::endl;
                    } else {
                        std::cerr << "  HRTF: Failed to initialize streaming mode" << std::endl;
                        delete g_hrtf_processor;
                        g_hrtf_processor = nullptr;
                    }
                } else {
                    std::cerr << "  HRTF: Failed to initialize processor" << std::endl;
                    std::cerr
                        << "  Hint: Run 'uv run python scripts/generate_hrtf.py' to generate HRTF "
                           "filters"
                        << std::endl;
                    delete g_hrtf_processor;
                    g_hrtf_processor = nullptr;
                }
            } else {
                std::cout << "HRTF directory not found (" << hrtfDir
                          << "), crossfeed feature disabled" << std::endl;
                std::cout
                    << "  Hint: Run 'uv run python scripts/generate_hrtf.py' to generate HRTF "
                       "filters"
                    << std::endl;
            }
        } else {
            std::cout << "[Partition] Crossfeed initialization skipped (low-latency mode)"
                      << std::endl;
        }

        std::cout << std::endl;

        // Check for early abort before starting threads
        if (!g_running) {
            std::cout << "Startup interrupted by signal" << std::endl;
            delete g_upsampler;
            g_upsampler = nullptr;
            break;
        }

        // Initialize soft mute controller with output sample rate
        using namespace DaemonConstants;
        int outputSampleRate = g_input_sample_rate * g_config.upsampleRatio;
        g_soft_mute = new SoftMute::Controller(DEFAULT_SOFT_MUTE_FADE_MS, outputSampleRate);
        std::cout << "Soft mute initialized (" << DEFAULT_SOFT_MUTE_FADE_MS << "ms fade at "
                  << outputSampleRate << "Hz)" << std::endl;

        // Initialize RTP engine coordinator
        rtp_engine::RtpEngineCoordinator::Dependencies rtpDeps{};
        rtpDeps.config = &g_config;
        rtpDeps.runningFlag = &g_running;
        rtpDeps.currentInputRate = &g_current_input_rate;
        rtpDeps.currentOutputRate = &g_current_output_rate;
        rtpDeps.alsaReconfigureNeeded = &g_alsa_reconfigure_needed;
        rtpDeps.handleRateChange = handle_rate_change;
        rtpDeps.isUpsamplerReady = []() { return g_upsampler != nullptr; };
        rtpDeps.isMultiRateEnabled = []() {
            return g_upsampler && g_upsampler->isMultiRateEnabled();
        };
        rtpDeps.getUpsampleRatio = []() {
            return g_upsampler ? g_upsampler->getUpsampleRatio() : 0;
        };
        rtpDeps.getInputSampleRate = []() { return g_input_sample_rate; };
        rtpDeps.processInterleaved = [](const float* data, size_t frames, uint32_t sampleRate) {
            (void)sampleRate;
            process_interleaved_block(data, static_cast<uint32_t>(frames));
        };
        rtpDeps.resetStreamingCache = []() {
            if (g_streaming_cache_manager) {
                g_streaming_cache_manager->flushCaches();
            }
        };
        rtpDeps.ptpProvider = []() -> Network::PtpSyncState { return {}; };
        rtpDeps.telemetry = [](const Network::SessionMetrics& metrics) {
            LOG_DEBUG("RTP session {} stats: packets={} dropped={}", metrics.sessionId,
                      metrics.packetsReceived, metrics.packetsDropped);
        };
        g_rtp_coordinator = std::make_unique<rtp_engine::RtpEngineCoordinator>(rtpDeps);

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
                          << std::endl;
            } else {
                std::cerr << "Warning: Failed to initialize fallback manager" << std::endl;
                delete g_fallback_manager;
                g_fallback_manager = nullptr;
                g_fallback_active.store(false, std::memory_order_relaxed);
            }
        } else {
            std::cout << "Fallback manager disabled" << std::endl;
            g_fallback_active.store(false, std::memory_order_relaxed);
        }

        Data data{};
        data.gpu_ready = true;
        struct pw_loop* loop = nullptr;
        bool pipewireInitCalled = false;
        bool pipewireActive = false;

        if (pipewireEnabled) {
            pw_init(&argc, &argv);
            pipewireInitCalled = true;
            data.loop = pw_main_loop_new(nullptr);
            if (!data.loop) {
                std::cerr << "PipeWire: Failed to create main loop. Falling back to RTP-only mode."
                          << std::endl;
            } else {
                g_pw_loop = data.loop;
                loop = pw_main_loop_get_loop(data.loop);
                pipewireActive = true;
            }
        }

        // Start ZeroMQ server after g_pw_loop is set to allow RELOAD to quit the main loop
        start_zmq_server();

        g_dac_manager.initialize();

        // Start ALSA output thread
        g_dac_manager.start();
        std::cout << "Starting ALSA output thread..." << std::endl;
        std::thread alsa_thread(alsa_output_thread);

        if (g_rtp_coordinator) {
            g_rtp_coordinator->startFromConfig();
        }

        if (pipewireActive) {
            std::cout << "Creating PipeWire input (capturing from gpu_upsampler_sink)..."
                      << std::endl;
            data.input_stream = pw_stream_new_simple(
                loop, "GPU Upsampler Input",
                pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio", PW_KEY_MEDIA_CATEGORY, "Capture",
                                  PW_KEY_MEDIA_ROLE, "Music", PW_KEY_NODE_DESCRIPTION,
                                  "GPU Upsampler Input", PW_KEY_TARGET_OBJECT,
                                  "gpu_upsampler_sink.monitor", "audio.channels", "2",
                                  "audio.position", "FL,FR", nullptr),
                &input_stream_events, &data);

            if (!data.input_stream) {
                std::cerr << "PipeWire: Failed to create input stream. Disabling PipeWire path."
                          << std::endl;
                pipewireActive = false;
            } else {
                uint8_t input_buffer[1024];
                struct spa_pod_builder input_builder =
                    SPA_POD_BUILDER_INIT(input_buffer, sizeof(input_buffer));
                struct spa_audio_info_raw input_info = {};
                input_info.format = SPA_AUDIO_FORMAT_F32;
                input_info.rate = g_input_sample_rate;
                input_info.channels = CHANNELS;
                input_info.position[0] = SPA_AUDIO_CHANNEL_FL;
                input_info.position[1] = SPA_AUDIO_CHANNEL_FR;

                const struct spa_pod* input_params[1];
                input_params[0] =
                    spa_format_audio_raw_build(&input_builder, SPA_PARAM_EnumFormat, &input_info);

                int connectResult =
                    pw_stream_connect(data.input_stream, PW_DIRECTION_INPUT, PW_ID_ANY,
                                      static_cast<pw_stream_flags>(PW_STREAM_FLAG_MAP_BUFFERS |
                                                                   PW_STREAM_FLAG_RT_PROCESS),
                                      input_params, 1);
                if (connectResult < 0) {
                    std::cerr << "PipeWire: Failed to connect input stream (" << connectResult
                              << "). Disabling PipeWire path." << std::endl;
                    pw_stream_destroy(data.input_stream);
                    data.input_stream = nullptr;
                    pipewireActive = false;
                } else {
                    struct timespec interval = {0, 100000000};  // 100ms
                    data.signal_check_timer = pw_loop_add_timer(loop, on_signal_check_timer, &data);
                    if (data.signal_check_timer) {
                        pw_loop_update_timer(loop, data.signal_check_timer, &interval, &interval,
                                             false);
                        std::cout << "Signal check timer initialized (100ms interval)" << std::endl;
                    } else {
                        std::cerr << "Warning: Failed to create signal check timer" << std::endl;
                    }
                }
            }
        } else if (!pipewireEnabled) {
            std::cout << "PipeWire input path skipped (disabled)." << std::endl;
        }

        if (!pipewireActive && data.loop) {
            pw_main_loop_destroy(data.loop);
            data.loop = nullptr;
            g_pw_loop = nullptr;
            loop = nullptr;
        }

        double outputRateKHz = g_input_sample_rate * g_config.upsampleRatio / 1000.0;
        std::cout << std::endl;
        if (pipewireActive) {
            std::cout << "System ready. Audio routing configured:" << std::endl;
            std::cout << "  1. Applications → gpu_upsampler_sink (select in GNOME settings)"
                      << std::endl;
            std::cout << "  2. gpu_upsampler_sink.monitor → GPU Upsampler ("
                      << g_config.upsampleRatio << "x upsampling)" << std::endl;
            std::cout << "  3. GPU Upsampler → ALSA → SMSL DAC (" << outputRateKHz << "kHz direct)"
                      << std::endl;
            std::cout << std::endl;
            std::cout << "Select 'GPU Upsampler (" << outputRateKHz
                      << "kHz)' as output device in sound settings." << std::endl;
        } else {
            std::cout << "System ready (RTP-only mode). Audio routing configured:" << std::endl;
            std::cout << "  1. RTP network source → GPU Upsampler (" << g_config.upsampleRatio
                      << "x upsampling)" << std::endl;
            std::cout << "  2. GPU Upsampler → ALSA → SMSL DAC (" << outputRateKHz << "kHz direct)"
                      << std::endl;
        }
        std::cout << "Press Ctrl+C to stop." << std::endl;
        std::cout << "========================================" << std::endl;

        shutdownManager.notifyReady(pipewireActive);

        auto runRtpOnlyMainLoop = [&]() {
            while (g_running.load() && !g_reload_requested.load() && !g_zmq_bind_failed.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                shutdownManager.tick(false);
            }
        };

        if (pipewireActive && data.loop && !g_zmq_bind_failed.load()) {
            if (g_running.load() && !g_reload_requested.load()) {
                g_main_loop_running = true;
                pw_main_loop_run(data.loop);
                g_main_loop_running = false;
            } else if (!g_running.load()) {
                std::cout << "Startup interrupted by signal, skipping main loop." << std::endl;
            } else if (g_reload_requested.load()) {
                std::cout << "RELOAD requested during startup, skipping main loop." << std::endl;
            }
        } else if (g_zmq_bind_failed.load()) {
            std::cerr << "Startup aborted due to ZeroMQ bind failure." << std::endl;
        } else {
            runRtpOnlyMainLoop();
        }

        shutdownManager.runShutdownSequence(pipewireActive);

        // Step 5: Signal worker threads to stop and wait for them
        std::cout << "  Step 5: Stopping worker threads..." << std::endl;
        g_running = false;
        g_buffer_cv.notify_all();
        if (g_zmq_server) {
            g_zmq_server->stop();
            g_zmq_server.reset();
        }
        alsa_thread.join();  // ALSA thread will call snd_pcm_drain() before exit

        // Step 6: Release audio processing resources
        std::cout << "  Step 6: Releasing resources..." << std::endl;
        if (g_rtp_coordinator) {
            g_rtp_coordinator->shutdown();
            g_rtp_coordinator.reset();
        }
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
        delete g_upsampler;
        g_upsampler = nullptr;
        g_dac_manager.stop();

        // Step 7: Deinitialize PipeWire
        if (pipewireInitCalled) {
            pw_deinit();
        }

        // Don't reload if ZMQ bind failed - exit completely
        if (g_zmq_bind_failed) {
            std::cerr << "Exiting due to ZeroMQ initialization failure." << std::endl;
            exitCode = 1;
            break;
        }

        if (g_reload_requested) {
            std::cout << "Reload requested. Restarting daemon with updated config..." << std::endl;
        }
    } while (g_reload_requested);

    // Release PID lock and remove file on clean exit
    release_pid_lock();
    std::cout << "Goodbye!" << std::endl;
    return exitCode;
}
