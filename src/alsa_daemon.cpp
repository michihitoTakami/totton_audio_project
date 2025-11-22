#include "convolution_engine.h"
#include "config_loader.h"
#include "eq_parser.h"
#include "eq_to_fir.h"
#include <alsa/asoundlib.h>
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <csignal>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <unistd.h>
#include <sys/types.h>
#include <sys/file.h>
#include <fcntl.h>
#include <cerrno>
#include <cstdio>

// PID file path (also serves as lock file)
constexpr const char* PID_FILE_PATH = "/tmp/gpu_upsampler_alsa.pid";

// Default configuration values
constexpr int DEFAULT_INPUT_SAMPLE_RATE = 44100;
constexpr int DEFAULT_UPSAMPLE_RATIO = 16;
constexpr int DEFAULT_BLOCK_SIZE = 4096;
constexpr int CHANNELS = 2;
constexpr const char* DEFAULT_ALSA_DEVICE = "hw:USB";
constexpr const char* DEFAULT_FILTER_PATH = "data/coefficients/filter_1m_min_phase.bin";
constexpr const char* CONFIG_FILE_PATH = DEFAULT_CONFIG_FILE;

// Runtime configuration (loaded from config.json)
static AppConfig g_config;

// Global state
static std::atomic<bool> g_running{true};
static std::atomic<bool> g_reload_requested{false};
static ConvolutionEngine::GPUUpsampler* g_upsampler = nullptr;
static struct pw_main_loop* g_pw_loop = nullptr;  // For SIGHUP handler

// Audio buffer for thread communication
static std::mutex g_buffer_mutex;
static std::condition_variable g_buffer_cv;
static std::vector<float> g_output_buffer_left;
static std::vector<float> g_output_buffer_right;
static size_t g_output_read_pos = 0;

// Streaming input accumulation buffers
static std::vector<float> g_stream_input_left;
static std::vector<float> g_stream_input_right;
static size_t g_stream_accumulated_left = 0;
static size_t g_stream_accumulated_right = 0;

// ========== PID File Lock (flock-based) ==========

// File descriptor for the PID lock file (kept open while running)
static int g_pid_lock_fd = -1;

// Read PID from lock file (for display purposes)
static pid_t read_pid_from_lockfile() {
    std::ifstream pidfile(PID_FILE_PATH);
    if (!pidfile.is_open()) return 0;
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
        std::cerr << "Error: Cannot open PID file: " << PID_FILE_PATH
                  << " (" << strerror(errno) << ")" << std::endl;
        return false;
    }

    // Try to acquire exclusive lock (non-blocking)
    if (flock(g_pid_lock_fd, LOCK_EX | LOCK_NB) < 0) {
        if (errno == EWOULDBLOCK) {
            // Another process holds the lock
            pid_t existing_pid = read_pid_from_lockfile();
            std::cerr << "Error: Another instance is already running";
            if (existing_pid > 0) {
                std::cerr << " (PID: " << existing_pid << ")";
            }
            std::cerr << std::endl;
            std::cerr << "       Use './scripts/daemon.sh stop' to stop it." << std::endl;
        } else {
            std::cerr << "Error: Cannot lock PID file: " << strerror(errno) << std::endl;
        }
        close(g_pid_lock_fd);
        g_pid_lock_fd = -1;
        return false;
    }

    // Lock acquired - write our PID to the file
    if (ftruncate(g_pid_lock_fd, 0) < 0) {
        std::cerr << "Warning: Cannot truncate PID file" << std::endl;
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
}

// ========== Configuration ==========

static void print_config_summary(const AppConfig& cfg) {
    int outputRate = cfg.inputSampleRate * cfg.upsampleRatio;
    std::cout << "Config:" << std::endl;
    std::cout << "  ALSA device:    " << cfg.alsaDevice << std::endl;
    std::cout << "  Input rate:     " << cfg.inputSampleRate << " Hz" << std::endl;
    std::cout << "  Output rate:    " << outputRate << " Hz (" << (outputRate / 1000.0) << " kHz)" << std::endl;
    std::cout << "  Buffer size:    " << cfg.bufferSize << std::endl;
    std::cout << "  Period size:    " << cfg.periodSize << std::endl;
    std::cout << "  Upsample ratio: " << cfg.upsampleRatio << std::endl;
    std::cout << "  Block size:     " << cfg.blockSize << std::endl;
    std::cout << "  Gain:           " << cfg.gain << std::endl;
    std::cout << "  Filter path:    " << cfg.filterPath << std::endl;
    std::cout << "  EQ enabled:     " << (cfg.eqEnabled ? "yes" : "no") << std::endl;
    if (cfg.eqEnabled && !cfg.eqProfilePath.empty()) {
        std::cout << "  EQ profile:     " << cfg.eqProfilePath << std::endl;
    }
}

static void reset_runtime_state() {
    g_output_buffer_left.clear();
    g_output_buffer_right.clear();
    g_output_read_pos = 0;
    g_stream_input_left.clear();
    g_stream_input_right.clear();
    g_stream_accumulated_left = 0;
    g_stream_accumulated_right = 0;
}

static void load_runtime_config() {
    AppConfig loaded;
    bool found = loadAppConfig(CONFIG_FILE_PATH, loaded);
    g_config = loaded;

    if (g_config.alsaDevice.empty()) g_config.alsaDevice = DEFAULT_ALSA_DEVICE;
    if (g_config.filterPath.empty()) g_config.filterPath = DEFAULT_FILTER_PATH;
    if (g_config.upsampleRatio <= 0) g_config.upsampleRatio = DEFAULT_UPSAMPLE_RATIO;
    if (g_config.blockSize <= 0) g_config.blockSize = DEFAULT_BLOCK_SIZE;
    if (g_config.bufferSize <= 0) g_config.bufferSize = 262144;
    if (g_config.periodSize <= 0) g_config.periodSize = 32768;
    if (g_config.inputSampleRate != 44100 && g_config.inputSampleRate != 48000) {
        g_config.inputSampleRate = DEFAULT_INPUT_SAMPLE_RATE;
    }

    if (!found) {
        std::cout << "Config: Using defaults (no config.json found)" << std::endl;
    }
    print_config_summary(g_config);
}

// PipeWire objects
struct Data {
    struct pw_main_loop* loop;
    struct pw_stream* input_stream;
    bool gpu_ready;
};

// Signal handler for graceful shutdown
static void signal_handler(int sig) {
    if (sig == SIGHUP) {
        std::cout << "\nReceived SIGHUP, restarting for config reload..." << std::endl;
        g_reload_requested = true;
    } else {
        std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    }
    g_running = false;
    // Quit PipeWire main loop to trigger clean shutdown
    if (g_pw_loop) {
        pw_main_loop_quit(g_pw_loop);
    }
}

// Input stream process callback (44.1kHz audio from PipeWire)
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
        // Deinterleave input (stereo interleaved → separate L/R)
        std::vector<float> left(n_frames);
        std::vector<float> right(n_frames);

        for (uint32_t i = 0; i < n_frames; ++i) {
            left[i] = input_samples[i * 2];
            right[i] = input_samples[i * 2 + 1];
        }

        // Process through GPU upsampler using streaming API
        // This preserves overlap buffer across calls
        std::vector<float> output_left, output_right;

        // Process left channel
        bool left_generated = g_upsampler->processStreamBlock(
            left.data(),
            n_frames,
            output_left,
            g_upsampler->streamLeft_,
            g_stream_input_left,
            g_stream_accumulated_left
        );

        // Process right channel
        bool right_generated = g_upsampler->processStreamBlock(
            right.data(),
            n_frames,
            output_right,
            g_upsampler->streamRight_,
            g_stream_input_right,
            g_stream_accumulated_right
        );

        // Only store output if both channels generated output
        // (they should synchronize due to same input size)
        if (left_generated && right_generated) {
            // Store output for ALSA thread consumption
            std::lock_guard<std::mutex> lock(g_buffer_mutex);
            g_output_buffer_left.insert(g_output_buffer_left.end(),
                                       output_left.begin(), output_left.end());
            g_output_buffer_right.insert(g_output_buffer_right.end(),
                                        output_right.begin(), output_right.end());
            g_buffer_cv.notify_one();
        }
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

static const struct pw_stream_events input_stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
    .process = on_input_process,
};

// Open and configure ALSA device. Returns nullptr on failure.
static snd_pcm_t* open_and_configure_pcm() {
    snd_pcm_t* pcm_handle = nullptr;
    int err;

    err = snd_pcm_open(&pcm_handle, g_config.alsaDevice.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        std::cerr << "ALSA: Cannot open device " << g_config.alsaDevice << ": "
                  << snd_strerror(err) << std::endl;
        return nullptr;
    }

    // Set hardware parameters
    snd_pcm_hw_params_t* hw_params;
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(pcm_handle, hw_params);

    if ((err = snd_pcm_hw_params_set_access(pcm_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0 ||
        (err = snd_pcm_hw_params_set_format(pcm_handle, hw_params, SND_PCM_FORMAT_S32_LE)) < 0) {
        std::cerr << "ALSA: Cannot set access/format: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    // Calculate output sample rate from config
    unsigned int rate = g_config.inputSampleRate * g_config.upsampleRatio;
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

    std::cout << "ALSA: Output device configured (" << rate << " Hz, 32-bit int, stereo)"
              << " buffer " << buffer_size << " frames, period " << period_size << " frames" << std::endl;
    return pcm_handle;
}

// Check current PCM state; return false if disconnected/suspended
static bool pcm_alive(snd_pcm_t* pcm_handle) {
    if (!pcm_handle) return false;
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
    snd_pcm_t* pcm_handle = open_and_configure_pcm();
    std::vector<int32_t> interleaved_buffer(32768 * CHANNELS);  // resized after open
    snd_pcm_uframes_t period_size = 32768;
    if (!pcm_handle) {
        // Retry loop until device appears or shutdown
        while (g_running && !pcm_handle) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            pcm_handle = open_and_configure_pcm();
        }
    }
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
                while (g_running && !pcm_handle) {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    pcm_handle = open_and_configure_pcm();
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

        // Wait for GPU processed data (3x period to ensure sufficient buffering)
        std::unique_lock<std::mutex> lock(g_buffer_mutex);
        g_buffer_cv.wait_for(lock, std::chrono::milliseconds(200), [period_size] {
            return (g_output_buffer_left.size() - g_output_read_pos) >= (period_size * 3) || !g_running;
        });

        if (!g_running) break;

        size_t available = g_output_buffer_left.size() - g_output_read_pos;
        if (available >= period_size) {
            // Interleave L/R channels and convert float→int32
            // Apply configured gain (compensates for energy distribution in upsampled signal)
            const float gain = g_config.gain;

            // Statistics for debugging
            static size_t clip_count = 0;
            static size_t total_samples = 0;
            size_t current_clips = 0;

            for (size_t i = 0; i < period_size; ++i) {
                // Convert float [-1.0, 1.0] to int32 [-2147483648, 2147483647]
                // Apply gain to compensate for energy distribution in upsampled signal
                float left_sample = g_output_buffer_left[g_output_read_pos + i] * gain;
                float right_sample = g_output_buffer_right[g_output_read_pos + i] * gain;

                // Detect clipping before soft-clipping
                if (left_sample > 1.0f || left_sample < -1.0f ||
                    right_sample > 1.0f || right_sample < -1.0f) {
                    current_clips++;
                    clip_count++;
                }

                // Soft clipping using tanh (reduces harmonic distortion compared to hard clipping)
                // tanh(x) smoothly compresses values above ±1.0 instead of abruptly cutting them
                // This is gentler on the audio and produces less audible artifacts
                left_sample = std::tanh(left_sample);
                right_sample = std::tanh(right_sample);

                // CRITICAL FIX: Float to int32 conversion
                // Use 2^31-1 = 2147483647 (INT32_MAX) to avoid overflow
                // 2^31 = 2147483648 causes undefined behavior for sample = 1.0
                // Use lroundf() for proper rounding instead of truncation
                constexpr float INT32_MAX_FLOAT = 2147483647.0f;
                int32_t left_int = static_cast<int32_t>(std::lroundf(left_sample * INT32_MAX_FLOAT));
                int32_t right_int = static_cast<int32_t>(std::lroundf(right_sample * INT32_MAX_FLOAT));

                interleaved_buffer[i * 2] = left_int;
                interleaved_buffer[i * 2 + 1] = right_int;
            }

            total_samples += period_size * 2;

            // Report clipping infrequently to avoid log spam
            if (total_samples % (period_size * 2 * 100) == 0 && clip_count > 0) {
                std::cout << "WARNING: Clipping detected - " << clip_count
                          << " samples clipped out of " << total_samples
                          << " (" << (100.0 * clip_count / total_samples) << "%)" << std::endl;
            }
            g_output_read_pos += period_size;

            // Clear consumed data more frequently to avoid large vector reallocations
            // Reallocations can cause audio glitches due to memory operations
            if (g_output_read_pos > period_size * 4) {  // Clean up after 4 periods
                g_output_buffer_left.erase(g_output_buffer_left.begin(),
                                          g_output_buffer_left.begin() + g_output_read_pos);
                g_output_buffer_right.erase(g_output_buffer_right.begin(),
                                           g_output_buffer_right.begin() + g_output_read_pos);
                g_output_read_pos = 0;
            }

            lock.unlock();

            // Write to ALSA device
            snd_pcm_sframes_t frames_written = snd_pcm_writei(pcm_handle, interleaved_buffer.data(), period_size);
            if (frames_written < 0) {
                // Try standard recovery first
                snd_pcm_sframes_t rec = snd_pcm_recover(pcm_handle, frames_written, 0);
                if (rec < 0) {
                    // Device may be gone; attempt reopen
                    std::cerr << "ALSA: Write error: " << snd_strerror(frames_written)
                              << " (recover=" << snd_strerror(rec) << "), retrying reopen..." << std::endl;
                    snd_pcm_close(pcm_handle);
                    pcm_handle = nullptr;
                    while (g_running && !pcm_handle) {
                        std::this_thread::sleep_for(std::chrono::seconds(5));
                        pcm_handle = open_and_configure_pcm();
                    }
                    if (pcm_handle) {
                        // resize buffer to new period size if needed
                        snd_pcm_uframes_t new_period = 0;
                        snd_pcm_hw_params_t* hw_params;
                        snd_pcm_hw_params_alloca(&hw_params);
                        if (snd_pcm_hw_params_current(pcm_handle, hw_params) == 0 &&
                            snd_pcm_hw_params_get_period_size(hw_params, &new_period, nullptr) == 0 &&
                            new_period != period_size) {
                            period_size = new_period;
                            interleaved_buffer.resize(period_size * CHANNELS);
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
            std::fill(interleaved_buffer.begin(), interleaved_buffer.end(), 0);
            snd_pcm_sframes_t frames_written = snd_pcm_writei(pcm_handle, interleaved_buffer.data(), period_size);
            if (frames_written < 0) {
                snd_pcm_sframes_t rec = snd_pcm_recover(pcm_handle, frames_written, 0);
                if (rec < 0) {
                    std::cerr << "ALSA: Silence write error: " << snd_strerror(frames_written)
                              << " (recover=" << snd_strerror(rec) << "), retrying reopen..." << std::endl;
                    if (pcm_handle) {
                        snd_pcm_close(pcm_handle);
                        pcm_handle = nullptr;
                    }
                    while (g_running && !pcm_handle) {
                        std::this_thread::sleep_for(std::chrono::seconds(5));
                        pcm_handle = open_and_configure_pcm();
                    }
                }
            }
        }
    }

    // Cleanup
    if (pcm_handle) {
        snd_pcm_drain(pcm_handle);
        snd_pcm_close(pcm_handle);
    }
    std::cout << "ALSA: Output thread terminated" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  GPU Audio Upsampler - ALSA Direct Output" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Install signal handlers (SIGHUP for restart, SIGINT/SIGTERM for shutdown)
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGHUP, signal_handler);

    // Acquire PID file lock (prevent multiple instances)
    if (!acquire_pid_lock()) {
        return 1;
    }
    std::cout << "PID: " << getpid() << " (file: " << PID_FILE_PATH << ")" << std::endl;

    int exitCode = 0;

    do {
        g_running = true;
        g_reload_requested = false;
        reset_runtime_state();

        // Load configuration from config.json (if exists)
        load_runtime_config();

        // Environment variable overrides config.json
        if (const char* env_dev = std::getenv("ALSA_DEVICE")) {
            g_config.alsaDevice = env_dev;
            std::cout << "Config: ALSA_DEVICE env override: " << env_dev << std::endl;
        }

        // Command line argument overrides filter path
        if (argc > 1) {
            g_config.filterPath = argv[1];
            std::cout << "Config: CLI filter path override: " << argv[1] << std::endl;
        }

        // Auto-select filter based on sample rate if configured filter doesn't exist
        // but a sample-rate-specific version does
        if (!std::filesystem::exists(g_config.filterPath)) {
            // Try to find sample-rate-specific filter
            std::string basePath = g_config.filterPath;
            size_t dotPos = basePath.rfind('.');
            if (dotPos != std::string::npos) {
                std::string rateSpecificPath = basePath.substr(0, dotPos) + "_" +
                    std::to_string(g_config.inputSampleRate) + basePath.substr(dotPos);
                if (std::filesystem::exists(rateSpecificPath)) {
                    std::cout << "Config: Using sample-rate-specific filter: " << rateSpecificPath << std::endl;
                    g_config.filterPath = rateSpecificPath;
                }
            }
        }

        // Warn if using 44.1kHz filter with 48kHz input
        if (g_config.inputSampleRate == 48000 &&
            g_config.filterPath.find("44100") == std::string::npos &&
            g_config.filterPath.find("48000") == std::string::npos) {
            std::cout << "Warning: Using generic filter with 48kHz input. "
                      << "For optimal quality, generate a 48kHz-optimized filter." << std::endl;
        }

        if (!std::filesystem::exists(g_config.filterPath)) {
            std::cerr << "Config error: Filter file not found: " << g_config.filterPath << std::endl;
            exitCode = 1;
            break;
        }

        // Initialize GPU upsampler with configured values
        std::cout << "Initializing GPU upsampler..." << std::endl;
        g_upsampler = new ConvolutionEngine::GPUUpsampler();
        if (!g_upsampler->initialize(g_config.filterPath, g_config.upsampleRatio, g_config.blockSize)) {
            std::cerr << "Failed to initialize GPU upsampler" << std::endl;
            delete g_upsampler;
            exitCode = 1;
            break;
        }
        std::cout << "GPU upsampler ready (" << g_config.upsampleRatio << "x upsampling, "
                  << g_config.blockSize << " samples/block)" << std::endl;

        // Initialize streaming mode to preserve overlap buffers across PipeWire callbacks
        if (!g_upsampler->initializeStreaming()) {
            std::cerr << "Failed to initialize streaming mode" << std::endl;
            delete g_upsampler;
            exitCode = 1;
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
                double outputSampleRate = static_cast<double>(g_config.inputSampleRate) * g_config.upsampleRatio;
                auto eqMagnitude = EQ::computeEqMagnitudeForFft(filterFftSize, fullFftSize, outputSampleRate, eqProfile);

                if (g_upsampler->applyEqMagnitude(eqMagnitude)) {
                    std::cout << "  EQ: Applied with minimum phase reconstruction" << std::endl;
                } else {
                    std::cerr << "  EQ: Failed to apply frequency response" << std::endl;
                }
            } else {
                std::cerr << "  EQ: Failed to parse profile: " << g_config.eqProfilePath << std::endl;
            }
        }

        // Pre-allocate streaming input buffers (based on streamValidInputPerBlock_)
        size_t buffer_capacity = 10000;  // Conservative estimate for buffer size
        g_stream_input_left.resize(buffer_capacity, 0.0f);
        g_stream_input_right.resize(buffer_capacity, 0.0f);
        g_stream_accumulated_left = 0;
        g_stream_accumulated_right = 0;

        std::cout << std::endl;

        // Start ALSA output thread
        std::cout << "Starting ALSA output thread..." << std::endl;
        std::thread alsa_thread(alsa_output_thread);

        // Initialize PipeWire input
        pw_init(&argc, &argv);

        Data data = {};
        data.gpu_ready = true;

        // Create main loop (store globally for signal handler access)
        data.loop = pw_main_loop_new(nullptr);
        g_pw_loop = data.loop;
        struct pw_loop* loop = pw_main_loop_get_loop(data.loop);

        // Create Capture stream from gpu_upsampler_sink.monitor
        std::cout << "Creating PipeWire input (capturing from gpu_upsampler_sink)..." << std::endl;
        data.input_stream = pw_stream_new_simple(
            loop,
            "GPU Upsampler Input",
            pw_properties_new(
                PW_KEY_MEDIA_TYPE, "Audio",
                PW_KEY_MEDIA_CATEGORY, "Capture",
                PW_KEY_MEDIA_ROLE, "Music",
                PW_KEY_NODE_DESCRIPTION, "GPU Upsampler Input",
                PW_KEY_NODE_TARGET, "gpu_upsampler_sink.monitor",
                "audio.channels", "2",
                "audio.position", "FL,FR",
                nullptr
            ),
            &input_stream_events,
            &data
        );

        // Configure input stream audio format (32-bit float stereo)
        uint8_t input_buffer[1024];
        struct spa_pod_builder input_builder = SPA_POD_BUILDER_INIT(input_buffer, sizeof(input_buffer));

        struct spa_audio_info_raw input_info = {};
        input_info.format = SPA_AUDIO_FORMAT_F32;
        input_info.rate = g_config.inputSampleRate;
        input_info.channels = CHANNELS;
        input_info.position[0] = SPA_AUDIO_CHANNEL_FL;
        input_info.position[1] = SPA_AUDIO_CHANNEL_FR;

        const struct spa_pod* input_params[1];
        input_params[0] = spa_format_audio_raw_build(&input_builder, SPA_PARAM_EnumFormat, &input_info);

        pw_stream_connect(
            data.input_stream,
            PW_DIRECTION_INPUT,
            PW_ID_ANY,
            static_cast<pw_stream_flags>(
                PW_STREAM_FLAG_MAP_BUFFERS |
                PW_STREAM_FLAG_RT_PROCESS
            ),
            input_params, 1
        );

        double outputRateKHz = g_config.inputSampleRate * g_config.upsampleRatio / 1000.0;
        std::cout << std::endl;
        std::cout << "System ready. Audio routing configured:" << std::endl;
        std::cout << "  1. Applications → gpu_upsampler_sink (select in GNOME settings)" << std::endl;
        std::cout << "  2. gpu_upsampler_sink.monitor → GPU Upsampler (" << g_config.upsampleRatio << "x upsampling)" << std::endl;
        std::cout << "  3. GPU Upsampler → ALSA → SMSL DAC (" << outputRateKHz << "kHz direct)" << std::endl;
        std::cout << std::endl;
        std::cout << "Select 'GPU Upsampler (" << outputRateKHz << "kHz)' as output device in sound settings." << std::endl;
        std::cout << "Press Ctrl+C to stop." << std::endl;
        std::cout << "========================================" << std::endl;

        // Run main loop
        pw_main_loop_run(data.loop);

        // Cleanup
        std::cout << "Shutting down..." << std::endl;

        if (data.input_stream) {
            pw_stream_destroy(data.input_stream);
        }
        if (data.loop) {
            pw_main_loop_destroy(data.loop);
            g_pw_loop = nullptr;
        }

        g_running = false;
        g_buffer_cv.notify_all();
        alsa_thread.join();

        delete g_upsampler;
        g_upsampler = nullptr;
        pw_deinit();

        if (g_reload_requested) {
            std::cout << "Reload requested. Restarting daemon with updated config..." << std::endl;
        }
    } while (g_reload_requested);

    // Release PID lock and remove file on clean exit
    release_pid_lock();
    std::cout << "Goodbye!" << std::endl;
    return exitCode;
}
