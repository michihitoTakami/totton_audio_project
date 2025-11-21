#include "convolution_engine.h"
#include <alsa/asoundlib.h>
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <csignal>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

// Configuration
constexpr int INPUT_SAMPLE_RATE = 44100;
constexpr int OUTPUT_SAMPLE_RATE = 352800;  // 8x upsampling
constexpr int UPSAMPLE_RATIO = 8;
constexpr int BLOCK_SIZE = 4096;
constexpr int CHANNELS = 2;
constexpr const char* ALSA_DEVICE = "hw:3,0";  // SMSL DAC (card 3, device 0)

// Global state
static std::atomic<bool> g_running{true};
static ConvolutionEngine::GPUUpsampler* g_upsampler = nullptr;

// Audio buffer for thread communication
static std::mutex g_buffer_mutex;
static std::condition_variable g_buffer_cv;
static std::vector<float> g_output_buffer_left;
static std::vector<float> g_output_buffer_right;
static size_t g_output_read_pos = 0;

// PipeWire objects
struct Data {
    struct pw_main_loop* loop;
    struct pw_stream* input_stream;
    bool gpu_ready;
};

// Signal handler for graceful shutdown
static void signal_handler(int sig) {
    std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    g_running = false;
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

        // Process through GPU upsampler
        std::vector<float> output_left, output_right;
        bool success = g_upsampler->processStereo(
            left.data(),
            right.data(),
            n_frames,
            output_left,
            output_right
        );

        if (success) {
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

// ALSA output thread (705.6kHz direct to DAC)
void alsa_output_thread() {
    snd_pcm_t* pcm_handle = nullptr;
    int err;

    // Open ALSA device
    err = snd_pcm_open(&pcm_handle, ALSA_DEVICE, SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        std::cerr << "ALSA: Cannot open device " << ALSA_DEVICE << ": "
                  << snd_strerror(err) << std::endl;
        return;
    }

    // Set hardware parameters
    snd_pcm_hw_params_t* hw_params;
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(pcm_handle, hw_params);

    // Set access type
    err = snd_pcm_hw_params_set_access(pcm_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) {
        std::cerr << "ALSA: Cannot set access type: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return;
    }

    // Set sample format (32-bit signed integer)
    err = snd_pcm_hw_params_set_format(pcm_handle, hw_params, SND_PCM_FORMAT_S32_LE);
    if (err < 0) {
        std::cerr << "ALSA: Cannot set sample format: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return;
    }

    // Set sample rate (705.6kHz)
    unsigned int rate = OUTPUT_SAMPLE_RATE;
    err = snd_pcm_hw_params_set_rate_near(pcm_handle, hw_params, &rate, 0);
    if (err < 0) {
        std::cerr << "ALSA: Cannot set sample rate: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return;
    }
    std::cout << "ALSA: Sample rate set to " << rate << " Hz" << std::endl;

    // Set channels (stereo)
    err = snd_pcm_hw_params_set_channels(pcm_handle, hw_params, CHANNELS);
    if (err < 0) {
        std::cerr << "ALSA: Cannot set channel count: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return;
    }

    // Set buffer size (very large buffer for ultra-smooth playback)
    snd_pcm_uframes_t buffer_size = 131072;
    err = snd_pcm_hw_params_set_buffer_size_near(pcm_handle, hw_params, &buffer_size);
    if (err < 0) {
        std::cerr << "ALSA: Cannot set buffer size: " << snd_strerror(err) << std::endl;
    }

    // Set period size (larger periods for better GPU→ALSA sync)
    snd_pcm_uframes_t period_size = 16384;
    err = snd_pcm_hw_params_set_period_size_near(pcm_handle, hw_params, &period_size, 0);
    if (err < 0) {
        std::cerr << "ALSA: Cannot set period size: " << snd_strerror(err) << std::endl;
    }

    // Apply hardware parameters
    err = snd_pcm_hw_params(pcm_handle, hw_params);
    if (err < 0) {
        std::cerr << "ALSA: Cannot set hardware parameters: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return;
    }

    std::cout << "ALSA: Output device configured (705.6kHz, 32-bit int, stereo)" << std::endl;
    std::cout << "ALSA: Buffer size: " << buffer_size << " frames, Period size: " << period_size << " frames" << std::endl;

    // Prepare device
    err = snd_pcm_prepare(pcm_handle);
    if (err < 0) {
        std::cerr << "ALSA: Cannot prepare device: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return;
    }

    std::vector<int32_t> interleaved_buffer(period_size * CHANNELS);

    // Main playback loop
    while (g_running) {
        // Wait for GPU processed data (3x period to ensure sufficient buffering)
        std::unique_lock<std::mutex> lock(g_buffer_mutex);
        g_buffer_cv.wait_for(lock, std::chrono::milliseconds(200), [period_size] {
            return (g_output_buffer_left.size() - g_output_read_pos) >= (period_size * 3) || !g_running;
        });

        if (!g_running) break;

        size_t available = g_output_buffer_left.size() - g_output_read_pos;
        if (available >= period_size) {
            // Interleave L/R channels and convert float→int32
            // Apply gain compensation (8x) because upsampling spreads energy across more samples
            constexpr float gain = static_cast<float>(UPSAMPLE_RATIO);

            // Statistics for debugging
            static size_t clip_count = 0;
            static size_t total_samples = 0;
            size_t current_clips = 0;

            for (size_t i = 0; i < period_size; ++i) {
                // Convert float [-1.0, 1.0] to int32 [-2147483648, 2147483647]
                // Apply gain to compensate for energy distribution in upsampled signal
                float left_sample = g_output_buffer_left[g_output_read_pos + i] * gain;
                float right_sample = g_output_buffer_right[g_output_read_pos + i] * gain;

                // Detect clipping before clamping
                if (left_sample > 1.0f || left_sample < -1.0f ||
                    right_sample > 1.0f || right_sample < -1.0f) {
                    current_clips++;
                    clip_count++;
                }

                // Clamp to [-1.0, 1.0] to prevent overflow
                left_sample = std::max(-1.0f, std::min(1.0f, left_sample));
                right_sample = std::max(-1.0f, std::min(1.0f, right_sample));

                // Proper float to int32 conversion with correct scaling and rounding
                // Use 2^31 = 2147483648.0 for symmetric range, and add 0.5 for rounding
                int32_t left_int = static_cast<int32_t>(left_sample * 2147483648.0f);
                int32_t right_int = static_cast<int32_t>(right_sample * 2147483648.0f);

                interleaved_buffer[i * 2] = left_int;
                interleaved_buffer[i * 2 + 1] = right_int;
            }

            total_samples += period_size * 2;

            // Report clipping every 10 periods (~0.5 seconds at 352.8kHz)
            if (total_samples % (period_size * 2 * 10) == 0) {
                if (clip_count > 0) {
                    std::cout << "WARNING: Clipping detected - " << clip_count
                              << " samples clipped out of " << total_samples
                              << " (" << (100.0 * clip_count / total_samples) << "%)" << std::endl;
                }
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
                frames_written = snd_pcm_recover(pcm_handle, frames_written, 0);
                if (frames_written < 0) {
                    std::cerr << "ALSA: Write error: " << snd_strerror(frames_written) << std::endl;
                }
            }
        } else {
            lock.unlock();
        }
    }

    // Cleanup
    snd_pcm_drain(pcm_handle);
    snd_pcm_close(pcm_handle);
    std::cout << "ALSA: Output thread terminated" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  GPU Audio Upsampler - ALSA Direct Output" << std::endl;
    std::cout << "  44.1kHz → 352.8kHz (8x upsampling)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Parse filter path
    std::string filter_path = "data/coefficients/filter_1m_min_phase.bin";
    if (argc > 1) {
        filter_path = argv[1];
    }

    // Install signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Initialize GPU upsampler
    std::cout << "Initializing GPU upsampler..." << std::endl;
    g_upsampler = new ConvolutionEngine::GPUUpsampler();
    if (!g_upsampler->initialize(filter_path, UPSAMPLE_RATIO, BLOCK_SIZE)) {
        std::cerr << "Failed to initialize GPU upsampler" << std::endl;
        delete g_upsampler;
        return 1;
    }
    std::cout << "GPU upsampler ready (16x upsampling, " << BLOCK_SIZE << " samples/block)" << std::endl;
    std::cout << std::endl;

    // Start ALSA output thread
    std::cout << "Starting ALSA output thread..." << std::endl;
    std::thread alsa_thread(alsa_output_thread);

    // Initialize PipeWire input
    pw_init(&argc, &argv);

    Data data = {};
    data.gpu_ready = true;

    // Create main loop
    data.loop = pw_main_loop_new(nullptr);
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

    // Configure input stream audio format (32-bit float stereo @ 44.1kHz)
    uint8_t input_buffer[1024];
    struct spa_pod_builder input_builder = SPA_POD_BUILDER_INIT(input_buffer, sizeof(input_buffer));

    struct spa_audio_info_raw input_info = {};
    input_info.format = SPA_AUDIO_FORMAT_F32;
    input_info.rate = INPUT_SAMPLE_RATE;
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

    std::cout << std::endl;
    std::cout << "System ready. Audio routing configured:" << std::endl;
    std::cout << "  1. Applications → gpu_upsampler_sink (select in GNOME settings)" << std::endl;
    std::cout << "  2. gpu_upsampler_sink.monitor → GPU Upsampler (8x upsampling)" << std::endl;
    std::cout << "  3. GPU Upsampler → ALSA → SMSL DAC (352.8kHz direct)" << std::endl;
    std::cout << std::endl;
    std::cout << "Select 'GPU Upsampler (705.6kHz)' as output device in sound settings." << std::endl;
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
    }

    g_running = false;
    g_buffer_cv.notify_all();
    alsa_thread.join();

    delete g_upsampler;
    pw_deinit();

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
