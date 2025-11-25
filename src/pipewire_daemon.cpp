#include "audio_ring_buffer.h"
#include "audio_utils.h"
#include "config_loader.h"
#include "convolution_engine.h"
#include "daemon_constants.h"
#include "soft_mute.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/props.h>
#include <thread>
#include <vector>

// Configuration (using common constants from daemon_constants.h)
using namespace DaemonConstants;

// Dynamic rate configuration (updated at runtime based on input detection)
// Using atomic for thread-safety between PipeWire callbacks and main thread
static std::atomic<int> g_current_input_rate{DEFAULT_INPUT_SAMPLE_RATE};
static std::atomic<int> g_current_output_rate{DEFAULT_OUTPUT_SAMPLE_RATE};
static std::atomic<int> g_current_rate_family_int{
    static_cast<int>(ConvolutionEngine::RateFamily::RATE_44K)};

// Helper to get/set rate family atomically
inline ConvolutionEngine::RateFamily g_get_rate_family() {
    return static_cast<ConvolutionEngine::RateFamily>(
        g_current_rate_family_int.load(std::memory_order_acquire));
}
inline void g_set_rate_family(ConvolutionEngine::RateFamily family) {
    g_current_rate_family_int.store(static_cast<int>(family), std::memory_order_release);
}

// Global state
static std::atomic<bool> g_running{true};
static ConvolutionEngine::GPUUpsampler* g_upsampler = nullptr;
static SoftMute::Controller* g_soft_mute = nullptr;

// Pending rate change (set by PipeWire callback, processed in main loop)
// Value: 0 = no change pending, >0 = detected input sample rate
static std::atomic<int> g_pending_rate_change{0};

// Forward declaration for Data struct (defined below)
struct Data;

// Global pointer to Data for use in handle_rate_change
static Data* g_data = nullptr;

// Output ring buffers (using common AudioRingBuffer class)
static AudioRingBuffer g_output_buffer_left;
static AudioRingBuffer g_output_buffer_right;
static std::vector<float> g_output_temp_left;
static std::vector<float> g_output_temp_right;
// Use 768kHz (48k family max) as base to ensure sufficient capacity for both rate families
static constexpr size_t OUTPUT_RING_CAPACITY = 768000 * 2;  // ~2 seconds per channel at max rate

// PipeWire objects
struct Data {
    struct pw_main_loop* loop;
    struct pw_stream* input_stream;
    struct pw_stream* output_stream;
    bool gpu_ready;

    // Scratch buffers to avoid allocations in real-time callbacks
    std::vector<float> input_left;
    std::vector<float> input_right;
    std::vector<float> output_left;
    std::vector<float> output_right;

    // Streaming input accumulation buffers (per channel)
    std::vector<float> stream_input_left;
    std::vector<float> stream_input_right;
    size_t stream_accum_left = 0;
    size_t stream_accum_right = 0;

    // Flag to indicate output stream needs reconnection after rate family change
    bool needs_output_reconnect = false;
    int new_output_rate = 0;
};

// Signal handler for graceful shutdown
static void signal_handler(int sig) {
    std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    g_running = false;
}

// Rate family switching helper
// Called when input sample rate changes (e.g., from PipeWire param event or ZeroMQ)
// Connected via on_param_changed() event and processed in main loop.
// See: https://github.com/michihitoTakami/michy_os/issues/218
// Implements soft mute during filter switching (Issue #266)
static bool handle_rate_change(int detected_sample_rate) {
    if (!g_upsampler) {
        return false;
    }

    // Multi-rate mode: use switchToInputRate() for dynamic rate switching
    if (!g_upsampler->isMultiRateEnabled()) {
        std::cerr
            << "[Rate] ERROR: Multi-rate mode not enabled. Rate switching requires multi-rate mode."
            << std::endl;
        return false;
    }

    // Apply soft mute during filter switching (Issue #266)
    // Fade-out: 1.5 seconds, switch filter, fade-in: 1.5 seconds
    // Thread safety: Configuration changes are atomic inside SoftMute::Controller, and we still
    // wait for fade-out completion before mutating filter state to avoid artifacts.
    using namespace DaemonConstants;
    int current_output_rate = g_current_output_rate.load();
    int originalFadeDuration = DEFAULT_SOFT_MUTE_FADE_MS;

    if (g_soft_mute && current_output_rate > 0) {
        // Save original fade duration for restoration
        originalFadeDuration = g_soft_mute->getFadeDuration();

        // Update fade duration for filter switching
        g_soft_mute->setFadeDuration(FILTER_SWITCH_FADE_MS);
        g_soft_mute->setSampleRate(current_output_rate);

        std::cout << "[Rate] Starting fade-out for filter switch ("
                  << (FILTER_SWITCH_FADE_MS / 1000.0) << "s)..." << std::endl;
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
                std::cerr << "[Rate] Warning: Fade-out timeout (" << FILTER_SWITCH_FADE_TIMEOUT_MS
                          << "ms), proceeding with switch" << std::endl;
                break;
            }
        }

        // Ensure we're fully muted before switching
        if (g_soft_mute->getState() != SoftMute::MuteState::MUTED) {
            g_soft_mute->setMuted();
        }
    }

    // Perform filter switch
    bool switch_success = false;

    if (g_upsampler->switchToInputRate(detected_sample_rate)) {
        g_current_input_rate.store(detected_sample_rate, std::memory_order_release);

        // Output rate is dynamically calculated (input_rate × upsample_ratio)
        int new_output_rate = g_upsampler->getOutputSampleRate();
        g_current_output_rate.store(new_output_rate, std::memory_order_release);

        // Update soft mute sample rate if output rate changed
        if (g_soft_mute && new_output_rate != current_output_rate) {
            g_soft_mute->setSampleRate(new_output_rate);
        }

        // Output rate change only requires reconnection if it actually changed
        // (same-family hi-res switches keep the same output rate)
        if (g_data && new_output_rate != current_output_rate) {
            g_data->needs_output_reconnect = true;
            g_data->new_output_rate = new_output_rate;
            std::cout << "[Rate] Output stream reconnection scheduled for " << new_output_rate
                      << " Hz (" << g_upsampler->getUpsampleRatio() << "x upsampling)" << std::endl;
        } else {
            std::cout << "[Rate] Rate switched to " << detected_sample_rate << " Hz -> "
                      << new_output_rate << " Hz (" << g_upsampler->getUpsampleRatio()
                      << "x upsampling)" << std::endl;
        }
        switch_success = true;
    } else {
        std::cerr << "[Rate] Failed to switch to input rate: " << detected_sample_rate << " Hz"
                  << std::endl;
    }

    // Start fade-in after filter switch (or restore state on failure)
    if (g_soft_mute) {
        if (switch_success) {
            std::cout << "[Rate] Starting fade-in after filter switch ("
                      << (FILTER_SWITCH_FADE_MS / 1000.0) << "s)..." << std::endl;
            g_soft_mute->startFadeIn();
            // Fade duration will be reset to default in output processing thread when fade-in
            // completes
        } else {
            // If switch failed, restore original state immediately
            std::cerr << "[Rate] Switch failed, restoring audio state" << std::endl;
            g_soft_mute->setPlaying();
            g_soft_mute->setFadeDuration(originalFadeDuration);
        }
    }

    return switch_success;
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
        // Deinterleave input (stereo interleaved → separate L/R) using scratch buffers
        if (data->input_left.size() < n_frames)
            data->input_left.resize(n_frames);
        if (data->input_right.size() < n_frames)
            data->input_right.resize(n_frames);

        AudioUtils::deinterleaveStereo(input_samples, data->input_left.data(),
                                       data->input_right.data(), n_frames);

        // Process through GPU upsampler using streaming API (pre-allocated buffers)
        // PipeWire may deliver larger chunks than the streaming block size; add once, then
        // keep draining as long as accumulated input can produce full blocks.
        bool first_iteration = true;
        while (true) {
            size_t frames_to_process = first_iteration ? n_frames : 0;
            bool left_generated = g_upsampler->processStreamBlock(
                data->input_left.data(), frames_to_process, data->output_left,
                g_upsampler->streamLeft_, data->stream_input_left, data->stream_accum_left);
            bool right_generated = g_upsampler->processStreamBlock(
                data->input_right.data(), frames_to_process, data->output_right,
                g_upsampler->streamRight_, data->stream_input_right, data->stream_accum_right);

            if (left_generated && right_generated) {
                // Store output for consumption by output stream
                if (!g_output_buffer_left.write(data->output_left.data(),
                                                data->output_left.size()) ||
                    !g_output_buffer_right.write(data->output_right.data(),
                                                 data->output_right.size())) {
                    std::cerr << "Warning: Output ring buffer overflow - dropping samples"
                              << std::endl;
                }
                // Continue loop: there might be enough accumulated input for another block.
                first_iteration = false;
                continue;
            }

            // No output generated (either insufficient input or error) - exit loop.
            break;
        }
    }

    pw_stream_queue_buffer(data->input_stream, buf);
}

// Output stream process callback (705.6kHz audio to DAC)
static void on_output_process(void* userdata) {
    Data* data = static_cast<Data*>(userdata);

    struct pw_buffer* buf = pw_stream_dequeue_buffer(data->output_stream);
    if (!buf) {
        return;
    }

    struct spa_buffer* spa_buf = buf->buffer;
    float* output_samples = static_cast<float*>(spa_buf->datas[0].data);
    uint32_t n_frames = spa_buf->datas[0].chunk->size / (sizeof(float) * CHANNELS);

    if (output_samples && n_frames > 0) {
        size_t available = std::min(g_output_buffer_left.availableToRead(),
                                    g_output_buffer_right.availableToRead());

        if (available >= n_frames) {
            if (g_output_temp_left.size() < n_frames)
                g_output_temp_left.resize(n_frames);
            if (g_output_temp_right.size() < n_frames)
                g_output_temp_right.resize(n_frames);

            bool read_left = g_output_buffer_left.read(g_output_temp_left.data(), n_frames);
            bool read_right = g_output_buffer_right.read(g_output_temp_right.data(), n_frames);

            if (read_left && read_right) {
                // Interleave output (separate L/R → stereo interleaved)
                AudioUtils::interleaveStereo(g_output_temp_left.data(), g_output_temp_right.data(),
                                             output_samples, n_frames);

                // Apply soft mute if transitioning (for filter switching)
                if (g_soft_mute) {
                    g_soft_mute->process(output_samples, n_frames);

                    // Reset fade duration to default after filter switch fade-in completes
                    // Check if we just completed a fade-in from filter switching
                    using namespace DaemonConstants;
                    if (g_soft_mute->getState() == SoftMute::MuteState::PLAYING &&
                        g_soft_mute->getFadeDuration() > DEFAULT_SOFT_MUTE_FADE_MS) {
                        g_soft_mute->setFadeDuration(DEFAULT_SOFT_MUTE_FADE_MS);
                    }
                }
            } else {
                std::memset(output_samples, 0, n_frames * CHANNELS * sizeof(float));
            }
        } else {
            // Underrun - output silence
            std::memset(output_samples, 0, n_frames * CHANNELS * sizeof(float));
        }

        spa_buf->datas[0].chunk->size = n_frames * CHANNELS * sizeof(float);
    }

    pw_stream_queue_buffer(data->output_stream, buf);
}

// Stream state changed callback
static void on_stream_state_changed(void* userdata, enum pw_stream_state old_state,
                                    enum pw_stream_state state, const char* error) {
    (void)userdata;
    (void)old_state;

    std::cout << "Stream state: " << pw_stream_state_as_string(state);
    if (error) {
        std::cout << " (error: " << error << ")";
    }
    std::cout << std::endl;
}

// Input stream param changed callback (detects sample rate changes)
// This is called in the PipeWire real-time thread, so we only set a flag
// and let the main loop handle the actual rate change.
// See: docs/architecture/rate-negotiation-handshake.md
static void on_param_changed(void* userdata, uint32_t id, const struct spa_pod* param) {
    (void)userdata;

    // Only handle format changes
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

    // Check if rate has changed
    if (detected_rate != current_rate && detected_rate > 0) {
        std::cout << "[PipeWire] Sample rate change detected: " << current_rate << " -> "
                  << detected_rate << " Hz" << std::endl;

        // Set pending rate change (will be processed in main loop)
        // This avoids blocking the real-time audio thread
        g_pending_rate_change.store(detected_rate, std::memory_order_release);
    }
}

static const struct pw_stream_events input_stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
    .param_changed = on_param_changed,
    .process = on_input_process,
};

static const struct pw_stream_events output_stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
    .process = on_output_process,
};

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  GPU Audio Upsampler - PipeWire Daemon" << std::endl;
    std::cout << "  Multi-Rate: 44.1k/48k → 705.6k/768k" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Parse coefficient directory (multi-rate mode only)
    // Usage: pipewire_daemon [coefficient_dir]
    std::string coefficient_dir = "data/coefficients";

    if (argc >= 2) {
        coefficient_dir = argv[1];
    }

    // Install signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Initialize GPU upsampler (multi-rate mode only)
    std::cout << "Initializing GPU upsampler..." << std::endl;
    g_upsampler = new ConvolutionEngine::GPUUpsampler();

    std::cout << "Mode: Multi-Rate (all supported rates: 44.1k/48k families, 16x/8x/4x/2x)"
              << std::endl;
    bool init_success = g_upsampler->initializeMultiRate(
        coefficient_dir, DEFAULT_BLOCK_SIZE,
        44100);  // Initial input rate (will be updated by PipeWire detection)

    if (!init_success) {
        std::cerr << "Failed to initialize GPU upsampler" << std::endl;
        delete g_upsampler;
        return 1;
    }
    std::cout << "GPU upsampler ready (" << g_upsampler->getUpsampleRatio() << "x upsampling, "
              << DEFAULT_BLOCK_SIZE << " samples/block)" << std::endl;

    // Load config and set phase type (input sample rate is auto-detected from PipeWire)
    AppConfig appConfig;
    if (loadAppConfig(DEFAULT_CONFIG_FILE, appConfig, false)) {
        g_upsampler->setPhaseType(appConfig.phaseType);
        std::cout << "Phase type: " << phaseTypeToString(appConfig.phaseType) << std::endl;

        // Log latency warning for linear phase
        if (appConfig.phaseType == PhaseType::Linear) {
            double latencySec = g_upsampler->getLatencySeconds();
            std::cout << "  WARNING: Linear phase latency: " << latencySec << " seconds ("
                      << g_upsampler->getLatencySamples() << " samples)" << std::endl;
        }
    } else {
        std::cout << "Phase type: minimum (default)" << std::endl;
    }
    // Input sample rate will be auto-detected from PipeWire stream
    std::cout << "Input sample rate: auto-detected from PipeWire" << std::endl;

    if (!g_upsampler->initializeStreaming()) {
        std::cerr << "Failed to initialize streaming mode" << std::endl;
        delete g_upsampler;
        return 1;
    }

    // Initialize soft mute controller (default fade duration, will be extended for filter
    // switching)
    using namespace DaemonConstants;
    int initial_output_rate = g_upsampler->getOutputSampleRate();
    g_soft_mute = new SoftMute::Controller(DEFAULT_SOFT_MUTE_FADE_MS, initial_output_rate);
    std::cout << "Soft mute initialized (" << DEFAULT_SOFT_MUTE_FADE_MS << "ms fade at "
              << initial_output_rate << "Hz)" << std::endl;

    std::cout << std::endl;

    // Initialize PipeWire
    pw_init(&argc, &argv);

    g_output_buffer_left.init(OUTPUT_RING_CAPACITY);
    g_output_buffer_right.init(OUTPUT_RING_CAPACITY);

    Data data = {};
    data.gpu_ready = true;
    g_data = &data;  // Set global pointer for handle_rate_change

    // Pre-allocate streaming input buffers (based on streamValidInputPerBlock_)
    // Use 2x safety margin to handle timing variations
    size_t buffer_capacity = g_upsampler->getStreamValidInputPerBlock() * 2;
    data.stream_input_left.resize(buffer_capacity, 0.0f);
    data.stream_input_right.resize(buffer_capacity, 0.0f);
    std::cout << "Streaming buffer capacity: " << buffer_capacity
              << " samples (2x streamValidInputPerBlock)" << std::endl;
    data.stream_accum_left = 0;
    data.stream_accum_right = 0;

    // Create main loop
    data.loop = pw_main_loop_new(nullptr);
    struct pw_loop* loop = pw_main_loop_get_loop(data.loop);

    // Create input stream to capture from gpu_upsampler_sink.monitor
    std::cout << "Creating input stream (capturing from gpu_upsampler_sink)..." << std::endl;
    data.input_stream = pw_stream_new_simple(
        loop, "GPU Upsampler Input",
        pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio", PW_KEY_MEDIA_CATEGORY, "Capture",
                          PW_KEY_MEDIA_ROLE, "Music", PW_KEY_NODE_DESCRIPTION,
                          "GPU Upsampler Input", PW_KEY_NODE_TARGET, "gpu_upsampler_sink.monitor",
                          "audio.channels", "2", "audio.position", "FL,FR", nullptr),
        &input_stream_events, &data);

    // Configure input stream audio format (32-bit float stereo @ 44.1kHz)
    uint8_t input_buffer[1024];
    struct spa_pod_builder input_builder = SPA_POD_BUILDER_INIT(input_buffer, sizeof(input_buffer));

    struct spa_audio_info_raw input_info = {};
    input_info.format = SPA_AUDIO_FORMAT_F32;
    input_info.rate = g_current_input_rate.load();
    input_info.channels = CHANNELS;
    input_info.position[0] = SPA_AUDIO_CHANNEL_FL;
    input_info.position[1] = SPA_AUDIO_CHANNEL_FR;

    const struct spa_pod* input_params[1];
    input_params[0] = spa_format_audio_raw_build(&input_builder, SPA_PARAM_EnumFormat, &input_info);

    pw_stream_connect(
        data.input_stream, PW_DIRECTION_INPUT, PW_ID_ANY,
        static_cast<pw_stream_flags>(PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS |
                                     PW_STREAM_FLAG_AUTOCONNECT),
        input_params, 1);

    // Create output stream (705.6kHz playback to DAC)
    std::cout << "Creating output stream (705.6kHz)..." << std::endl;
    data.output_stream = pw_stream_new_simple(
        loop, "GPU Upsampler Output",
        pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio", PW_KEY_MEDIA_CATEGORY, "Playback",
                          PW_KEY_MEDIA_ROLE, "Music", PW_KEY_NODE_TARGET,
                          "alsa_output.usb-SMSL_SMSL_USB_AUDIO-00.iec958-stereo", "audio.channels",
                          "2", "audio.position", "FL,FR", nullptr),
        &output_stream_events, &data);

    // Configure output stream audio format (32-bit float stereo @ 705.6kHz)
    uint8_t output_buffer[1024];
    struct spa_pod_builder output_builder =
        SPA_POD_BUILDER_INIT(output_buffer, sizeof(output_buffer));

    struct spa_audio_info_raw output_info = {};
    output_info.format = SPA_AUDIO_FORMAT_F32;
    output_info.rate = g_current_output_rate.load();
    output_info.channels = CHANNELS;
    output_info.position[0] = SPA_AUDIO_CHANNEL_FL;
    output_info.position[1] = SPA_AUDIO_CHANNEL_FR;

    const struct spa_pod* output_params[1];
    output_params[0] =
        spa_format_audio_raw_build(&output_builder, SPA_PARAM_EnumFormat, &output_info);

    pw_stream_connect(
        data.output_stream, PW_DIRECTION_OUTPUT, PW_ID_ANY,
        static_cast<pw_stream_flags>(PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS |
                                     PW_STREAM_FLAG_AUTOCONNECT),
        output_params, 1);

    std::cout << std::endl;
    std::cout << "System ready. Audio routing configured:" << std::endl;
    std::cout << "  1. Applications → gpu_upsampler_sink (select in GNOME settings)" << std::endl;
    std::cout << "  2. gpu_upsampler_sink.monitor → GPU Upsampler (16x upsampling)" << std::endl;
    std::cout << "  3. GPU Upsampler → SMSL DAC (705.6kHz output)" << std::endl;
    std::cout << std::endl;
    std::cout << "Select 'GPU Upsampler (705.6kHz)' as output device in sound settings."
              << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;
    std::cout << "========================================" << std::endl;

    // Run main loop with rate change handling
    // Using iterate instead of run to check for pending rate changes
    while (g_running) {
        // Check for pending rate change (set by on_param_changed callback)
        int pending_rate = g_pending_rate_change.exchange(0, std::memory_order_acq_rel);
        if (pending_rate > 0) {
            std::cout << "[Main] Processing rate change to " << pending_rate << " Hz..."
                      << std::endl;
            if (handle_rate_change(pending_rate)) {
                std::cout << "[Main] Rate change successful: " << g_current_input_rate.load()
                          << " Hz input -> " << g_current_output_rate.load() << " Hz output"
                          << std::endl;
            } else {
                std::cerr << "[Main] Rate change failed or not supported" << std::endl;
            }
        }

        // Handle output stream reconnection after rate family change
        if (data.needs_output_reconnect && data.new_output_rate > 0) {
            std::cout << "[Main] Reconnecting output stream at " << data.new_output_rate << " Hz..."
                      << std::endl;

            // Disconnect and destroy old output stream
            if (data.output_stream) {
                pw_stream_disconnect(data.output_stream);
                pw_stream_destroy(data.output_stream);
                data.output_stream = nullptr;
            }

            // Create new output stream with updated rate
            data.output_stream = pw_stream_new_simple(
                loop, "GPU Upsampler Output",
                pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio", PW_KEY_MEDIA_CATEGORY, "Playback",
                                  PW_KEY_MEDIA_ROLE, "Music", PW_KEY_NODE_TARGET,
                                  "alsa_output.usb-SMSL_SMSL_USB_AUDIO-00.iec958-stereo",
                                  "audio.channels", "2", "audio.position", "FL,FR", nullptr),
                &output_stream_events, &data);

            // Configure new output format with updated rate
            uint8_t reconnect_buffer[1024];
            struct spa_pod_builder reconnect_builder =
                SPA_POD_BUILDER_INIT(reconnect_buffer, sizeof(reconnect_buffer));

            struct spa_audio_info_raw reconnect_info = {};
            reconnect_info.format = SPA_AUDIO_FORMAT_F32;
            reconnect_info.rate = static_cast<uint32_t>(data.new_output_rate);
            reconnect_info.channels = CHANNELS;
            reconnect_info.position[0] = SPA_AUDIO_CHANNEL_FL;
            reconnect_info.position[1] = SPA_AUDIO_CHANNEL_FR;

            const struct spa_pod* reconnect_params[1];
            reconnect_params[0] = spa_format_audio_raw_build(&reconnect_builder,
                                                             SPA_PARAM_EnumFormat, &reconnect_info);

            pw_stream_connect(data.output_stream, PW_DIRECTION_OUTPUT, PW_ID_ANY,
                              static_cast<pw_stream_flags>(PW_STREAM_FLAG_MAP_BUFFERS |
                                                           PW_STREAM_FLAG_RT_PROCESS |
                                                           PW_STREAM_FLAG_AUTOCONNECT),
                              reconnect_params, 1);

            std::cout << "[Main] Output stream reconnected at " << data.new_output_rate << " Hz"
                      << std::endl;

            // Clear reconnection flag
            data.needs_output_reconnect = false;
            data.new_output_rate = 0;
        }

        // Process PipeWire events (non-blocking iteration)
        // timeout_ms = -1 means wait indefinitely, but we use a short timeout
        // to periodically check g_running and pending rate changes
        pw_loop_iterate(pw_main_loop_get_loop(data.loop), 10);  // 10ms timeout
    }

    // Cleanup
    std::cout << "Shutting down..." << std::endl;

    if (data.input_stream) {
        pw_stream_destroy(data.input_stream);
    }
    if (data.output_stream) {
        pw_stream_destroy(data.output_stream);
    }
    if (data.loop) {
        pw_main_loop_destroy(data.loop);
    }

    delete g_upsampler;
    if (g_soft_mute) {
        delete g_soft_mute;
        g_soft_mute = nullptr;
    }
    pw_deinit();

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
