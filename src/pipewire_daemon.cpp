#include "audio_ring_buffer.h"
#include "audio_utils.h"
#include "config_loader.h"
#include "convolution_engine.h"
#include "daemon_constants.h"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstring>
#include <iostream>
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/props.h>
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
};

// Signal handler for graceful shutdown
static void signal_handler(int sig) {
    std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    g_running = false;
}

// Rate family switching helper
// Called when input sample rate changes (e.g., from PipeWire param event or ZeroMQ)
// TODO: Connect this to PipeWire param_changed event or ZeroMQ command handler
// Currently this is a prepared skeleton for future dynamic rate detection.
// See: https://github.com/michihitoTakami/michy_os/issues/38
[[maybe_unused]] static bool handle_rate_change(int detected_sample_rate) {
    if (!g_upsampler || !g_upsampler->isDualRateEnabled()) {
        return false;  // Dual-rate not enabled
    }

    auto new_family = ConvolutionEngine::detectRateFamily(detected_sample_rate);
    if (new_family == ConvolutionEngine::RateFamily::RATE_UNKNOWN) {
        // Note: Avoid std::cerr in real-time path if called from audio callback
        return false;
    }

    if (new_family == g_get_rate_family()) {
        return true;  // Already at correct family
    }

    // Switch coefficient set (glitch-free via double buffering)
    if (g_upsampler->switchRateFamily(new_family)) {
        g_set_rate_family(new_family);
        g_current_input_rate.store(ConvolutionEngine::getBaseSampleRate(new_family),
                                   std::memory_order_release);
        g_current_output_rate.store(ConvolutionEngine::getOutputSampleRate(new_family),
                                    std::memory_order_release);
        return true;
    }

    return false;
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

static const struct pw_stream_events input_stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
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

    // Parse filter paths (supports both single and dual-rate modes)
    // Usage: pipewire_daemon [filter_44k.bin] [filter_48k.bin]
    //   Single-rate: pipewire_daemon filter.bin
    //   Dual-rate:   pipewire_daemon filter_44k.bin filter_48k.bin
    std::string filter_path_44k = "data/coefficients/filter_44k_2m_min_phase.bin";
    std::string filter_path_48k = "data/coefficients/filter_48k_2m_min_phase.bin";
    bool use_dual_rate = true;

    if (argc == 2) {
        // Single filter path - use legacy single-rate mode
        filter_path_44k = argv[1];
        use_dual_rate = false;
    } else if (argc >= 3) {
        // Dual filter paths
        filter_path_44k = argv[1];
        filter_path_48k = argv[2];
        use_dual_rate = true;
    }

    // Install signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Initialize GPU upsampler
    std::cout << "Initializing GPU upsampler..." << std::endl;
    g_upsampler = new ConvolutionEngine::GPUUpsampler();

    bool init_success = false;
    if (use_dual_rate) {
        std::cout << "Mode: Dual-Rate (44.1kHz + 48kHz families)" << std::endl;
        init_success = g_upsampler->initializeDualRate(filter_path_44k, filter_path_48k,
                                                       DEFAULT_UPSAMPLE_RATIO, DEFAULT_BLOCK_SIZE,
                                                       ConvolutionEngine::RateFamily::RATE_44K);
    } else {
        std::cout << "Mode: Single-Rate (44.1kHz family only)" << std::endl;
        init_success =
            g_upsampler->initialize(filter_path_44k, DEFAULT_UPSAMPLE_RATIO, DEFAULT_BLOCK_SIZE);
    }

    if (!init_success) {
        std::cerr << "Failed to initialize GPU upsampler" << std::endl;
        delete g_upsampler;
        return 1;
    }
    std::cout << "GPU upsampler ready (16x upsampling, " << DEFAULT_BLOCK_SIZE << " samples/block)"
              << std::endl;

    // Load config and set phase type
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

    if (!g_upsampler->initializeStreaming()) {
        std::cerr << "Failed to initialize streaming mode" << std::endl;
        delete g_upsampler;
        return 1;
    }
    std::cout << std::endl;

    // Initialize PipeWire
    pw_init(&argc, &argv);

    g_output_buffer_left.init(OUTPUT_RING_CAPACITY);
    g_output_buffer_right.init(OUTPUT_RING_CAPACITY);

    Data data = {};
    data.gpu_ready = true;

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

    // Run main loop
    while (g_running) {
        pw_main_loop_run(data.loop);
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
    pw_deinit();

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
