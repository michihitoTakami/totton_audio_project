#include "convolution_engine.h"
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/props.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <csignal>
#include <atomic>

// Configuration
constexpr int INPUT_SAMPLE_RATE = 44100;
constexpr int OUTPUT_SAMPLE_RATE = 705600;  // 16x upsampling
constexpr int UPSAMPLE_RATIO = 16;
constexpr int BLOCK_SIZE = 4096;
constexpr int CHANNELS = 2;

// Global state
static std::atomic<bool> g_running{true};
static ConvolutionEngine::GPUUpsampler* g_upsampler = nullptr;

// Ring buffers for async processing
static std::vector<float> g_input_buffer_left;
static std::vector<float> g_input_buffer_right;
static std::vector<float> g_output_buffer_left;
static std::vector<float> g_output_buffer_right;
static size_t g_output_read_pos = 0;

// PipeWire objects
struct Data {
    struct pw_main_loop* loop;
    struct pw_stream* input_stream;
    struct pw_stream* output_stream;
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
            // Store output for consumption by output stream
            g_output_buffer_left.insert(g_output_buffer_left.end(),
                                       output_left.begin(), output_left.end());
            g_output_buffer_right.insert(g_output_buffer_right.end(),
                                        output_right.begin(), output_right.end());
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
        size_t available = g_output_buffer_left.size() - g_output_read_pos;

        if (available >= n_frames) {
            // Interleave output (separate L/R → stereo interleaved)
            for (uint32_t i = 0; i < n_frames; ++i) {
                output_samples[i * 2] = g_output_buffer_left[g_output_read_pos + i];
                output_samples[i * 2 + 1] = g_output_buffer_right[g_output_read_pos + i];
            }
            g_output_read_pos += n_frames;

            // Clear consumed data when buffer gets large
            if (g_output_read_pos > 100000) {
                g_output_buffer_left.erase(g_output_buffer_left.begin(),
                                          g_output_buffer_left.begin() + g_output_read_pos);
                g_output_buffer_right.erase(g_output_buffer_right.begin(),
                                           g_output_buffer_right.begin() + g_output_read_pos);
                g_output_read_pos = 0;
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
    std::cout << "  44.1kHz → 705.6kHz (16x upsampling)" << std::endl;
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

    // Initialize PipeWire
    pw_init(&argc, &argv);

    Data data = {};
    data.gpu_ready = true;

    // Create main loop
    data.loop = pw_main_loop_new(nullptr);
    struct pw_loop* loop = pw_main_loop_get_loop(data.loop);

    // Create input stream to capture from gpu_upsampler_sink.monitor
    std::cout << "Creating input stream (capturing from gpu_upsampler_sink)..." << std::endl;
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
            PW_STREAM_FLAG_RT_PROCESS |
            PW_STREAM_FLAG_AUTOCONNECT
        ),
        input_params, 1
    );

    // Create output stream (705.6kHz playback to DAC)
    std::cout << "Creating output stream (705.6kHz)..." << std::endl;
    data.output_stream = pw_stream_new_simple(
        loop,
        "GPU Upsampler Output",
        pw_properties_new(
            PW_KEY_MEDIA_TYPE, "Audio",
            PW_KEY_MEDIA_CATEGORY, "Playback",
            PW_KEY_MEDIA_ROLE, "Music",
            PW_KEY_NODE_TARGET, "alsa_output.usb-SMSL_SMSL_USB_AUDIO-00.iec958-stereo",
            "audio.channels", "2",
            "audio.position", "FL,FR",
            nullptr
        ),
        &output_stream_events,
        &data
    );

    // Configure output stream audio format (32-bit float stereo @ 705.6kHz)
    uint8_t output_buffer[1024];
    struct spa_pod_builder output_builder = SPA_POD_BUILDER_INIT(output_buffer, sizeof(output_buffer));

    struct spa_audio_info_raw output_info = {};
    output_info.format = SPA_AUDIO_FORMAT_F32;
    output_info.rate = OUTPUT_SAMPLE_RATE;
    output_info.channels = CHANNELS;
    output_info.position[0] = SPA_AUDIO_CHANNEL_FL;
    output_info.position[1] = SPA_AUDIO_CHANNEL_FR;

    const struct spa_pod* output_params[1];
    output_params[0] = spa_format_audio_raw_build(&output_builder, SPA_PARAM_EnumFormat, &output_info);

    pw_stream_connect(
        data.output_stream,
        PW_DIRECTION_OUTPUT,
        PW_ID_ANY,
        static_cast<pw_stream_flags>(
            PW_STREAM_FLAG_MAP_BUFFERS |
            PW_STREAM_FLAG_RT_PROCESS |
            PW_STREAM_FLAG_AUTOCONNECT
        ),
        output_params, 1
    );

    std::cout << std::endl;
    std::cout << "System ready. Audio routing configured:" << std::endl;
    std::cout << "  1. Applications → gpu_upsampler_sink (select in GNOME settings)" << std::endl;
    std::cout << "  2. gpu_upsampler_sink.monitor → GPU Upsampler (16x upsampling)" << std::endl;
    std::cout << "  3. GPU Upsampler → SMSL DAC (705.6kHz output)" << std::endl;
    std::cout << std::endl;
    std::cout << "Select 'GPU Upsampler (705.6kHz)' as output device in sound settings." << std::endl;
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
