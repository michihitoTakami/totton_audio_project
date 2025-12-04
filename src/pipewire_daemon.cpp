#include "audio_ring_buffer.h"
#include "audio_utils.h"
#include "config_loader.h"
#include "convolution_engine.h"
#include "daemon_constants.h"
#include "partition_runtime_utils.h"
#include "soft_mute.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/props.h>
#include <thread>
#include <vector>

// Configuration (using common constants from daemon_constants.h)
using namespace DaemonConstants;

struct PipewireRuntimeContext {
    AppConfig config{};
    std::atomic<float> limiterGain{1.0f};
    std::atomic<float> effectiveGain{1.0f};
    std::atomic<int> currentInputRate{DEFAULT_INPUT_SAMPLE_RATE};
    std::atomic<int> currentOutputRate{DEFAULT_OUTPUT_SAMPLE_RATE};
    std::atomic<int> currentRateFamilyInt{
        static_cast<int>(ConvolutionEngine::RateFamily::RATE_44K)};
    std::atomic<bool> running{true};
    std::atomic<int> pendingRateChange{0};
    AudioRingBuffer outputBufferLeft;
    AudioRingBuffer outputBufferRight;
    std::vector<float> outputTempLeft;
    std::vector<float> outputTempRight;
    std::unique_ptr<ConvolutionEngine::GPUUpsampler> upsampler;
    std::unique_ptr<SoftMute::Controller> softMute;

    ConvolutionEngine::RateFamily getRateFamily() const {
        return static_cast<ConvolutionEngine::RateFamily>(
            currentRateFamilyInt.load(std::memory_order_acquire));
    }

    void setRateFamily(ConvolutionEngine::RateFamily family) {
        currentRateFamilyInt.store(static_cast<int>(family), std::memory_order_release);
    }
};

static void enforce_phase_partition_constraints(AppConfig& config) {
    if (config.partitionedConvolution.enabled && config.phaseType == PhaseType::Linear) {
        std::cout << "[Partition] Hybrid phase is incompatible with low-latency mode. "
                  << "Switching to minimum phase." << std::endl;
        config.phaseType = PhaseType::Minimum;
    }
}

// Forward declaration for Data struct (defined below)
struct Data;

// Signal handler needs access to runtime context
namespace {
PipewireRuntimeContext* g_runtime_ctx = nullptr;
}

static float apply_output_limiter(PipewireRuntimeContext& ctx, float* interleaved, size_t frames) {
    constexpr float kEpsilon = 1e-6f;
    if (!interleaved || frames == 0) {
        ctx.limiterGain.store(1.0f, std::memory_order_relaxed);
        ctx.effectiveGain.store(ctx.config.gain, std::memory_order_relaxed);
        return 0.0f;
    }
    float peak = AudioUtils::computeInterleavedPeak(interleaved, frames);
    float limiterGain = 1.0f;
    const float target = ctx.config.headroomTarget;
    if (target > 0.0f && peak > target) {
        limiterGain = target / (peak + kEpsilon);
        AudioUtils::applyInterleavedGain(interleaved, frames, limiterGain);
        peak = target;
    }
    ctx.limiterGain.store(limiterGain, std::memory_order_relaxed);
    ctx.effectiveGain.store(ctx.config.gain * limiterGain, std::memory_order_relaxed);
    return peak;
}

// Use 768kHz (48k family max) as base to ensure sufficient capacity for both rate families
static constexpr size_t OUTPUT_RING_CAPACITY = 768000 * 2;  // ~2 seconds per channel at max rate

// PipeWire objects
using StreamBuffer = ConvolutionEngine::StreamFloatVector;

struct Data {
    PipewireRuntimeContext* ctx = nullptr;
    struct pw_main_loop* loop;
    struct pw_stream* input_stream;
    struct pw_stream* output_stream;
    bool gpu_ready;

    // Scratch buffers to avoid allocations in real-time callbacks
    std::vector<float> input_left;
    std::vector<float> input_right;
    StreamBuffer output_left;
    StreamBuffer output_right;

    // Streaming input accumulation buffers (per channel)
    StreamBuffer stream_input_left;
    StreamBuffer stream_input_right;
    size_t stream_accum_left = 0;
    size_t stream_accum_right = 0;

    // Flag to indicate output stream needs reconnection after rate family change
    bool needs_output_reconnect = false;
    int new_output_rate = 0;
};

// Signal handler for graceful shutdown
static void signal_handler(int sig) {
    std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    if (g_runtime_ctx) {
        g_runtime_ctx->running.store(false, std::memory_order_release);
    }
}

// Rate family switching helper
// Called when input sample rate changes (e.g., from PipeWire param event or ZeroMQ)
// Connected via on_param_changed() event and processed in main loop.
// See: https://github.com/michihitoTakami/michy_os/issues/218
// Implements soft mute during filter switching (Issue #266)
static bool handle_rate_change(PipewireRuntimeContext& ctx, Data& data, int detected_sample_rate) {
    auto* upsampler = ctx.upsampler.get();
    if (!upsampler) {
        return false;
    }

    // Multi-rate mode: use switchToInputRate() for dynamic rate switching
    if (!upsampler->isMultiRateEnabled()) {
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
    int current_output_rate = ctx.currentOutputRate.load();
    int originalFadeDuration = DEFAULT_SOFT_MUTE_FADE_MS;

    auto* softMute = ctx.softMute.get();

    if (softMute && current_output_rate > 0) {
        // Save original fade duration for restoration
        originalFadeDuration = softMute->getFadeDuration();

        // Update fade duration for filter switching
        softMute->setFadeDuration(FILTER_SWITCH_FADE_MS);
        softMute->setSampleRate(current_output_rate);

        std::cout << "[Rate] Starting fade-out for filter switch ("
                  << (FILTER_SWITCH_FADE_MS / 1000.0) << "s)..." << std::endl;
        softMute->startFadeOut();

        // Wait for fade-out to complete (approximately 1.5 seconds)
        // Polling is necessary because fade is processed in audio thread
        auto fade_start = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::milliseconds(FILTER_SWITCH_FADE_TIMEOUT_MS);
        while (softMute->isTransitioning() &&
               softMute->getState() == SoftMute::MuteState::FADING_OUT) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto elapsed = std::chrono::steady_clock::now() - fade_start;
            if (elapsed > timeout) {
                std::cerr << "[Rate] Warning: Fade-out timeout (" << FILTER_SWITCH_FADE_TIMEOUT_MS
                          << "ms), proceeding with switch" << std::endl;
                break;
            }
        }

        // Ensure we're fully muted before switching
        if (softMute->getState() != SoftMute::MuteState::MUTED) {
            softMute->setMuted();
        }
    }

    // Save previous state for rollback
    int prev_input_rate = ctx.currentInputRate.load(std::memory_order_acquire);
    int prev_output_rate = ctx.currentOutputRate.load(std::memory_order_acquire);

    // Perform filter switch
    bool switch_success = false;

    if (upsampler->switchToInputRate(detected_sample_rate)) {
        ctx.currentInputRate.store(detected_sample_rate, std::memory_order_release);

        // Output rate is dynamically calculated (input_rate × upsample_ratio)
        int new_output_rate = upsampler->getOutputSampleRate();
        ctx.currentOutputRate.store(new_output_rate, std::memory_order_release);

        // Re-initialize streaming after rate switch (Issue #219)
        // switchToInputRate() invalidates streaming buffers, so we must reinitialize
        if (!upsampler->initializeStreaming()) {
            std::cerr << "[Rate] Failed to reinitialize streaming, rolling back..." << std::endl;
            // Rollback: switch back to previous rate
            if (upsampler->switchToInputRate(prev_input_rate)) {
                upsampler->initializeStreaming();
            }
            ctx.currentInputRate.store(prev_input_rate, std::memory_order_release);
            ctx.currentOutputRate.store(prev_output_rate, std::memory_order_release);
            if (softMute) {
                softMute->setPlaying();
                softMute->setFadeDuration(originalFadeDuration);
            }
            return false;
        }

        // Resize streaming input buffers based on new streamValidInputPerBlock (Issue #219)
        size_t new_capacity = upsampler->getStreamValidInputPerBlock() * 2;
        data.stream_input_left.resize(new_capacity, 0.0f);
        data.stream_input_right.resize(new_capacity, 0.0f);

        // Clear accumulated samples (old rate data is invalid)
        data.stream_accum_left = 0;
        data.stream_accum_right = 0;

        std::cout << "[Rate] Streaming buffers resized to " << new_capacity
                  << " samples (streamValidInputPerBlock="
                  << upsampler->getStreamValidInputPerBlock() << ")" << std::endl;

        // Clear output ring buffers to discard old rate samples (Issue #219)
        ctx.outputBufferLeft.clear();
        ctx.outputBufferRight.clear();
        std::cout << "[Rate] Output ring buffers cleared" << std::endl;

        // Update soft mute sample rate if output rate changed
        if (softMute && new_output_rate != current_output_rate) {
            softMute->setSampleRate(new_output_rate);
        }

        // Output rate change only requires reconnection if it actually changed
        // (same-family hi-res switches keep the same output rate)
        if (new_output_rate != current_output_rate) {
            data.needs_output_reconnect = true;
            data.new_output_rate = new_output_rate;
            std::cout << "[Rate] Output stream reconnection scheduled for " << new_output_rate
                      << " Hz (" << upsampler->getUpsampleRatio() << "x upsampling)" << std::endl;
        } else {
            std::cout << "[Rate] Rate switched to " << detected_sample_rate << " Hz -> "
                      << new_output_rate << " Hz (" << upsampler->getUpsampleRatio()
                      << "x upsampling)" << std::endl;
        }
        switch_success = true;
    } else {
        std::cerr << "[Rate] Failed to switch to input rate: " << detected_sample_rate << " Hz"
                  << std::endl;
    }

    // Start fade-in after filter switch (or restore state on failure)
    if (softMute) {
        if (switch_success) {
            std::cout << "[Rate] Starting fade-in after filter switch ("
                      << (FILTER_SWITCH_FADE_MS / 1000.0) << "s)..." << std::endl;
            softMute->startFadeIn();
            // Fade duration will be reset to default in output processing thread when fade-in
            // completes
        } else {
            // If switch failed, restore original state immediately
            std::cerr << "[Rate] Switch failed, restoring audio state" << std::endl;
            softMute->setPlaying();
            softMute->setFadeDuration(originalFadeDuration);
        }
    }

    return switch_success;
}

// Input stream process callback (44.1kHz audio from PipeWire)
static void on_input_process(void* userdata) {
    Data* data = static_cast<Data*>(userdata);
    if (!data || !data->ctx) {
        return;
    }
    auto* ctx = data->ctx;
    auto* upsampler = ctx->upsampler.get();
    if (!upsampler) {
        return;
    }

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
            bool left_generated = upsampler->processStreamBlock(
                data->input_left.data(), frames_to_process, data->output_left,
                upsampler->streamLeft_, data->stream_input_left, data->stream_accum_left);
            bool right_generated = upsampler->processStreamBlock(
                data->input_right.data(), frames_to_process, data->output_right,
                upsampler->streamRight_, data->stream_input_right, data->stream_accum_right);

            if (left_generated && right_generated) {
                // Store output for consumption by output stream
                if (!ctx->outputBufferLeft.write(data->output_left.data(),
                                                 data->output_left.size()) ||
                    !ctx->outputBufferRight.write(data->output_right.data(),
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
    if (!data || !data->ctx) {
        return;
    }
    auto* ctx = data->ctx;

    struct pw_buffer* buf = pw_stream_dequeue_buffer(data->output_stream);
    if (!buf) {
        return;
    }

    struct spa_buffer* spa_buf = buf->buffer;
    float* output_samples = static_cast<float*>(spa_buf->datas[0].data);
    uint32_t n_frames = spa_buf->datas[0].chunk->size / (sizeof(float) * CHANNELS);

    if (output_samples && n_frames > 0) {
        size_t available = std::min(ctx->outputBufferLeft.availableToRead(),
                                    ctx->outputBufferRight.availableToRead());

        if (available >= n_frames) {
            if (ctx->outputTempLeft.size() < n_frames)
                ctx->outputTempLeft.resize(n_frames);
            if (ctx->outputTempRight.size() < n_frames)
                ctx->outputTempRight.resize(n_frames);

            bool read_left = ctx->outputBufferLeft.read(ctx->outputTempLeft.data(), n_frames);
            bool read_right = ctx->outputBufferRight.read(ctx->outputTempRight.data(), n_frames);

            if (read_left && read_right) {
                // Interleave output (separate L/R → stereo interleaved)
                AudioUtils::interleaveStereo(ctx->outputTempLeft.data(),
                                             ctx->outputTempRight.data(), output_samples, n_frames);
                AudioUtils::applyInterleavedGain(output_samples, n_frames, ctx->config.gain);

                // Apply soft mute if transitioning (for filter switching)
                if (ctx->softMute) {
                    ctx->softMute->process(output_samples, n_frames);

                    // Reset fade duration to default after filter switch fade-in completes
                    // Check if we just completed a fade-in from filter switching
                    using namespace DaemonConstants;
                    if (ctx->softMute->getState() == SoftMute::MuteState::PLAYING &&
                        ctx->softMute->getFadeDuration() > DEFAULT_SOFT_MUTE_FADE_MS) {
                        ctx->softMute->setFadeDuration(DEFAULT_SOFT_MUTE_FADE_MS);
                    }
                }

                apply_output_limiter(*ctx, output_samples, n_frames);
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
    Data* data = static_cast<Data*>(userdata);
    PipewireRuntimeContext* ctx = data ? data->ctx : nullptr;
    if (!ctx) {
        return;
    }
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
    int current_rate = ctx->currentInputRate.load(std::memory_order_acquire);

    // Check if rate has changed
    if (detected_rate != current_rate && detected_rate > 0) {
        std::cout << "[PipeWire] Sample rate change detected: " << current_rate << " -> "
                  << detected_rate << " Hz" << std::endl;

        // Set pending rate change (will be processed in main loop)
        // This avoids blocking the real-time audio thread
        ctx->pendingRateChange.store(detected_rate, std::memory_order_release);
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

    AppConfig appConfig;
    bool configLoaded = loadAppConfig(DEFAULT_CONFIG_FILE, appConfig, false);
    enforce_phase_partition_constraints(appConfig);
    PartitionRuntime::RuntimeRequest partitionRequest{
        appConfig.partitionedConvolution.enabled, appConfig.eqEnabled, appConfig.crossfeed.enabled};

    PipewireRuntimeContext ctx;
    ctx.config = appConfig;
    g_runtime_ctx = &ctx;

    // Install signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Initialize GPU upsampler (multi-rate mode only)
    std::cout << "Initializing GPU upsampler..." << std::endl;
    ctx.upsampler = std::make_unique<ConvolutionEngine::GPUUpsampler>();
    ctx.upsampler->setPartitionedConvolutionConfig(appConfig.partitionedConvolution);

    std::cout << "Mode: Multi-Rate (all supported rates: 44.1k/48k families, 16x/8x/4x/2x)"
              << std::endl;
    bool init_success = ctx.upsampler->initializeMultiRate(
        coefficient_dir, DEFAULT_BLOCK_SIZE,
        44100);  // Initial input rate (will be updated by PipeWire detection)

    if (!init_success) {
        std::cerr << "Failed to initialize GPU upsampler" << std::endl;
        return 1;
    }
    std::cout << "GPU upsampler ready (" << ctx.upsampler->getUpsampleRatio() << "x upsampling, "
              << DEFAULT_BLOCK_SIZE << " samples/block)" << std::endl;

    // Load config and set phase type (input sample rate is auto-detected from PipeWire)
    if (configLoaded) {
        ctx.upsampler->setPhaseType(appConfig.phaseType);
        std::cout << "Phase type: " << phaseTypeToString(appConfig.phaseType) << std::endl;

        // Log latency warning for linear phase
        if (appConfig.phaseType == PhaseType::Linear) {
            double latencySec = ctx.upsampler->getLatencySeconds();
            std::cout << "  WARNING: Hybrid phase latency: " << latencySec << " seconds ("
                      << ctx.upsampler->getLatencySamples() << " samples)" << std::endl;
        }
    } else {
        std::cout << "Phase type: minimum (default)" << std::endl;
    }
    ctx.limiterGain.store(1.0f, std::memory_order_relaxed);
    ctx.effectiveGain.store(ctx.config.gain, std::memory_order_relaxed);
    ctx.currentInputRate.store(ctx.upsampler->getInputSampleRate(), std::memory_order_release);
    ctx.currentOutputRate.store(ctx.upsampler->getOutputSampleRate(), std::memory_order_release);
    std::cout << "Input sample rate: auto-detected from PipeWire" << std::endl;

    if (!ctx.upsampler->initializeStreaming()) {
        std::cerr << "Failed to initialize streaming mode" << std::endl;
        return 1;
    }
    PartitionRuntime::applyPartitionPolicy(partitionRequest, *ctx.upsampler, ctx.config,
                                           "PipeWire");

    // Initialize soft mute controller (default fade duration, will be extended for filter
    // switching)
    int initial_output_rate = ctx.upsampler->getOutputSampleRate();
    ctx.softMute =
        std::make_unique<SoftMute::Controller>(DEFAULT_SOFT_MUTE_FADE_MS, initial_output_rate);
    std::cout << "Soft mute initialized (" << DEFAULT_SOFT_MUTE_FADE_MS << "ms fade at "
              << initial_output_rate << "Hz)" << std::endl;

    std::cout << std::endl;

    // Initialize PipeWire
    pw_init(&argc, &argv);

    ctx.outputBufferLeft.init(OUTPUT_RING_CAPACITY);
    ctx.outputBufferRight.init(OUTPUT_RING_CAPACITY);

    Data data = {};
    data.ctx = &ctx;
    data.gpu_ready = true;

    // Pre-allocate streaming input buffers (based on streamValidInputPerBlock_)
    // Use 2x safety margin to handle timing variations
    size_t buffer_capacity = ctx.upsampler->getStreamValidInputPerBlock() * 2;
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
                          "GPU Upsampler Input", PW_KEY_TARGET_OBJECT, "gpu_upsampler_sink.monitor",
                          "audio.channels", "2", "audio.position", "FL,FR", nullptr),
        &input_stream_events, &data);

    // Configure input stream audio format (32-bit float stereo @ 44.1kHz)
    uint8_t input_buffer[1024];
    struct spa_pod_builder input_builder = SPA_POD_BUILDER_INIT(input_buffer, sizeof(input_buffer));

    struct spa_audio_info_raw input_info = {};
    input_info.format = SPA_AUDIO_FORMAT_F32;
    input_info.rate = ctx.currentInputRate.load();
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
                          PW_KEY_MEDIA_ROLE, "Music", PW_KEY_TARGET_OBJECT,
                          "alsa_output.usb-SMSL_SMSL_USB_AUDIO-00.iec958-stereo", "audio.channels",
                          "2", "audio.position", "FL,FR", nullptr),
        &output_stream_events, &data);

    // Configure output stream audio format (32-bit float stereo @ 705.6kHz)
    uint8_t output_buffer[1024];
    struct spa_pod_builder output_builder =
        SPA_POD_BUILDER_INIT(output_buffer, sizeof(output_buffer));

    struct spa_audio_info_raw output_info = {};
    output_info.format = SPA_AUDIO_FORMAT_F32;
    output_info.rate = ctx.currentOutputRate.load();
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
    while (ctx.running.load(std::memory_order_acquire)) {
        // Check for pending rate change (set by on_param_changed callback)
        int pending_rate = ctx.pendingRateChange.exchange(0, std::memory_order_acq_rel);
        if (pending_rate > 0) {
            std::cout << "[Main] Processing rate change to " << pending_rate << " Hz..."
                      << std::endl;
            if (handle_rate_change(ctx, data, pending_rate)) {
                std::cout << "[Main] Rate change successful: "
                          << ctx.currentInputRate.load(std::memory_order_acquire) << " Hz input -> "
                          << ctx.currentOutputRate.load(std::memory_order_acquire) << " Hz output"
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
                                  PW_KEY_MEDIA_ROLE, "Music", PW_KEY_TARGET_OBJECT,
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
        // to periodically check running flag and pending rate changes
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

    ctx.softMute.reset();
    ctx.upsampler.reset();
    pw_deinit();

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
