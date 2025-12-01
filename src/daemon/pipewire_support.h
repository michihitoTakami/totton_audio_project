#pragma once

#include <algorithm>
#include <mutex>
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <vector>

namespace pipewire_support {

struct InputContext {
    struct pw_main_loop* loop = nullptr;
    struct pw_stream* input_stream = nullptr;
    struct spa_source* signal_check_timer = nullptr;  // Timer for checking signal flags
    bool gpu_ready = false;
};

inline void process_interleaved_block(const float* input_samples, uint32_t n_frames) {
    if (!input_samples || n_frames == 0 || !g_upsampler) {
        return;
    }

    std::lock_guard<std::mutex> inputLock(g_input_process_mutex);

    std::vector<float> left(n_frames);
    std::vector<float> right(n_frames);
    AudioUtils::deinterleaveStereo(input_samples, left.data(), right.data(), n_frames);
    float inputPeak = compute_stereo_peak(left.data(), right.data(), n_frames);
    update_peak_level(g_peak_input_level, inputPeak);

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
        update_peak_level(g_peak_upsampler_level, upsamplerPeak);
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
                    update_peak_level(g_peak_post_crossfeed_level, cfPeak);
                }
                std::lock_guard<std::mutex> lock(g_buffer_mutex);
                g_output_buffer_left.insert(g_output_buffer_left.end(), g_cf_output_left.begin(),
                                            g_cf_output_left.end());
                g_output_buffer_right.insert(g_output_buffer_right.end(), g_cf_output_right.begin(),
                                             g_cf_output_right.end());
                g_buffer_cv.notify_one();
                return;
            }
        }
    }

    std::lock_guard<std::mutex> lock(g_buffer_mutex);
    g_output_buffer_left.insert(g_output_buffer_left.end(), output_left.begin(), output_left.end());
    g_output_buffer_right.insert(g_output_buffer_right.end(), output_right.begin(),
                                 output_right.end());
    g_buffer_cv.notify_one();
    if (frames_generated > 0) {
        float postPeak =
            compute_stereo_peak(output_left.data(), output_right.data(), frames_generated);
        update_peak_level(g_peak_post_crossfeed_level, postPeak);
    }
}

inline void on_input_process(void* userdata) {
    InputContext* data = static_cast<InputContext*>(userdata);
    if (!data || !data->input_stream) {
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
        process_interleaved_block(input_samples, n_frames);
    }

    pw_stream_queue_buffer(data->input_stream, buf);
}

inline void on_stream_state_changed(void* userdata, enum pw_stream_state old_state,
                                    enum pw_stream_state state, const char* error) {
    (void)userdata;
    (void)old_state;

    std::cout << "PipeWire input stream state: " << pw_stream_state_as_string(state);
    if (error) {
        std::cout << " (error: " << error << ")";
    }
    std::cout << std::endl;
}

inline void on_param_changed(void* userdata, uint32_t id, const struct spa_pod* param) {
    (void)userdata;

    if (id != SPA_PARAM_Format || param == nullptr) {
        return;
    }

    struct spa_audio_info_raw info;
    if (spa_format_audio_raw_parse(param, &info) < 0) {
        return;
    }

    int detected_rate = static_cast<int>(info.rate);
    int current_rate = g_current_input_rate.load(std::memory_order_acquire);

    if (detected_rate != current_rate && detected_rate > 0) {
        LOG_INFO("[PipeWire] Sample rate change detected: {} -> {} Hz", current_rate,
                 detected_rate);
        g_pending_rate_change.store(detected_rate, std::memory_order_release);
    }
}

inline constexpr struct pw_stream_events kInputStreamEvents = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
    .param_changed = on_param_changed,
    .process = on_input_process,
};

inline bool handle_rate_change(int detected_sample_rate) {
    if (!g_upsampler) {
        return false;
    }

    if (!g_upsampler->isMultiRateEnabled()) {
        LOG_ERROR("[Rate] Multi-rate mode not enabled. Rate switching requires multi-rate mode.");
        return false;
    }

    int prev_input_rate = g_current_input_rate.load(std::memory_order_acquire);
    int prev_output_rate = g_current_output_rate.load(std::memory_order_acquire);

    bool switch_success = false;

    applySoftMuteForFilterSwitch([&]() {
        if (!g_upsampler->switchToInputRate(detected_sample_rate)) {
            LOG_ERROR("[Rate] Failed to switch to input rate: {} Hz", detected_sample_rate);
            return false;
        }

        g_current_input_rate.store(detected_sample_rate, std::memory_order_release);

        int new_output_rate = g_upsampler->getOutputSampleRate();
        g_current_output_rate.store(new_output_rate, std::memory_order_release);

        g_set_rate_family(ConvolutionEngine::detectRateFamily(detected_sample_rate));

        if (!g_upsampler->initializeStreaming()) {
            LOG_ERROR("[Rate] Failed to reinitialize streaming, rolling back...");
            if (g_upsampler->switchToInputRate(prev_input_rate)) {
                g_upsampler->initializeStreaming();
            }
            g_current_input_rate.store(prev_input_rate, std::memory_order_release);
            g_current_output_rate.store(prev_output_rate, std::memory_order_release);
            return false;
        }

        size_t new_capacity = g_upsampler->getStreamValidInputPerBlock() * 2;
        g_stream_input_left.resize(new_capacity, 0.0f);
        g_stream_input_right.resize(new_capacity, 0.0f);
        g_stream_accumulated_left = 0;
        g_stream_accumulated_right = 0;

        LOG_INFO("[Rate] Streaming buffers resized to {} samples (streamValidInputPerBlock={})",
                 new_capacity, g_upsampler->getStreamValidInputPerBlock());

        g_output_buffer_left.clear();
        g_output_buffer_right.clear();
        LOG_INFO("[Rate] Output ring buffers cleared");

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

}  // namespace pipewire_support


