#include "daemon/input/i2s_capture.h"

#include "core/daemon_constants.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/core/thread_priority.h"
#include "daemon/input/loopback_capture.h"
#include "logging/logger.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

namespace daemon_input {

snd_pcm_format_t parseI2sFormat(const std::string& formatStr) {
    // Same set as loopback (MVP)
    return parseLoopbackFormat(formatStr);
}

bool validateI2sConfig(const AppConfig& cfg) {
    if (!cfg.i2s.enabled) {
        return true;
    }
    if (cfg.i2s.device.empty()) {
        LOG_ERROR("[I2S] device must not be empty");
        return false;
    }
    if (cfg.i2s.channels != DaemonConstants::CHANNELS) {
        LOG_ERROR("[I2S] Unsupported channels {} (expected {})", cfg.i2s.channels,
                  DaemonConstants::CHANNELS);
        return false;
    }
    if (cfg.i2s.periodFrames == 0) {
        LOG_ERROR("[I2S] periodFrames must be > 0");
        return false;
    }
    if (parseI2sFormat(cfg.i2s.format) == SND_PCM_FORMAT_UNKNOWN) {
        LOG_ERROR("[I2S] Unsupported format '{}'", cfg.i2s.format);
        return false;
    }
    // sampleRate may be 0 (follow runtime/negotiated); otherwise accept MVP rates.
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

        // ALSA "hw/plughw" allow both:
        // - card,device
        // - card,device,subdevice
        // Field reports show Jetson environments where one form works and the other fails.
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

        // Use non-blocking mode so shutdown doesn't hang on snd_pcm_readi().
        int err =
            snd_pcm_open(&handle, target_device.c_str(), SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK);
        if (err < 0) {
            LOG_ERROR("[I2S] Cannot open capture device {}: {}", target_device, snd_strerror(err));
            return nullptr;
        }

        snd_pcm_hw_params_t* hw_params;
        snd_pcm_hw_params_alloca(&hw_params);
        snd_pcm_hw_params_any(handle, hw_params);

        if ((err = snd_pcm_hw_params_set_access(handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) <
                0 ||
            (err = snd_pcm_hw_params_set_format(handle, hw_params, format)) < 0) {
            LOG_ERROR("[I2S] Cannot set access/format: {}", snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }

        if ((err = snd_pcm_hw_params_set_channels(handle, hw_params, channels)) < 0) {
            LOG_ERROR("[I2S] Cannot set channels {}: {}", channels, snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }

        unsigned int rate_near = candidate_rate;
        if (!auto_rate) {
            if ((err = snd_pcm_hw_params_set_rate_near(handle, hw_params, &rate_near, nullptr)) <
                0) {
                LOG_ERROR("[I2S] Cannot set rate {}: {}", candidate_rate, snd_strerror(err));
                snd_pcm_close(handle);
                return nullptr;
            }
            if (rate_near != candidate_rate) {
                LOG_ERROR("[I2S] Requested rate {} not supported (got {})", candidate_rate,
                          rate_near);
                snd_pcm_close(handle);
                return nullptr;
            }
        }

        snd_pcm_uframes_t buffer_frames =
            static_cast<snd_pcm_uframes_t>(std::max<uint32_t>(local_period * 4, local_period));
        if ((err = snd_pcm_hw_params_set_period_size_near(handle, hw_params, &local_period,
                                                          nullptr)) < 0) {
            LOG_ERROR("[I2S] Cannot set period size: {}", snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }
        buffer_frames = std::max<snd_pcm_uframes_t>(buffer_frames, local_period * 2);
        if ((err = snd_pcm_hw_params_set_buffer_size_near(handle, hw_params, &buffer_frames)) < 0) {
            LOG_ERROR("[I2S] Cannot set buffer size: {}", snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }

        if ((err = snd_pcm_hw_params(handle, hw_params)) < 0) {
            LOG_ERROR(
                "[I2S] Cannot apply hardware parameters (rate {}, ch {}, fmt {}, "
                "period {} frames, buffer {} frames): {}",
                rate_label, channels, snd_pcm_format_name(format), local_period, buffer_frames,
                snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }

        snd_pcm_hw_params_get_period_size(hw_params, &local_period, nullptr);
        snd_pcm_hw_params_get_buffer_size(hw_params, &buffer_frames);
        unsigned int effective_rate = candidate_rate;
        snd_pcm_hw_params_get_rate(hw_params, &effective_rate, nullptr);
        if (!auto_rate && effective_rate != candidate_rate) {
            LOG_ERROR("[I2S] Negotiated rate {} differs from requested {}", effective_rate,
                      candidate_rate);
            snd_pcm_close(handle);
            return nullptr;
        }

        if ((err = snd_pcm_prepare(handle)) < 0) {
            LOG_ERROR("[I2S] Cannot prepare capture device: {}", snd_strerror(err));
            snd_pcm_close(handle);
            return nullptr;
        }

        // Ensure non-blocking is effective even if driver flips it.
        snd_pcm_nonblock(handle, 1);

        period_frames = local_period;
        actual_rate = effective_rate;

        LOG_INFO(
            "[I2S] Capture device {} configured ({} Hz, {} ch, fmt={}, period {} frames, "
            "buffer {} frames)",
            target_device, actual_rate, channels, snd_pcm_format_name(format), period_frames,
            buffer_frames);

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
                if (candidate_device != device) {
                    LOG_WARN("[I2S] Using ALSA device fallback '{}' (configured was '{}')",
                             candidate_device, device);
                }
                return handle;
            }
        }
    }

    return nullptr;
}

void i2sCaptureThread(daemon_app::RuntimeState& state, const std::string& device,
                      snd_pcm_format_t format, unsigned int requestedRate, unsigned int channels,
                      snd_pcm_uframes_t periodFrames) {
    daemon_core::elevateRealtimePriority("I2S capture");

    if (channels != static_cast<unsigned int>(DaemonConstants::CHANNELS)) {
        LOG_ERROR("[I2S] Unsupported channel count {} (expected {})", channels,
                  DaemonConstants::CHANNELS);
        return;
    }

    const snd_pcm_uframes_t configured_period_frames = periodFrames;
    std::vector<int16_t> buffer16;
    std::vector<int32_t> buffer32;
    std::vector<uint8_t> buffer24;
    std::vector<float> floatBuffer;

    state.i2s.captureRunning.store(true, std::memory_order_release);

    while (state.flags.running.load(std::memory_order_acquire)) {
        snd_pcm_uframes_t negotiated_period = configured_period_frames;
        unsigned int actual_rate = requestedRate;
        snd_pcm_t* handle = open_i2s_capture(device, format, requestedRate, channels,
                                             negotiated_period, actual_rate);
        {
            std::lock_guard<std::mutex> lock(state.i2s.handleMutex);
            state.i2s.handle = handle;
        }

        if (!handle) {
            // Device not ready / unplugged. Retry with backoff.
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

        // If the hardware rate differs from current engine input rate, schedule follow-up.
        // (Actual reinit is handled in main loop; network-driven follow is #824.)
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
            // Important: For I2S capture in external-clock/slave scenarios, we must keep draining
            // the ALSA capture buffer. Applying backpressure here can stall reads long enough to
            // trigger capture overruns (XRUN), resulting in periodic "burst + noise" artifacts.
            //
            // If downstream is behind, we prefer to drop at the daemon-level playback queue
            // (PlaybackBufferManager::enqueue enforces capacity) rather than blocking capture.

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
                waitForCaptureReady(handle);
                continue;
            }
            if (frames == -EPIPE) {
                LOG_WARN("[I2S] XRUN detected, recovering");
                // Use recover() to handle driver-specific XRUN recovery requirements.
                if (snd_pcm_recover(handle, frames, 1) < 0) {
                    snd_pcm_prepare(handle);
                }
                continue;
            }
            if (frames < 0) {
                LOG_WARN("[I2S] Read error: {}", snd_strerror(frames));
                if (snd_pcm_recover(handle, frames, 1) < 0) {
                    // Treat as a device fault; close and attempt reopen.
                    LOG_ERROR("[I2S] Recover failed, attempting reopen");
                    needReopen = true;
                    break;
                }
                continue;
            }
            if (frames == 0) {
                waitForCaptureReady(handle);
                continue;
            }

            if (!convertPcmToFloat(rawBuffer, format, static_cast<size_t>(frames), channels,
                                   floatBuffer)) {
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

}  // namespace daemon_input
