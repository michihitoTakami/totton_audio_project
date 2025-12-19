#include "daemon/input/loopback_capture.h"

#include "core/daemon_constants.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/output/playback_buffer_manager.h"
#include "logging/logger.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <thread>

namespace daemon_input {

snd_pcm_format_t parseLoopbackFormat(const std::string& formatStr) {
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

bool validateLoopbackConfig(const AppConfig& cfg) {
    if (!cfg.loopback.enabled) {
        return true;
    }
    if (cfg.loopback.sampleRate != 44100 && cfg.loopback.sampleRate != 48000) {
        LOG_ERROR("[Loopback] Unsupported sample rate {} (expected 44100 or 48000)",
                  cfg.loopback.sampleRate);
        return false;
    }
    if (cfg.loopback.channels != DaemonConstants::CHANNELS) {
        LOG_ERROR("[Loopback] Unsupported channels {} (expected {})", cfg.loopback.channels,
                  DaemonConstants::CHANNELS);
        return false;
    }
    if (cfg.loopback.periodFrames == 0) {
        LOG_ERROR("[Loopback] periodFrames must be > 0");
        return false;
    }
    if (parseLoopbackFormat(cfg.loopback.format) == SND_PCM_FORMAT_UNKNOWN) {
        LOG_ERROR("[Loopback] Unsupported format '{}'", cfg.loopback.format);
        return false;
    }
    return true;
}

bool convertPcmToFloat(const void* src, snd_pcm_format_t format, size_t frames,
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

void waitForCaptureReady(snd_pcm_t* handle) {
    if (!handle) {
        return;
    }
    int ret = snd_pcm_wait(handle, 100);
    if (ret < 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

static snd_pcm_t* openLoopbackCapture(const std::string& device, snd_pcm_format_t format,
                                      unsigned int rate, unsigned int channels,
                                      snd_pcm_uframes_t& periodFrames) {
    snd_pcm_t* handle = nullptr;
    int err = snd_pcm_open(&handle, device.c_str(), SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK);
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
    if (rate_near != rate) {
        LOG_ERROR("[Loopback] Requested rate {} not supported (got {})", rate, rate_near);
        snd_pcm_close(handle);
        return nullptr;
    }

    snd_pcm_uframes_t buffer_frames =
        static_cast<snd_pcm_uframes_t>(std::max<uint32_t>(periodFrames * 4, periodFrames));
    if ((err = snd_pcm_hw_params_set_period_size_near(handle, hw_params, &periodFrames, nullptr)) <
        0) {
        LOG_ERROR("[Loopback] Cannot set period size: {}", snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }
    buffer_frames = std::max<snd_pcm_uframes_t>(buffer_frames, periodFrames * 2);
    if ((err = snd_pcm_hw_params_set_buffer_size_near(handle, hw_params, &buffer_frames)) < 0) {
        LOG_ERROR("[Loopback] Cannot set buffer size: {}", snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }

    if ((err = snd_pcm_hw_params(handle, hw_params)) < 0) {
        LOG_ERROR("[Loopback] Cannot apply hardware parameters: {}", snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }

    snd_pcm_hw_params_get_period_size(hw_params, &periodFrames, nullptr);
    snd_pcm_hw_params_get_buffer_size(hw_params, &buffer_frames);

    if ((err = snd_pcm_prepare(handle)) < 0) {
        LOG_ERROR("[Loopback] Cannot prepare capture device: {}", snd_strerror(err));
        snd_pcm_close(handle);
        return nullptr;
    }

    snd_pcm_nonblock(handle, 1);

    LOG_INFO(
        "[Loopback] Capture device {} configured ({} Hz, {} ch, period {} frames, buffer {} "
        "frames)",
        device, rate_near, channels, periodFrames, buffer_frames);
    return handle;
}

void loopbackCaptureThread(const std::string& device, snd_pcm_format_t format, unsigned int rate,
                           unsigned int channels, snd_pcm_uframes_t periodFrames,
                           const LoopbackCaptureDependencies& deps) {
    if (channels != static_cast<unsigned int>(DaemonConstants::CHANNELS)) {
        LOG_ERROR("[Loopback] Unsupported channel count {} (expected {})", channels,
                  DaemonConstants::CHANNELS);
        return;
    }
    if (!deps.handleMutex || !deps.handle || !deps.captureRunning || !deps.captureReady ||
        !deps.running) {
        LOG_ERROR("[Loopback] Capture dependencies not configured");
        return;
    }

    snd_pcm_uframes_t negotiated_period = periodFrames;
    snd_pcm_t* handle = openLoopbackCapture(device, format, rate, channels, negotiated_period);
    {
        std::lock_guard<std::mutex> lock(*deps.handleMutex);
        *deps.handle = handle;
    }

    if (!handle) {
        return;
    }

    deps.captureRunning->store(true, std::memory_order_release);
    deps.captureReady->store(true, std::memory_order_release);

    std::vector<int16_t> buffer16;
    std::vector<int32_t> buffer32;
    std::vector<uint8_t> buffer24;
    std::vector<float> floatBuffer;

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

    while (deps.running->load(std::memory_order_acquire)) {
        if (deps.playbackBuffer && deps.currentOutputRate) {
            deps.playbackBuffer->throttleProducerIfFull(*deps.running, [&]() {
                return deps.currentOutputRate->load(std::memory_order_acquire);
            });
        }

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

        snd_pcm_sframes_t frames = snd_pcm_readi(handle, rawBuffer, negotiated_period);
        if (frames == -EAGAIN) {
            waitForCaptureReady(handle);
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
            waitForCaptureReady(handle);
            continue;
        }

        if (!convertPcmToFloat(rawBuffer, format, static_cast<size_t>(frames), channels,
                               floatBuffer)) {
            LOG_ERROR("[Loopback] Unsupported format during conversion");
            break;
        }

        if (deps.audioPipeline && *deps.audioPipeline) {
            (*deps.audioPipeline)->process(floatBuffer.data(), static_cast<uint32_t>(frames));
        }
    }

    snd_pcm_drop(handle);
    snd_pcm_close(handle);
    {
        std::lock_guard<std::mutex> lock(*deps.handleMutex);
        *deps.handle = nullptr;
    }
    deps.captureRunning->store(false, std::memory_order_release);
    deps.captureReady->store(false, std::memory_order_release);
    LOG_INFO("[Loopback] Capture thread terminated");
}

}  // namespace daemon_input
