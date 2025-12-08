#include "alsa_playback.h"

#include "logging.h"
#include "status_tracker.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>

namespace {

constexpr snd_pcm_uframes_t DEFAULT_PERIOD_FRAMES = 512;
constexpr snd_pcm_uframes_t DEFAULT_BUFFER_FRAMES = DEFAULT_PERIOD_FRAMES * 4;
constexpr uint32_t BASE_RATES[] = {44100, 48000};
constexpr uint32_t MULTIPLIERS[] = {1, 2, 4, 8, 16};

snd_pcm_format_t toPcmFormat(uint16_t format) {
    switch (format) {
    case 1:
        return SND_PCM_FORMAT_S16_LE;
    case 2:
        return SND_PCM_FORMAT_S24_3LE;
    case 4:
        return SND_PCM_FORMAT_S32_LE;
    default:
        return SND_PCM_FORMAT_UNKNOWN;
    }
}

std::string formatName(snd_pcm_format_t fmt) {
    switch (fmt) {
    case SND_PCM_FORMAT_S16_LE:
        return "S16_LE";
    case SND_PCM_FORMAT_S24_3LE:
        return "S24_3LE";
    case SND_PCM_FORMAT_S32_LE:
        return "S32_LE";
    default:
        return "UNKNOWN";
    }
}

bool isSupportedRate(uint32_t rate) {
    for (auto base : BASE_RATES) {
        for (auto mul : MULTIPLIERS) {
            if (base * mul == rate) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

AlsaPlayback::AlsaPlayback(std::string device) : device_(std::move(device)) {}

AlsaPlayback::~AlsaPlayback() {
    close();
}

struct HwParamsDeleter {
    void operator()(snd_pcm_hw_params_t *p) const {
        if (p) {
            snd_pcm_hw_params_free(p);
        }
    }
};

bool AlsaPlayback::configureHardware(uint32_t sampleRate, uint16_t channels,
                                     snd_pcm_format_t format) {
    std::unique_ptr<snd_pcm_hw_params_t, HwParamsDeleter> params;
    snd_pcm_hw_params_t *raw = nullptr;
    snd_pcm_hw_params_malloc(&raw);
    params.reset(raw);
    if (!params) {
        logError("[AlsaPlayback] failed to alloc hw_params");
        return false;
    }

    if (snd_pcm_hw_params_any(handle_, params.get()) < 0) {
        logError("[AlsaPlayback] snd_pcm_hw_params_any failed");
        return false;
    }

    if (snd_pcm_hw_params_set_access(handle_, params.get(), SND_PCM_ACCESS_RW_INTERLEAVED) < 0) {
        logError("[AlsaPlayback] failed to set access");
        return false;
    }

    if (snd_pcm_hw_params_set_format(handle_, params.get(), format) < 0) {
        logError("[AlsaPlayback] failed to set format");
        return false;
    }

    if (snd_pcm_hw_params_set_channels(handle_, params.get(), channels) < 0) {
        logError("[AlsaPlayback] failed to set channels=" + std::to_string(channels));
        return false;
    }

    unsigned int rate = sampleRate;
    if (snd_pcm_hw_params_set_rate_near(handle_, params.get(), &rate, nullptr) < 0) {
        logError("[AlsaPlayback] failed to set rate=" + std::to_string(sampleRate));
        return false;
    }
    if (rate != sampleRate) {
        logError("[AlsaPlayback] rate mismatch (requested " + std::to_string(sampleRate) +
                 ", got " + std::to_string(rate) + ")");
        return false;
    }

    snd_pcm_uframes_t period = DEFAULT_PERIOD_FRAMES;
    if (snd_pcm_hw_params_set_period_size_near(handle_, params.get(), &period, nullptr) < 0) {
        logError("[AlsaPlayback] failed to set period size");
        return false;
    }

    snd_pcm_uframes_t buffer = DEFAULT_BUFFER_FRAMES;
    if (snd_pcm_hw_params_set_buffer_size_near(handle_, params.get(), &buffer) < 0) {
        logError("[AlsaPlayback] failed to set buffer size");
        return false;
    }

    if (snd_pcm_hw_params(handle_, params.get()) < 0) {
        logError("[AlsaPlayback] snd_pcm_hw_params apply failed");
        return false;
    }

    snd_pcm_hw_params_get_period_size(params.get(), &period, nullptr);
    snd_pcm_hw_params_get_buffer_size(params.get(), &buffer);
    periodSize_ = period;
    bufferSize_ = buffer;
    return true;
}

bool AlsaPlayback::validateCapabilities(uint32_t sampleRate, uint16_t channels,
                                        snd_pcm_format_t format) {
    std::unique_ptr<snd_pcm_hw_params_t, HwParamsDeleter> params;
    snd_pcm_hw_params_t *raw = nullptr;
    snd_pcm_hw_params_malloc(&raw);
    params.reset(raw);
    if (!params) {
        logError("[AlsaPlayback] failed to alloc hw_params for validation");
        return false;
    }

    if (snd_pcm_hw_params_any(handle_, params.get()) < 0) {
        logError("[AlsaPlayback] snd_pcm_hw_params_any failed during validation");
        return false;
    }

    if (snd_pcm_hw_params_test_access(handle_, params.get(), SND_PCM_ACCESS_RW_INTERLEAVED) < 0) {
        logError("[AlsaPlayback] device does not support interleaved access");
        return false;
    }
    if (snd_pcm_hw_params_test_format(handle_, params.get(), format) < 0) {
        logError("[AlsaPlayback] device does not support format " + formatName(format));
        return false;
    }
    if (snd_pcm_hw_params_test_channels(handle_, params.get(), channels) < 0) {
        logError("[AlsaPlayback] device does not support channels=" + std::to_string(channels));
        return false;
    }
    if (snd_pcm_hw_params_test_rate(handle_, params.get(), sampleRate, 0) < 0) {
        logError("[AlsaPlayback] device does not support rate=" + std::to_string(sampleRate));
        return false;
    }

    logInfo("[AlsaPlayback] capability check OK (rate=" + std::to_string(sampleRate) +
            ", channels=" + std::to_string(channels) + ", format=" + formatName(format) + ")");
    return true;
}

bool AlsaPlayback::open(uint32_t sampleRate, uint16_t channels, uint16_t format) {
    if (!isSupportedRate(sampleRate) || channels != 2) {
        logError("[AlsaPlayback] unsupported params (rate=" + std::to_string(sampleRate) +
                 ", channels=" + std::to_string(channels) +
                 "). Supported rates: 44.1k/48k * {1,2,4,8,16}, channels=2");
        return false;
    }

    snd_pcm_format_t pcmFormat = toPcmFormat(format);
    if (pcmFormat == SND_PCM_FORMAT_UNKNOWN) {
        logError("[AlsaPlayback] unsupported format=" + std::to_string(format) +
                 " (supported: 1=S16_LE, 2=S24_3LE, 4=S32_LE)");
        return false;
    }

    if (handle_) {
        close();
    }

    int rc = snd_pcm_open(&handle_, device_.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (rc < 0) {
        logError("[AlsaPlayback] failed to open device " + device_ + ": " + snd_strerror(rc) +
                 " (check available devices with `aplay -L`)");
        handle_ = nullptr;
        return false;
    }

    if (!validateCapabilities(sampleRate, channels, pcmFormat)) {
        close();
        return false;
    }

    if (!configureHardware(sampleRate, channels, pcmFormat)) {
        close();
        return false;
    }

    sampleRate_ = sampleRate;
    channels_ = channels;
    pcmFormat_ = pcmFormat;

    logInfo("[AlsaPlayback] opened " + device_ + " rate=" + std::to_string(sampleRate_) +
            " channels=" + std::to_string(channels_) + " format=" + formatName(pcmFormat_) +
            " period=" + std::to_string(periodSize_) + " buffer=" + std::to_string(bufferSize_));
    return true;
}

bool AlsaPlayback::recoverFromXrun() {
    int rc = snd_pcm_prepare(handle_);
    if (rc < 0) {
        logError(std::string("[AlsaPlayback] XRUN recover failed: ") + snd_strerror(rc));
        return false;
    }
    logWarn("[AlsaPlayback] XRUN recovered with snd_pcm_prepare()");
    return true;
}

bool AlsaPlayback::write(const void *data, std::size_t frames) {
    if (!handle_) {
        logError("[AlsaPlayback] write called before open()");
        return false;
    }
    if (frames == 0) {
        return true;
    }

    const std::uint8_t *ptr = static_cast<const std::uint8_t *>(data);
    std::size_t framesLeft = frames;
    const std::size_t frameBytes =
        snd_pcm_frames_to_bytes(handle_, static_cast<snd_pcm_uframes_t>(1));

    while (framesLeft > 0) {
        snd_pcm_sframes_t written = snd_pcm_writei(handle_, ptr, framesLeft);
        if (written == -EPIPE) {
            logWarn("[AlsaPlayback] XRUN detected (EPIPE)");
            if (statusTracker_) {
                statusTracker_->incrementXrun();
            }
            if (!recoverFromXrun()) {
                return false;
            }
            continue;  // retry after recover
        }
        if (written == -EINTR) {
            continue;  // interrupted; retry
        }
        if (written == -EAGAIN) {
            // Non-blocking edge; yield briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (written < 0) {
            logError(std::string("[AlsaPlayback] write failed: ") + snd_strerror(written));
            return false;
        }

        if (static_cast<std::size_t>(written) < framesLeft) {
            logWarn("[AlsaPlayback] partial write: " + std::to_string(written) + "/" +
                    std::to_string(framesLeft) + " frames; retrying remainder");
        }

        const std::size_t bytesWritten = static_cast<std::size_t>(written) * frameBytes;
        framesLeft -= static_cast<std::size_t>(written);
        ptr += bytesWritten;
    }

    return true;
}

void AlsaPlayback::close() {
    if (!handle_) {
        return;
    }

    snd_pcm_drop(handle_);
    snd_pcm_close(handle_);
    handle_ = nullptr;
    pcmFormat_ = SND_PCM_FORMAT_UNKNOWN;
    sampleRate_ = 0;
    channels_ = 0;
    periodSize_ = 0;
    bufferSize_ = 0;

    logInfo("[AlsaPlayback] closed " + device_);
}
