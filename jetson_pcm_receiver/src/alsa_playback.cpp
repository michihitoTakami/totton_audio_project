#include "alsa_playback.h"

#include <iostream>
#include <memory>
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
        std::cerr << "[AlsaPlayback] failed to alloc hw_params" << std::endl;
        return false;
    }

    if (snd_pcm_hw_params_any(handle_, params.get()) < 0) {
        std::cerr << "[AlsaPlayback] snd_pcm_hw_params_any failed" << std::endl;
        return false;
    }

    if (snd_pcm_hw_params_set_access(handle_, params.get(), SND_PCM_ACCESS_RW_INTERLEAVED) < 0) {
        std::cerr << "[AlsaPlayback] failed to set access" << std::endl;
        return false;
    }

    if (snd_pcm_hw_params_set_format(handle_, params.get(), format) < 0) {
        std::cerr << "[AlsaPlayback] failed to set format" << std::endl;
        return false;
    }

    if (snd_pcm_hw_params_set_channels(handle_, params.get(), channels) < 0) {
        std::cerr << "[AlsaPlayback] failed to set channels=" << channels << std::endl;
        return false;
    }

    unsigned int rate = sampleRate;
    if (snd_pcm_hw_params_set_rate_near(handle_, params.get(), &rate, nullptr) < 0) {
        std::cerr << "[AlsaPlayback] failed to set rate=" << sampleRate << std::endl;
        return false;
    }
    if (rate != sampleRate) {
        std::cerr << "[AlsaPlayback] rate mismatch (requested " << sampleRate << ", got " << rate
                  << ")" << std::endl;
        return false;
    }

    snd_pcm_uframes_t period = DEFAULT_PERIOD_FRAMES;
    if (snd_pcm_hw_params_set_period_size_near(handle_, params.get(), &period, nullptr) < 0) {
        std::cerr << "[AlsaPlayback] failed to set period size" << std::endl;
        return false;
    }

    snd_pcm_uframes_t buffer = DEFAULT_BUFFER_FRAMES;
    if (snd_pcm_hw_params_set_buffer_size_near(handle_, params.get(), &buffer) < 0) {
        std::cerr << "[AlsaPlayback] failed to set buffer size" << std::endl;
        return false;
    }

    if (snd_pcm_hw_params(handle_, params.get()) < 0) {
        std::cerr << "[AlsaPlayback] snd_pcm_hw_params apply failed" << std::endl;
        return false;
    }

    snd_pcm_hw_params_get_period_size(params.get(), &period, nullptr);
    snd_pcm_hw_params_get_buffer_size(params.get(), &buffer);
    periodSize_ = period;
    bufferSize_ = buffer;
    return true;
}

bool AlsaPlayback::open(uint32_t sampleRate, uint16_t channels, uint16_t format) {
    if (!isSupportedRate(sampleRate) || channels != 2) {
        std::cerr << "[AlsaPlayback] unsupported params (rate=" << sampleRate
                  << ", channels=" << channels
                  << "). Supported rates: 44.1k/48k * {1,2,4,8,16}, channels=2" << std::endl;
        return false;
    }

    snd_pcm_format_t pcmFormat = toPcmFormat(format);
    if (pcmFormat == SND_PCM_FORMAT_UNKNOWN) {
        std::cerr << "[AlsaPlayback] unsupported format=" << format
                  << " (supported: 1=S16_LE, 2=S24_3LE, 4=S32_LE)" << std::endl;
        return false;
    }

    if (handle_) {
        close();
    }

    int rc = snd_pcm_open(&handle_, device_.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (rc < 0) {
        std::cerr << "[AlsaPlayback] failed to open device " << device_ << ": " << snd_strerror(rc)
                  << std::endl;
        handle_ = nullptr;
        return false;
    }

    if (!configureHardware(sampleRate, channels, pcmFormat)) {
        close();
        return false;
    }

    sampleRate_ = sampleRate;
    channels_ = channels;
    pcmFormat_ = pcmFormat;

    std::cout << "[AlsaPlayback] opened " << device_ << " rate=" << sampleRate_
              << " channels=" << channels_ << " format=S16_LE"
              << " period=" << periodSize_ << " buffer=" << bufferSize_ << std::endl;
    return true;
}

bool AlsaPlayback::recoverFromXrun() {
    int rc = snd_pcm_prepare(handle_);
    if (rc < 0) {
        std::cerr << "[AlsaPlayback] XRUN recover failed: " << snd_strerror(rc) << std::endl;
        return false;
    }
    std::cerr << "[AlsaPlayback] XRUN recovered with snd_pcm_prepare()" << std::endl;
    return true;
}

bool AlsaPlayback::write(const void *data, std::size_t frames) {
    if (!handle_) {
        std::cerr << "[AlsaPlayback] write called before open()" << std::endl;
        return false;
    }
    if (frames == 0) {
        return true;
    }

    snd_pcm_sframes_t written = snd_pcm_writei(handle_, data, frames);
    if (written == -EPIPE) {
        std::cerr << "[AlsaPlayback] XRUN detected (EPIPE)" << std::endl;
        if (!recoverFromXrun()) {
            return false;
        }
        written = snd_pcm_writei(handle_, data, frames);
    }

    if (written < 0) {
        std::cerr << "[AlsaPlayback] write failed: " << snd_strerror(written) << std::endl;
        return false;
    }

    if (static_cast<std::size_t>(written) != frames) {
        std::cerr << "[AlsaPlayback] partial write: " << written << "/" << frames << " frames"
                  << std::endl;
        return false;
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

    std::cout << "[AlsaPlayback] closed " << device_ << std::endl;
}
