#include "AlsaCapture.h"

#include "logging.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace {

void logAlsaError(const std::string &prefix, int err) {
    logError(prefix + ": " + snd_strerror(err));
}

}  // namespace

AlsaCapture::AlsaCapture() = default;

AlsaCapture::~AlsaCapture() {
    close();
}

bool AlsaCapture::open(const Config &config) {
    close();

    config_ = config;

    auto fail = [&](const std::string &message, int err) {
        logAlsaError(message, err);
        close();
        return false;
    };

    int rc = snd_pcm_open(&handle_, config_.deviceName.c_str(), SND_PCM_STREAM_CAPTURE, 0);
    if (rc < 0) {
        logAlsaError("[AlsaCapture] snd_pcm_open failed", rc);
        handle_ = nullptr;
        return false;
    }

    snd_pcm_hw_params_t *hwParams = nullptr;
    snd_pcm_hw_params_alloca(&hwParams);
    snd_pcm_hw_params_any(handle_, hwParams);

    rc = snd_pcm_hw_params_set_access(handle_, hwParams, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (rc < 0) {
        return fail("[AlsaCapture] set_access failed", rc);
    }

    auto selectedFormat = selectSupportedFormat(config_.format, [&](SampleFormat fmt) {
        const snd_pcm_format_t alsaFmt = toAlsaFormat(fmt);
        return snd_pcm_hw_params_test_format(handle_, hwParams, alsaFmt) == 0;
    });
    if (!selectedFormat) {
        logError("[AlsaCapture] No supported PCM format found for device");
        close();
        return false;
    }
    if (*selectedFormat != config_.format) {
        logInfo("[AlsaCapture] Requested format " +
                std::string(snd_pcm_format_name(toAlsaFormat(config_.format))) +
                " not supported, falling back to " +
                std::string(snd_pcm_format_name(toAlsaFormat(*selectedFormat))));
        config_.format = *selectedFormat;
    }

    const snd_pcm_format_t alsaFormat = toAlsaFormat(config_.format);
    rc = snd_pcm_hw_params_set_format(handle_, hwParams, alsaFormat);
    if (rc < 0) {
        return fail("[AlsaCapture] set_format failed", rc);
    }

    rc = snd_pcm_hw_params_set_channels(handle_, hwParams, config_.channels);
    if (rc < 0) {
        return fail("[AlsaCapture] set_channels failed", rc);
    }

    unsigned int rate = config_.sampleRate;
    rc = snd_pcm_hw_params_set_rate_near(handle_, hwParams, &rate, nullptr);
    if (rc < 0) {
        return fail("[AlsaCapture] set_rate_near failed", rc);
    }
    if (rate != config_.sampleRate) {
        logWarn("[AlsaCapture] Requested rate " + std::to_string(config_.sampleRate) +
                " differs from configured rate " + std::to_string(rate) + " (continuing)");
        config_.sampleRate = rate;
    }

    snd_pcm_uframes_t period = config_.periodFrames;
    rc = snd_pcm_hw_params_set_period_size_near(handle_, hwParams, &period, nullptr);
    if (rc < 0) {
        return fail("[AlsaCapture] set_period_size failed", rc);
    }

    rc = snd_pcm_hw_params(handle_, hwParams);
    if (rc < 0) {
        return fail("[AlsaCapture] apply hw_params failed", rc);
    }

    frameBytes_ = bytesPerFrame(config_);

    logInfo("[AlsaCapture] opened device=" + config_.deviceName + " rate=" +
            std::to_string(config_.sampleRate) + " ch=" + std::to_string(config_.channels) +
            " fmt=" + std::string(snd_pcm_format_name(alsaFormat)) +
            " period_frames=" + std::to_string(period));
    return true;
}

bool AlsaCapture::start() {
    if (!handle_) {
        logWarn("[AlsaCapture] start requested without open");
        return false;
    }
    int rc = snd_pcm_prepare(handle_);
    if (rc < 0) {
        logAlsaError("[AlsaCapture] snd_pcm_prepare failed", rc);
        return false;
    }
    rc = snd_pcm_start(handle_);
    if (rc < 0) {
        logAlsaError("[AlsaCapture] snd_pcm_start failed", rc);
        return false;
    }
    return true;
}

int AlsaCapture::read(std::vector<std::uint8_t> &buffer) {
    if (!handle_) {
        logWarn("[AlsaCapture] read called before open");
        return -1;
    }

    const std::size_t bytesPerPeriod = static_cast<std::size_t>(config_.periodFrames) * frameBytes_;
    if (buffer.size() < bytesPerPeriod) {
        buffer.resize(bytesPerPeriod);
    }

    const snd_pcm_sframes_t frames = snd_pcm_readi(handle_, buffer.data(), config_.periodFrames);
    if (frames == -EPIPE) {
        logWarn("[AlsaCapture] XRUN detected, recovering...");
        int rc = snd_pcm_prepare(handle_);
        if (rc < 0) {
            logAlsaError("[AlsaCapture] snd_pcm_prepare failed after XRUN", rc);
            return -1;
        }
        return -EPIPE;
    }
    if (frames < 0) {
        logAlsaError("[AlsaCapture] snd_pcm_readi failed", static_cast<int>(frames));
        return static_cast<int>(frames);
    }
    return static_cast<int>(frames * frameBytes_);
}

void AlsaCapture::stop() {
    if (handle_) {
        snd_pcm_drop(handle_);
        snd_pcm_drain(handle_);
        logInfo("[AlsaCapture] stopped");
    }
}

void AlsaCapture::close() {
    if (handle_) {
        snd_pcm_close(handle_);
        handle_ = nullptr;
        logInfo("[AlsaCapture] closed");
    }
}

bool AlsaCapture::isOpen() const {
    return handle_ != nullptr;
}

std::optional<unsigned int> AlsaCapture::currentSampleRate() const {
    if (!handle_) {
        return std::nullopt;
    }
    snd_pcm_hw_params_t *params = nullptr;
    snd_pcm_hw_params_alloca(&params);
    if (snd_pcm_hw_params_current(handle_, params) < 0) {
        return std::nullopt;
    }
    unsigned int rate = 0;
    if (snd_pcm_hw_params_get_rate(params, &rate, nullptr) < 0) {
        return std::nullopt;
    }
    return rate;
}

std::optional<unsigned int> AlsaCapture::currentChannels() const {
    if (!handle_) {
        return std::nullopt;
    }
    snd_pcm_hw_params_t *params = nullptr;
    snd_pcm_hw_params_alloca(&params);
    if (snd_pcm_hw_params_current(handle_, params) < 0) {
        return std::nullopt;
    }
    unsigned int channels = 0;
    if (snd_pcm_hw_params_get_channels(params, &channels) < 0) {
        return std::nullopt;
    }
    return channels;
}

std::optional<AlsaCapture::SampleFormat> AlsaCapture::currentFormat() const {
    if (!handle_) {
        return std::nullopt;
    }
    snd_pcm_hw_params_t *params = nullptr;
    snd_pcm_hw_params_alloca(&params);
    if (snd_pcm_hw_params_current(handle_, params) < 0) {
        return std::nullopt;
    }
    snd_pcm_format_t fmt = SND_PCM_FORMAT_UNKNOWN;
    if (snd_pcm_hw_params_get_format(params, &fmt) < 0) {
        return std::nullopt;
    }
    return fromAlsaFormat(fmt);
}

snd_pcm_format_t AlsaCapture::toAlsaFormat(SampleFormat format) {
    switch (format) {
    case SampleFormat::S16_LE:
        return SND_PCM_FORMAT_S16_LE;
    case SampleFormat::S24_3LE:
        return SND_PCM_FORMAT_S24_3LE;
    case SampleFormat::S32_LE:
        return SND_PCM_FORMAT_S32_LE;
    }
    return SND_PCM_FORMAT_UNKNOWN;
}

std::optional<AlsaCapture::SampleFormat> AlsaCapture::fromAlsaFormat(snd_pcm_format_t format) {
    switch (format) {
    case SND_PCM_FORMAT_S16_LE:
        return SampleFormat::S16_LE;
    case SND_PCM_FORMAT_S24_3LE:
        return SampleFormat::S24_3LE;
    case SND_PCM_FORMAT_S32_LE:
        return SampleFormat::S32_LE;
    default:
        return std::nullopt;
    }
}

std::size_t AlsaCapture::bytesPerFrame(const Config &config) {
    switch (config.format) {
    case SampleFormat::S16_LE:
        return static_cast<std::size_t>(config.channels) * 2U;
    case SampleFormat::S24_3LE:
        return static_cast<std::size_t>(config.channels) * 3U;
    case SampleFormat::S32_LE:
        return static_cast<std::size_t>(config.channels) * 4U;
    }
    throw std::runtime_error("Unsupported sample format");
}

std::optional<AlsaCapture::SampleFormat> AlsaCapture::selectSupportedFormat(
    SampleFormat requested, const std::function<bool(SampleFormat)> &isSupported) {
    std::vector<SampleFormat> candidates;
    candidates.push_back(requested);
    for (auto fmt : {SampleFormat::S32_LE, SampleFormat::S24_3LE, SampleFormat::S16_LE}) {
        if (std::find(candidates.begin(), candidates.end(), fmt) == candidates.end()) {
            candidates.push_back(fmt);
        }
    }

    for (auto fmt : candidates) {
        if (isSupported(fmt)) {
            return fmt;
        }
    }
    return std::nullopt;
}
