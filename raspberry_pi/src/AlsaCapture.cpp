#include "AlsaCapture.h"

#include <iostream>
#include <stdexcept>

namespace {

void logError(const std::string &prefix, int err)
{
    std::clog << prefix << ": " << snd_strerror(err) << std::endl;
}

} // namespace

AlsaCapture::AlsaCapture() = default;

AlsaCapture::~AlsaCapture()
{
    close();
}

bool AlsaCapture::open(const Config &config)
{
    close();

    config_ = config;
    frameBytes_ = bytesPerFrame();

    int rc = snd_pcm_open(&handle_, config_.deviceName.c_str(),
        SND_PCM_STREAM_CAPTURE, 0);
    if(rc < 0) {
        logError("[AlsaCapture] snd_pcm_open failed", rc);
        handle_ = nullptr;
        return false;
    }

    snd_pcm_hw_params_t *hwParams = nullptr;
    snd_pcm_hw_params_alloca(&hwParams);
    snd_pcm_hw_params_any(handle_, hwParams);

    rc = snd_pcm_hw_params_set_access(
        handle_, hwParams, SND_PCM_ACCESS_RW_INTERLEAVED);
    if(rc < 0) {
        logError("[AlsaCapture] set_access failed", rc);
        return false;
    }

    const snd_pcm_format_t alsaFormat = toAlsaFormat(config_.format);
    rc = snd_pcm_hw_params_set_format(handle_, hwParams, alsaFormat);
    if(rc < 0) {
        logError("[AlsaCapture] set_format failed", rc);
        return false;
    }

    rc = snd_pcm_hw_params_set_channels(handle_, hwParams, config_.channels);
    if(rc < 0) {
        logError("[AlsaCapture] set_channels failed", rc);
        return false;
    }

    unsigned int rate = config_.sampleRate;
    rc = snd_pcm_hw_params_set_rate_near(handle_, hwParams, &rate, nullptr);
    if(rc < 0 || rate != config_.sampleRate) {
        logError("[AlsaCapture] set_rate_near failed or mismatched rate", rc);
        return false;
    }

    snd_pcm_uframes_t period = config_.periodFrames;
    rc = snd_pcm_hw_params_set_period_size_near(
        handle_, hwParams, &period, nullptr);
    if(rc < 0) {
        logError("[AlsaCapture] set_period_size failed", rc);
        return false;
    }

    rc = snd_pcm_hw_params(handle_, hwParams);
    if(rc < 0) {
        logError("[AlsaCapture] apply hw_params failed", rc);
        return false;
    }

    std::clog << "[AlsaCapture] opened device=" << config_.deviceName
              << " rate=" << config_.sampleRate
              << " ch=" << config_.channels
              << " fmt=" << snd_pcm_format_name(alsaFormat)
              << " period_frames=" << period << std::endl;
    return true;
}

bool AlsaCapture::start()
{
    if(!handle_) {
        std::clog << "[AlsaCapture] start requested without open" << std::endl;
        return false;
    }
    int rc = snd_pcm_prepare(handle_);
    if(rc < 0) {
        logError("[AlsaCapture] snd_pcm_prepare failed", rc);
        return false;
    }
    rc = snd_pcm_start(handle_);
    if(rc < 0) {
        logError("[AlsaCapture] snd_pcm_start failed", rc);
        return false;
    }
    return true;
}

int AlsaCapture::read(std::vector<std::uint8_t> &buffer)
{
    if(!handle_) {
        std::clog << "[AlsaCapture] read called before open" << std::endl;
        return -1;
    }

    const std::size_t bytesPerPeriod = static_cast<std::size_t>(
        config_.periodFrames) * frameBytes_;
    if(buffer.size() < bytesPerPeriod) {
        buffer.resize(bytesPerPeriod);
    }

    const snd_pcm_sframes_t frames = snd_pcm_readi(
        handle_, buffer.data(), config_.periodFrames);
    if(frames == -EPIPE) {
        std::clog << "[AlsaCapture] XRUN detected, recovering..." << std::endl;
        int rc = snd_pcm_prepare(handle_);
        if(rc < 0) {
            logError("[AlsaCapture] snd_pcm_prepare failed after XRUN", rc);
            return -1;
        }
        return -EPIPE;
    }
    if(frames < 0) {
        logError("[AlsaCapture] snd_pcm_readi failed", static_cast<int>(frames));
        return static_cast<int>(frames);
    }
    return static_cast<int>(frames * frameBytes_);
}

void AlsaCapture::stop()
{
    if(handle_) {
        snd_pcm_drop(handle_);
        snd_pcm_drain(handle_);
        std::clog << "[AlsaCapture] stopped" << std::endl;
    }
}

void AlsaCapture::close()
{
    if(handle_) {
        snd_pcm_close(handle_);
        handle_ = nullptr;
        std::clog << "[AlsaCapture] closed" << std::endl;
    }
}

bool AlsaCapture::isOpen() const
{
    return handle_ != nullptr;
}

snd_pcm_format_t AlsaCapture::toAlsaFormat(SampleFormat format) const
{
    switch(format) {
    case SampleFormat::S16_LE:
        return SND_PCM_FORMAT_S16_LE;
    case SampleFormat::S24_3LE:
        return SND_PCM_FORMAT_S24_3LE;
    case SampleFormat::S32_LE:
        return SND_PCM_FORMAT_S32_LE;
    }
    return SND_PCM_FORMAT_UNKNOWN;
}

std::size_t AlsaCapture::bytesPerFrame() const
{
    switch(config_.format) {
    case SampleFormat::S16_LE:
        return static_cast<std::size_t>(config_.channels) * 2U;
    case SampleFormat::S24_3LE:
        return static_cast<std::size_t>(config_.channels) * 3U;
    case SampleFormat::S32_LE:
        return static_cast<std::size_t>(config_.channels) * 4U;
    }
    throw std::runtime_error("Unsupported sample format");
}

