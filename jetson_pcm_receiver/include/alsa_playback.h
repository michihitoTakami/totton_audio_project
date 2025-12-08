#pragma once

#include "xrun_detector.h"

#include <alsa/asoundlib.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

class StatusTracker;

// ALSA 再生デバイス操作。S16_LE/2ch/48k を初期ターゲットとし、
// 今後拡張しやすいようにフォーマット変換とパラメータ設定を分離する。
class AlsaPlayback {
   public:
    explicit AlsaPlayback(std::string device);
    virtual ~AlsaPlayback();

    virtual bool open(uint32_t sampleRate, uint16_t channels, uint16_t format);
    virtual bool write(const void *data, std::size_t frames);
    virtual void close();
    virtual bool wasXrunStorm() const;
    void setStatusTracker(StatusTracker *tracker) {
        statusTracker_ = tracker;
    }

    const std::string &device() const {
        return device_;
    }

   private:
    std::string device_;
    snd_pcm_t *handle_{nullptr};
    snd_pcm_format_t pcmFormat_{SND_PCM_FORMAT_UNKNOWN};
    uint32_t sampleRate_{0};
    uint16_t channels_{0};
    snd_pcm_uframes_t periodSize_{0};
    snd_pcm_uframes_t bufferSize_{0};
    StatusTracker *statusTracker_{nullptr};
    XrunDetector xrunDetector_{std::chrono::milliseconds(1000)};
    bool xrunStormDetected_{false};

    bool configureHardware(uint32_t sampleRate, uint16_t channels, snd_pcm_format_t format);
    bool validateCapabilities(uint32_t sampleRate, uint16_t channels, snd_pcm_format_t format);
    bool recoverFromXrun();
};
