#pragma once

#include <alsa/asoundlib.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

// ALSA 再生デバイス操作。S16_LE/2ch/48k を初期ターゲットとし、
// 今後拡張しやすいようにフォーマット変換とパラメータ設定を分離する。
class AlsaPlayback {
   public:
    explicit AlsaPlayback(std::string device);
    virtual ~AlsaPlayback();

    virtual bool open(uint32_t sampleRate, uint16_t channels, uint16_t format);
    virtual bool write(const void *data, std::size_t frames);
    virtual void close();
    std::size_t xrunCount() const {
        return xrunCount_.load(std::memory_order_relaxed);
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
    std::atomic_size_t xrunCount_{0};

    bool configureHardware(uint32_t sampleRate, uint16_t channels, snd_pcm_format_t format);
    bool recoverFromXrun();
};
