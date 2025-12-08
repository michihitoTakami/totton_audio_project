#pragma once

#include <alsa/asoundlib.h>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

class AlsaCapture {
   public:
    enum class SampleFormat {
        S16_LE,
        S24_3LE,
        S32_LE,
    };

    struct Config {
        std::string deviceName{"hw:0,0"};
        unsigned int sampleRate{48000};
        unsigned int channels{2};
        SampleFormat format{SampleFormat::S16_LE};
        snd_pcm_uframes_t periodFrames{4096};
    };

    AlsaCapture();
    ~AlsaCapture();

    bool open(const Config &config);
    bool start();
    /**
     * 読み取り結果: 正常時は読み取ったバイト数、0 は無音（タイムアウト等）、負数はエラー。
     */
    int read(std::vector<std::uint8_t> &buffer);
    void stop();
    void close();

    bool isOpen() const;
    std::optional<unsigned int> currentSampleRate() const;
    std::optional<unsigned int> currentChannels() const;
    std::optional<SampleFormat> currentFormat() const;

    // Exposed for unit tests and CLI validation
    static snd_pcm_format_t toAlsaFormat(SampleFormat format);
    static std::optional<SampleFormat> fromAlsaFormat(snd_pcm_format_t format);
    static std::size_t bytesPerFrame(const Config &config);
    static std::optional<SampleFormat> selectSupportedFormat(
        SampleFormat requested, const std::function<bool(SampleFormat)> &isSupported);

   private:
    Config config_{};
    snd_pcm_t *handle_{nullptr};
    std::size_t frameBytes_{0};
};
