#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// ALSA 再生デバイス操作の雛形。実装は後続タスクで追加する。
class AlsaPlayback {
public:
    explicit AlsaPlayback(std::string device);
    ~AlsaPlayback();

    bool open(uint32_t sampleRate, uint16_t channels, uint16_t format);
    bool write(const void *data, std::size_t frames);
    void close();

    const std::string &device() const { return device_; }

private:
    std::string device_;
};

