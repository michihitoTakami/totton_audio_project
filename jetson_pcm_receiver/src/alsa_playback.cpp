#include "alsa_playback.h"

#include <iostream>
#include <utility>

AlsaPlayback::AlsaPlayback(std::string device) : device_(std::move(device)) {}

AlsaPlayback::~AlsaPlayback() = default;

bool AlsaPlayback::open(uint32_t sampleRate, uint16_t channels, uint16_t format) {
    std::cout << "[AlsaPlayback] 未実装: open device=" << device_
              << " rate=" << sampleRate
              << " channels=" << channels
              << " format=" << format << std::endl;
    return false;
}

bool AlsaPlayback::write(const void * /*data*/, std::size_t frames) {
    std::cout << "[AlsaPlayback] 未実装: write frames=" << frames << std::endl;
    return false;
}

void AlsaPlayback::close() {
    std::cout << "[AlsaPlayback] 未実装: close device=" << device_ << std::endl;
}

