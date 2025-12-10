// Read ALSA hw_params from /proc to detect runtime rate/format changes
#pragma once

#include "AlsaCapture.h"

#include <optional>
#include <string>

struct CaptureParams {
    unsigned int sampleRate{0};
    unsigned int channels{0};
    AlsaCapture::SampleFormat format{AlsaCapture::SampleFormat::S16_LE};
};

inline bool operator==(const CaptureParams &lhs, const CaptureParams &rhs) {
    return lhs.sampleRate == rhs.sampleRate && lhs.channels == rhs.channels &&
           lhs.format == rhs.format;
}

inline bool operator!=(const CaptureParams &lhs, const CaptureParams &rhs) {
    return !(lhs == rhs);
}

class HwParamsMonitor {
   public:
    HwParamsMonitor(std::string deviceName, std::string overridePath = "");

    std::optional<CaptureParams> readCurrent() const;
    std::string describe() const;

   private:
    std::string deviceName_;
    std::string pathOverride_;

    std::optional<std::string> resolveHwParamsPath() const;
    static std::optional<CaptureParams> parseHwParams(const std::string &content);
};
