#include "pcm_header.h"

#include <cstring>

namespace {

constexpr uint32_t MIN_RATE = 44100;
constexpr uint32_t MAX_RATE = 768000;
constexpr uint16_t MIN_CHANNELS = 1;
constexpr uint16_t MAX_CHANNELS = 8;

bool hasValidMagic(const PcmHeader &h) {
    static constexpr char MAGIC[4] = {'P', 'C', 'M', 'A'};
    return std::memcmp(h.magic, MAGIC, 4) == 0;
}

bool isSupportedFormat(uint16_t format) {
    // 現時点の想定: 1=S16_LE, 2=S24_3LE, 3=S24_LE, 4=S32_LE
    switch (format) {
    case 1:
    case 2:
    case 3:
    case 4:
        return true;
    default:
        return false;
    }
}

}  // namespace

HeaderValidationResult validateHeader(const PcmHeader &header) {
    if (!hasValidMagic(header)) {
        return {false, "magic != PCMA"};
    }
    if (header.version != 1) {
        return {false, "version != 1"};
    }
    if (header.sample_rate < MIN_RATE || header.sample_rate > MAX_RATE) {
        return {false, "sample_rate out of range"};
    }
    if (header.channels < MIN_CHANNELS || header.channels > MAX_CHANNELS) {
        return {false, "channels out of range"};
    }
    if (!isSupportedFormat(header.format)) {
        return {false, "unsupported format"};
    }

    return {true, ""};
}
