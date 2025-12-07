#include "pcm_header.h"

#include <cstring>

namespace {

constexpr uint32_t BASE_RATES[] = {44100, 48000};
constexpr uint32_t MULTIPLIERS[] = {1, 2, 4, 8, 16};
constexpr uint16_t REQUIRED_CHANNELS = 2;

bool hasValidMagic(const PcmHeader &h) {
    static constexpr char MAGIC[4] = {'P', 'C', 'M', 'A'};
    return std::memcmp(h.magic, MAGIC, 4) == 0;
}

bool isSupportedFormat(uint16_t format) {
    switch (format) {
    case 1:
    case 2:
    case 4:
        return true;
    default:
        return false;
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

HeaderValidationResult validateHeader(const PcmHeader &header) {
    if (!hasValidMagic(header)) {
        return {false, "magic != PCMA"};
    }
    if (header.version != 1) {
        return {false, "version != 1"};
    }
    if (!isSupportedRate(header.sample_rate)) {
        return {false, "sample_rate unsupported"};
    }
    if (header.channels != REQUIRED_CHANNELS) {
        return {false, "channels unsupported"};
    }
    if (!isSupportedFormat(header.format)) {
        return {false, "unsupported format"};
    }

    return {true, ""};
}
