#include "pcm_header.h"

#include "audio/pcm_format_set.h"

#include <cstring>

namespace {

bool hasValidMagic(const PcmHeader &h) {
    static constexpr char MAGIC[4] = {'P', 'C', 'M', 'A'};
    return std::memcmp(h.magic, MAGIC, 4) == 0;
}

bool isSupportedFormat(uint16_t format) {
    return PcmFormatSet::isAllowedFormat(format);
}

bool isSupportedRate(uint32_t rate) {
    return PcmFormatSet::isAllowedSampleRate(rate);
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
    if (!PcmFormatSet::isAllowedChannels(header.channels)) {
        return {false, "channels unsupported"};
    }
    if (!isSupportedFormat(header.format)) {
        return {false, "unsupported format"};
    }

    return {true, ""};
}
