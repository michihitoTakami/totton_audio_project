#pragma once

#include <cstdint>
#include <string>

#pragma pack(push, 1)
struct PcmHeader {
    char magic[4];     // "PCMA"
    uint32_t version;  // currently 1
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t format;  // 1=S16_LE, 2=S24_3LE, 3=S24_LE, 4=S32_LE (想定)
};
#pragma pack(pop)

struct HeaderValidationResult {
    bool ok{false};
    std::string reason;
};

HeaderValidationResult validateHeader(const PcmHeader &header);
