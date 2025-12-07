#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

struct PcmHeader {
    char magic[4] = {'P', 'C', 'M', 'A'};
    std::uint32_t version{1};
    std::uint32_t sampleRate{48000};
    std::uint16_t channels{2};
    std::uint16_t format{1};  // 1=S16_LE, 2=S24_3LE, 4=S32_LE
};

constexpr std::size_t kPcmHeaderSize = sizeof(PcmHeader);

inline std::array<std::uint8_t, kPcmHeaderSize> packPcmHeader(const PcmHeader &header) {
    std::array<std::uint8_t, kPcmHeaderSize> buffer{};
    std::memcpy(buffer.data(), &header, kPcmHeaderSize);
    return buffer;
}

inline std::string headerToString(const PcmHeader &header) {
    auto bytes = packPcmHeader(header);
    return std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}
