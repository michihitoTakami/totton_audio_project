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

constexpr std::size_t kPcmHeaderSize = 16;  // 4 + 4 + 4 + 2 + 2

inline std::array<std::uint8_t, kPcmHeaderSize> packPcmHeader(const PcmHeader &header) {
    std::array<std::uint8_t, kPcmHeaderSize> buffer{};
    buffer[0] = static_cast<std::uint8_t>(header.magic[0]);
    buffer[1] = static_cast<std::uint8_t>(header.magic[1]);
    buffer[2] = static_cast<std::uint8_t>(header.magic[2]);
    buffer[3] = static_cast<std::uint8_t>(header.magic[3]);

    auto writeU32 = [&buffer](std::size_t offset, std::uint32_t value) {
        buffer[offset + 0] = static_cast<std::uint8_t>(value & 0xFF);
        buffer[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
        buffer[offset + 2] = static_cast<std::uint8_t>((value >> 16) & 0xFF);
        buffer[offset + 3] = static_cast<std::uint8_t>((value >> 24) & 0xFF);
    };
    auto writeU16 = [&buffer](std::size_t offset, std::uint16_t value) {
        buffer[offset + 0] = static_cast<std::uint8_t>(value & 0xFF);
        buffer[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
    };

    writeU32(4, header.version);
    writeU32(8, header.sampleRate);
    writeU16(12, header.channels);
    writeU16(14, header.format);
    return buffer;
}

inline std::string headerToString(const PcmHeader &header) {
    auto bytes = packPcmHeader(header);
    return std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}
