#include "core/base64.h"

#include <stdexcept>

namespace Base64 {

// RFC 4648 standard Base64 alphabet
static constexpr char ENCODE_TABLE[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Decode table: maps ASCII value to 6-bit value, 255 = invalid
// clang-format off
static constexpr uint8_t DECODE_TABLE[] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  // 0-15
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  // 16-31
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  62, 255, 255, 255,  63,  // 32-47 (+/)
     52,  53,  54,  55,  56,  57,  58,  59,  60,  61, 255, 255, 255, 255, 255, 255,  // 48-63 (0-9)
    255,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  // 64-79 (A-O)
     15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 255, 255, 255, 255, 255,  // 80-95 (P-Z)
    255,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  // 96-111 (a-o)
     41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51, 255, 255, 255, 255, 255,  // 112-127 (p-z)
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  // 128-143
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  // 144-159
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  // 160-175
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  // 176-191
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  // 192-207
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  // 208-223
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  // 224-239
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255   // 240-255
};
// clang-format on

size_t encodedSize(size_t inputLength) {
    // Base64 encodes 3 bytes to 4 chars, with padding
    return ((inputLength + 2) / 3) * 4;
}

size_t decodedSize(const std::string& encoded) {
    if (encoded.empty()) {
        return 0;
    }

    size_t len = encoded.length();

    // Must be multiple of 4
    if (len % 4 != 0) {
        return 0;
    }

    // Count padding
    size_t padding = 0;
    if (len >= 1 && encoded[len - 1] == '=')
        padding++;
    if (len >= 2 && encoded[len - 2] == '=')
        padding++;

    // Each 4 chars decode to 3 bytes, minus padding
    return (len / 4) * 3 - padding;
}

std::string encode(const uint8_t* data, size_t length) {
    std::string result;
    result.reserve(encodedSize(length));

    size_t i = 0;

    // Process 3 bytes at a time
    while (i + 3 <= length) {
        uint32_t triple = (static_cast<uint32_t>(data[i]) << 16) |
                          (static_cast<uint32_t>(data[i + 1]) << 8) |
                          static_cast<uint32_t>(data[i + 2]);

        result += ENCODE_TABLE[(triple >> 18) & 0x3F];
        result += ENCODE_TABLE[(triple >> 12) & 0x3F];
        result += ENCODE_TABLE[(triple >> 6) & 0x3F];
        result += ENCODE_TABLE[triple & 0x3F];

        i += 3;
    }

    // Handle remaining bytes with padding
    size_t remaining = length - i;
    if (remaining == 1) {
        uint32_t value = static_cast<uint32_t>(data[i]) << 16;
        result += ENCODE_TABLE[(value >> 18) & 0x3F];
        result += ENCODE_TABLE[(value >> 12) & 0x3F];
        result += '=';
        result += '=';
    } else if (remaining == 2) {
        uint32_t value =
            (static_cast<uint32_t>(data[i]) << 16) | (static_cast<uint32_t>(data[i + 1]) << 8);
        result += ENCODE_TABLE[(value >> 18) & 0x3F];
        result += ENCODE_TABLE[(value >> 12) & 0x3F];
        result += ENCODE_TABLE[(value >> 6) & 0x3F];
        result += '=';
    }

    return result;
}

std::string encode(const std::vector<uint8_t>& data) {
    return encode(data.data(), data.size());
}

std::vector<uint8_t> decode(const std::string& encoded) {
    if (encoded.empty()) {
        return {};
    }

    size_t len = encoded.length();

    // Must be multiple of 4
    if (len % 4 != 0) {
        return {};
    }

    // Count padding
    size_t padding = 0;
    if (len >= 1 && encoded[len - 1] == '=')
        padding++;
    if (len >= 2 && encoded[len - 2] == '=')
        padding++;

    std::vector<uint8_t> result;
    result.reserve((len / 4) * 3 - padding);

    // Process 4 chars at a time
    for (size_t i = 0; i < len; i += 4) {
        uint8_t a = DECODE_TABLE[static_cast<uint8_t>(encoded[i])];
        uint8_t b = DECODE_TABLE[static_cast<uint8_t>(encoded[i + 1])];
        uint8_t c = encoded[i + 2] == '=' ? 0 : DECODE_TABLE[static_cast<uint8_t>(encoded[i + 2])];
        uint8_t d = encoded[i + 3] == '=' ? 0 : DECODE_TABLE[static_cast<uint8_t>(encoded[i + 3])];

        // Check for invalid characters (before padding check)
        if (a == 255 || b == 255) {
            return {};
        }
        if (encoded[i + 2] != '=' && c == 255) {
            return {};
        }
        if (encoded[i + 3] != '=' && d == 255) {
            return {};
        }

        uint32_t triple = (static_cast<uint32_t>(a) << 18) | (static_cast<uint32_t>(b) << 12) |
                          (static_cast<uint32_t>(c) << 6) | static_cast<uint32_t>(d);

        result.push_back(static_cast<uint8_t>((triple >> 16) & 0xFF));

        if (encoded[i + 2] != '=') {
            result.push_back(static_cast<uint8_t>((triple >> 8) & 0xFF));
        }

        if (encoded[i + 3] != '=') {
            result.push_back(static_cast<uint8_t>(triple & 0xFF));
        }
    }

    return result;
}

bool isValid(const std::string& encoded) {
    if (encoded.empty()) {
        return true;
    }

    size_t len = encoded.length();

    // Must be multiple of 4
    if (len % 4 != 0) {
        return false;
    }

    // Check all characters except last two (which may be padding)
    for (size_t i = 0; i < len - 2; ++i) {
        if (DECODE_TABLE[static_cast<uint8_t>(encoded[i])] == 255) {
            return false;
        }
    }

    // Check last two characters (may be padding)
    char c2 = encoded[len - 2];
    char c1 = encoded[len - 1];

    if (c1 == '=') {
        // Last char is padding
        if (c2 == '=') {
            // Both are padding - valid
        } else if (DECODE_TABLE[static_cast<uint8_t>(c2)] == 255) {
            return false;
        }
    } else {
        // No padding
        if (DECODE_TABLE[static_cast<uint8_t>(c1)] == 255 ||
            DECODE_TABLE[static_cast<uint8_t>(c2)] == 255) {
            return false;
        }
    }

    return true;
}

}  // namespace Base64
