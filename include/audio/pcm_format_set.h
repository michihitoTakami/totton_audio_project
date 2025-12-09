/*
 * Shared PCM format constraints between pcm receiver bridge and main daemon.
 * Keeps the single source of truth for accepted sample rates / channels / formats.
 */
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace PcmFormatSet {

namespace detail {

template <typename T, std::size_t N>
constexpr bool contains(const std::array<T, N>& arr, T value) {
    for (auto v : arr) {
        if (v == value) {
            return true;
        }
    }
    return false;
}

template <std::size_t N1, std::size_t N2>
constexpr std::array<uint32_t, N1 + N2> concat(const std::array<uint32_t, N1>& a,
                                               const std::array<uint32_t, N2>& b) {
    std::array<uint32_t, N1 + N2> result{};
    for (std::size_t i = 0; i < N1; ++i) {
        result[i] = a[i];
    }
    for (std::size_t i = 0; i < N2; ++i) {
        result[N1 + i] = b[i];
    }
    return result;
}

}  // namespace detail

// Format codes
// 1 = S16_LE, 2 = S24_3LE, 4 = S32_LE
constexpr std::array<uint16_t, 3> kAllowedFormats = {1, 2, 4};
constexpr uint16_t kRequiredChannels = 2;

// Allowed input sample rates (base × {1,2,4,8,16})
constexpr std::array<uint32_t, 5> kRates44k = {44100, 88200, 176400, 352800, 705600};
constexpr std::array<uint32_t, 5> kRates48k = {48000, 96000, 192000, 384000, 768000};
constexpr auto kAllowedSampleRates = detail::concat(kRates44k, kRates48k);

constexpr bool isAllowedFormat(uint16_t format) {
    return detail::contains(kAllowedFormats, format);
}

constexpr bool isAllowedChannels(uint16_t channels) {
    return channels == kRequiredChannels;
}

constexpr bool isAllowedSampleRate(uint32_t rate) {
    return detail::contains(kAllowedSampleRates, rate);
}

constexpr bool is44kFamilyRate(uint32_t rate) {
    return detail::contains(kRates44k, rate);
}

constexpr bool is48kFamilyRate(uint32_t rate) {
    return detail::contains(kRates48k, rate);
}

inline std::vector<uint32_t> allowedSampleRatesVector() {
    return std::vector<uint32_t>(kAllowedSampleRates.begin(), kAllowedSampleRates.end());
}

inline std::vector<uint16_t> allowedFormatsVector() {
    return std::vector<uint16_t>(kAllowedFormats.begin(), kAllowedFormats.end());
}

inline std::string allowedSampleRatesString() {
    std::string result;
    for (std::size_t i = 0; i < kAllowedSampleRates.size(); ++i) {
        result += std::to_string(kAllowedSampleRates[i]);
        if (i + 1 < kAllowedSampleRates.size()) {
            result += ", ";
        }
    }
    return result;
}

}  // namespace PcmFormatSet
