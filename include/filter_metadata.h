#ifndef FILTER_METADATA_H
#define FILTER_METADATA_H

#include <array>
#include <cstddef>
#include <string_view>

struct FilterPreset {
    int inputSampleRate;
    int upsampleRatio;
    std::string_view path;
    std::string_view description;
    size_t taps;
};

inline constexpr FilterPreset FILTER_PRESET_44K = {
    44100, 16, "data/coefficients/filter_44k_16x_2m_min_phase.bin",
    "2M-tap min-phase FIR for 44.1kHz → 705.6kHz (16x)", 2'000'000};

inline constexpr FilterPreset FILTER_PRESET_48K = {
    48000, 16, "data/coefficients/filter_48k_2m_min_phase.bin",
    "2M-tap min-phase FIR for 48kHz → 768kHz (16x)", 2'000'000};

inline constexpr std::array<FilterPreset, 2> FILTER_PRESETS = {FILTER_PRESET_44K,
                                                               FILTER_PRESET_48K};

#endif  // FILTER_METADATA_H
