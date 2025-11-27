// Auto-generated multi-rate filter coefficients
// GPU Audio Upsampler - Multi-Rate Support
// Generated: 2025-11-27T23:46:33.888992

#ifndef FILTER_COEFFICIENTS_H
#define FILTER_COEFFICIENTS_H

#include <cstddef>
#include <cstdint>

// Multi-rate filter configurations
struct FilterConfig {
    const char* name;
    const char* filename;
    size_t taps;  // Actual tap count (matches .bin file length)
    int32_t input_rate;
    int32_t output_rate;
    int32_t ratio;
};

constexpr size_t FILTER_COUNT = 8;

constexpr FilterConfig FILTER_CONFIGS[FILTER_COUNT] = {
    {"44k_16x", "filter_44k_16x_2m_linear.bin", 640000, 44100, 705600, 16},
    {"44k_8x", "filter_44k_8x_2m_linear.bin", 640000, 88200, 705600, 8},
    {"44k_4x", "filter_44k_4x_2m_linear.bin", 640000, 176400, 705600, 4},
    {"44k_2x", "filter_44k_2x_2m_linear.bin", 640000, 352800, 705600, 2},
    {"48k_16x", "filter_48k_16x_2m_linear.bin", 640000, 48000, 768000, 16},
    {"48k_8x", "filter_48k_8x_2m_linear.bin", 640000, 96000, 768000, 8},
    {"48k_4x", "filter_48k_4x_2m_linear.bin", 640000, 192000, 768000, 4},
    {"48k_2x", "filter_48k_2x_2m_linear.bin", 640000, 384000, 768000, 2},
};

#endif  // FILTER_COEFFICIENTS_H
