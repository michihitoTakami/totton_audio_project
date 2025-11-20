// Auto-generated filter coefficients
// GPU Audio Upsampler - Phase 1
// Generated: 2025-11-21T01:00:43.263084

#ifndef FILTER_COEFFICIENTS_H
#define FILTER_COEFFICIENTS_H

#include <cstddef>

constexpr size_t FILTER_TAPS = 131072;
constexpr int SAMPLE_RATE_INPUT = 44100;
constexpr int SAMPLE_RATE_OUTPUT = 705600;
constexpr int UPSAMPLE_RATIO = 16;

// Filter coefficients (float32)
// Note: For large arrays, consider loading from binary file instead
extern const float FILTER_COEFFICIENTS[FILTER_TAPS];

#endif // FILTER_COEFFICIENTS_H
