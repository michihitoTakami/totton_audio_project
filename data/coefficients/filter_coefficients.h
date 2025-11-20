// Auto-generated filter coefficients
// GPU Audio Upsampler - Phase 1
// Generated: 2025-11-21T01:28:16.915716

#ifndef FILTER_COEFFICIENTS_H
#define FILTER_COEFFICIENTS_H

#include <cstddef>

constexpr size_t FILTER_TAPS = 131072;
constexpr int SAMPLE_RATE_INPUT = 44100;
constexpr int SAMPLE_RATE_OUTPUT = 705600;
constexpr int UPSAMPLE_RATIO = 16;

// Filter coefficients (float32)
// IMPORTANT: 131k taps (512KB) is too large for embedding in source code.
// Recommended approach: Load from binary file at runtime using std::ifstream.
// Binary file: filter_131k_min_phase.bin (same directory)
// Example:
//   std::ifstream ifs("filter_131k_min_phase.bin", std::ios::binary);
//   std::vector<float> coeffs(FILTER_TAPS);
//   ifs.read(reinterpret_cast<char*>(coeffs.data()), FILTER_TAPS * sizeof(float));
extern const float FILTER_COEFFICIENTS[FILTER_TAPS];

#endif // FILTER_COEFFICIENTS_H
