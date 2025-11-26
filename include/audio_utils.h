#ifndef AUDIO_UTILS_H
#define AUDIO_UTILS_H

#include <cmath>
#include <cstddef>

// Common audio utility functions shared between daemons
// Issue #105: Daemon common code extraction

namespace AudioUtils {

// Deinterleave stereo audio: LRLRLR... -> L[], R[]
inline void deinterleaveStereo(const float* src, float* left, float* right, size_t frames) {
    for (size_t i = 0; i < frames; ++i) {
        left[i] = src[i * 2];
        right[i] = src[i * 2 + 1];
    }
}

// Interleave stereo audio: L[], R[] -> LRLRLR...
inline void interleaveStereo(const float* left, const float* right, float* dst, size_t frames) {
    for (size_t i = 0; i < frames; ++i) {
        dst[i * 2] = left[i];
        dst[i * 2 + 1] = right[i];
    }
}

// Interleave stereo audio with gain: L[], R[] -> LRLRLR... (with gain applied)
inline void interleaveStereoWithGain(const float* left, const float* right, float* dst,
                                     size_t frames, float gain) {
    for (size_t i = 0; i < frames; ++i) {
        dst[i * 2] = left[i] * gain;
        dst[i * 2 + 1] = right[i] * gain;
    }
}

inline void applyInterleavedGain(float* interleaved, size_t frames, float gain) {
    if (!interleaved || frames == 0) {
        return;
    }
    if (std::fabs(gain - 1.0f) < 1e-6f) {
        return;
    }
    const size_t samples = frames * 2;
    for (size_t i = 0; i < samples; ++i) {
        interleaved[i] *= gain;
    }
}

inline float computeInterleavedPeak(const float* interleaved, size_t frames) {
    if (!interleaved || frames == 0) {
        return 0.0f;
    }
    float peak = 0.0f;
    const size_t samples = frames * 2;
    for (size_t i = 0; i < samples; ++i) {
        float value = std::fabs(interleaved[i]);
        if (value > peak) {
            peak = value;
        }
    }
    return peak;
}

}  // namespace AudioUtils

#endif  // AUDIO_UTILS_H
