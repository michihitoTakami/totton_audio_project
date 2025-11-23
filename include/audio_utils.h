#ifndef AUDIO_UTILS_H
#define AUDIO_UTILS_H

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

}  // namespace AudioUtils

#endif  // AUDIO_UTILS_H
