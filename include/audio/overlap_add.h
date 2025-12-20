#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Streaming-style overlap-add with raised-cosine crossfade.
//
// This is a reference implementation used for design validation (#1009).
// The realtime/high-latency integration is done in follow-up issues (#1010/#1014).

namespace AudioUtils {

// Raised-cosine fade-in window (0..1, inclusive endpoints).
// Equivalent to: fade(t) = 0.5 - 0.5*cos(pi*t), t in [0,1].
inline std::vector<float> makeRaisedCosineFade(std::size_t length) {
    std::vector<float> fade;
    fade.resize(length, 0.0f);
    if (length == 0) {
        return fade;
    }
    if (length == 1) {
        fade[0] = 1.0f;
        return fade;
    }

    constexpr float kPi = 3.14159265358979323846f;
    for (std::size_t i = 0; i < length; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(length - 1);
        fade[i] = 0.5f - 0.5f * std::cos(kPi * t);
    }
    return fade;
}

// Overlap-add multiple fixed-length stereo chunks into a single output.
//
// Parameters:
// - chunksLeft/chunksRight: list of chunks, all must have the same length.
// - hopFrames: start offset between chunks (chunkLen - overlapFrames).
// - overlapFrames: crossfade region length.
// - totalFrames: trim output to this length (0 = keep full length).
//
// Returns false when inputs are inconsistent.
inline bool overlapAddStereoChunks(const std::vector<std::vector<float>>& chunksLeft,
                                   const std::vector<std::vector<float>>& chunksRight,
                                   std::size_t hopFrames, std::size_t overlapFrames,
                                   std::size_t totalFrames, std::vector<float>& outLeft,
                                   std::vector<float>& outRight) {
    outLeft.clear();
    outRight.clear();

    if (chunksLeft.size() != chunksRight.size()) {
        return false;
    }
    if (chunksLeft.empty()) {
        return true;
    }

    const std::size_t n = chunksLeft.size();
    const std::size_t chunkLen = chunksLeft[0].size();
    if (chunkLen == 0) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (chunksLeft[i].size() != chunkLen || chunksRight[i].size() != chunkLen) {
            return false;
        }
    }
    if (overlapFrames >= chunkLen) {
        return false;
    }
    if (hopFrames != (chunkLen - overlapFrames)) {
        return false;
    }

    std::size_t outLen = hopFrames * (n - 1) + chunkLen;
    if (totalFrames > 0) {
        outLen = std::min(outLen, totalFrames);
    }

    std::vector<float> wsum;
    outLeft.assign(outLen, 0.0f);
    outRight.assign(outLen, 0.0f);
    wsum.assign(outLen, 0.0f);

    std::vector<float> fade = makeRaisedCosineFade(overlapFrames);

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t start = i * hopFrames;
        if (start >= outLen) {
            break;
        }

        // Build per-chunk weights: fade-in at head (except first chunk), fade-out at tail
        // (except last chunk).
        std::vector<float> w(chunkLen, 1.0f);
        if (overlapFrames > 0 && i > 0) {
            for (std::size_t j = 0; j < overlapFrames; ++j) {
                w[j] = fade[j];
            }
        }
        if (overlapFrames > 0 && i + 1 < n) {
            for (std::size_t j = 0; j < overlapFrames; ++j) {
                w[chunkLen - overlapFrames + j] = fade[overlapFrames - 1 - j];
            }
        }

        for (std::size_t j = 0; j < chunkLen; ++j) {
            std::size_t outPos = start + j;
            if (outPos >= outLen) {
                break;
            }
            float ww = w[j];
            outLeft[outPos] += chunksLeft[i][j] * ww;
            outRight[outPos] += chunksRight[i][j] * ww;
            wsum[outPos] += ww;
        }
    }

    constexpr float kEps = 1e-8f;
    for (std::size_t i = 0; i < outLen; ++i) {
        float denom = std::max(wsum[i], kEps);
        outLeft[i] /= denom;
        outRight[i] /= denom;
    }

    return true;
}

}  // namespace AudioUtils
