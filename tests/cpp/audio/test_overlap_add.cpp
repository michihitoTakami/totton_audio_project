/**
 * @file test_overlap_add.cpp
 * @brief Unit tests for overlap-add crossfade utilities (Issue #1009)
 */

#include "audio/overlap_add.h"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace {

std::vector<float> makeRamp(std::size_t n, float start, float step) {
    std::vector<float> out(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = start + static_cast<float>(i) * step;
    }
    return out;
}

}  // namespace

TEST(OverlapAdd, ReconstructsIdentityWithPadding) {
    const std::size_t chunkLen = 8;
    const std::size_t overlap = 2;
    const std::size_t hop = chunkLen - overlap;
    const std::size_t totalFrames = 20;

    std::vector<float> inL = makeRamp(totalFrames, 0.0f, 0.1f);
    std::vector<float> inR = makeRamp(totalFrames, 1.0f, -0.05f);

    // Split into fixed-length chunks with zero padding (like the offline PoC).
    std::vector<std::vector<float>> chunksL;
    std::vector<std::vector<float>> chunksR;
    for (std::size_t start = 0; start < totalFrames; start += hop) {
        std::vector<float> cL(chunkLen, 0.0f);
        std::vector<float> cR(chunkLen, 0.0f);
        for (std::size_t i = 0; i < chunkLen; ++i) {
            std::size_t pos = start + i;
            if (pos < totalFrames) {
                cL[i] = inL[pos];
                cR[i] = inR[pos];
            }
        }
        chunksL.push_back(std::move(cL));
        chunksR.push_back(std::move(cR));

        if (start + chunkLen >= totalFrames) {
            break;
        }
    }

    std::vector<float> outL;
    std::vector<float> outR;
    ASSERT_TRUE(AudioUtils::overlapAddStereoChunks(chunksL, chunksR, hop, overlap, totalFrames,
                                                   outL, outR));
    ASSERT_EQ(outL.size(), totalFrames);
    ASSERT_EQ(outR.size(), totalFrames);

    for (std::size_t i = 0; i < totalFrames; ++i) {
        EXPECT_NEAR(outL[i], inL[i], 1e-6f) << "i=" << i;
        EXPECT_NEAR(outR[i], inR[i], 1e-6f) << "i=" << i;
    }
}

TEST(OverlapAdd, CrossfadesWithRaisedCosine) {
    const std::size_t chunkLen = 8;
    const std::size_t overlap = 4;
    const std::size_t hop = chunkLen - overlap;  // = 4

    std::vector<std::vector<float>> chunksL;
    std::vector<std::vector<float>> chunksR;

    chunksL.push_back(std::vector<float>(chunkLen, 1.0f));
    chunksR.push_back(std::vector<float>(chunkLen, 1.0f));
    chunksL.push_back(std::vector<float>(chunkLen, 3.0f));
    chunksR.push_back(std::vector<float>(chunkLen, 3.0f));

    std::vector<float> outL;
    std::vector<float> outR;
    ASSERT_TRUE(AudioUtils::overlapAddStereoChunks(chunksL, chunksR, hop, overlap, 0, outL, outR));

    // Output length: hop*(n-1)+chunkLen = 4+8 = 12
    ASSERT_EQ(outL.size(), static_cast<std::size_t>(12));
    ASSERT_EQ(outR.size(), static_cast<std::size_t>(12));

    // Non-overlap regions.
    for (std::size_t i = 0; i < hop; ++i) {
        EXPECT_NEAR(outL[i], 1.0f, 1e-6f);
        EXPECT_NEAR(outR[i], 1.0f, 1e-6f);
    }
    for (std::size_t i = hop + overlap; i < outL.size(); ++i) {
        EXPECT_NEAR(outL[i], 3.0f, 1e-6f);
        EXPECT_NEAR(outR[i], 3.0f, 1e-6f);
    }

    // Overlap region should smoothly transition 1 -> 3 with raised cosine.
    std::vector<float> fade = AudioUtils::makeRaisedCosineFade(overlap);
    for (std::size_t j = 0; j < overlap; ++j) {
        float wIn = fade[j];
        float wOut = fade[overlap - 1 - j];
        float expected = 1.0f * wOut + 3.0f * wIn;
        std::size_t idx = hop + j;
        EXPECT_NEAR(outL[idx], expected, 1e-5f) << "j=" << j;
        EXPECT_NEAR(outR[idx], expected, 1e-5f) << "j=" << j;
    }
}
