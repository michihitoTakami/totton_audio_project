#include "playback_buffer.h"

#include <gtest/gtest.h>

namespace {

constexpr size_t kPeriod = 32768;

}  // namespace

TEST(PlaybackBufferThresholdTest, CrossfeedDisabledUsesTriplePeriod) {
    size_t expected = kPeriod * 3;
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, false, 0), expected);
}

TEST(PlaybackBufferThresholdTest, CrossfeedBlockSmallerThanPeriodClampsToPeriod) {
    size_t smallBlock = kPeriod / 2;
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, true, smallBlock), kPeriod);
}

TEST(PlaybackBufferThresholdTest, CrossfeedBlockWithinRangeUsesBlockSize) {
    size_t blockSize = static_cast<size_t>(kPeriod * 1.5);
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, true, blockSize), blockSize);
}

TEST(PlaybackBufferThresholdTest, CrossfeedBlockAboveDefaultFallsBackToTriplePeriod) {
    size_t largeBlock = kPeriod * 4;
    size_t expected = kPeriod * 3;
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, true, largeBlock), expected);
}

TEST(PlaybackBufferThresholdTest, ZeroBlockSizeFallsBackToDefault) {
    size_t expected = kPeriod * 3;
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, true, 0), expected);
}


