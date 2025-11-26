#include "playback_buffer.h"

#include <gtest/gtest.h>

namespace {

constexpr size_t kPeriod = 32768;
constexpr size_t kDefaultMultiplier = 3;
constexpr size_t kDefaultReady = kPeriod * kDefaultMultiplier;

}  // namespace

TEST(PlaybackBufferThresholdTest, CrossfeedDisabledDefaultsToProducerBlock) {
    size_t producerBlock = kPeriod * 2;  // Between period and 3×period
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, false, 0, producerBlock),
              producerBlock);
}

TEST(PlaybackBufferThresholdTest, CrossfeedDisabledWithoutProducerFallsBackToTriplePeriod) {
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, false, 0), kDefaultReady);
}

TEST(PlaybackBufferThresholdTest, CrossfeedDisabledProducerClampedToPeriod) {
    size_t producerBlock = kPeriod / 2;
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, false, 0, producerBlock), kPeriod);
}

TEST(PlaybackBufferThresholdTest, CrossfeedDisabledProducerClampedToDefault) {
    size_t producerBlock = kPeriod * 4;
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, false, 0, producerBlock),
              kDefaultReady);
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
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, true, largeBlock), kDefaultReady);
}

TEST(PlaybackBufferThresholdTest, ZeroBlockSizeFallsBackToDefault) {
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, true, 0), kDefaultReady);
}


