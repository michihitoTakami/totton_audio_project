#include "io/playback_buffer.h"

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
    auto blockSize = static_cast<size_t>(kPeriod * 1.5);
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, true, blockSize), blockSize);
}

TEST(PlaybackBufferThresholdTest, CrossfeedBlockAboveDefaultFallsBackToTriplePeriod) {
    size_t largeBlock = kPeriod * 4;
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, true, largeBlock), kDefaultReady);
}

TEST(PlaybackBufferThresholdTest, ZeroBlockSizeFallsBackToDefault) {
    EXPECT_EQ(PlaybackBuffer::computeReadyThreshold(kPeriod, true, 0), kDefaultReady);
}

TEST(PlaybackBufferCapacityTest, NoOverflowStoresAllFrames) {
    constexpr size_t kCurrent = 64;
    constexpr size_t kProduced = 128;
    constexpr size_t kMax = 256;
    auto decision = PlaybackBuffer::planCapacityEnforcement(kCurrent, kProduced, kMax);
    EXPECT_EQ(decision.dropFromExisting, 0u);
    EXPECT_EQ(decision.newDataOffset, 0u);
    EXPECT_EQ(decision.framesToStore, kProduced);
}

TEST(PlaybackBufferCapacityTest, DropsOldFramesWhenNearLimit) {
    constexpr size_t kCurrent = 200;
    constexpr size_t kProduced = 120;
    constexpr size_t kMax = 256;
    auto decision = PlaybackBuffer::planCapacityEnforcement(kCurrent, kProduced, kMax);
    EXPECT_EQ(decision.dropFromExisting, (kCurrent + kProduced - kMax));
    EXPECT_EQ(decision.newDataOffset, 0u);
    EXPECT_EQ(decision.framesToStore, kProduced);
}

TEST(PlaybackBufferCapacityTest, TrimsNewFramesWhenProducerExceedsLimit) {
    constexpr size_t kCurrent = 0;
    constexpr size_t kProduced = 400;
    constexpr size_t kMax = 256;
    auto decision = PlaybackBuffer::planCapacityEnforcement(kCurrent, kProduced, kMax);
    EXPECT_EQ(decision.dropFromExisting, 0u);
    EXPECT_EQ(decision.newDataOffset, kProduced - kMax);
    EXPECT_EQ(decision.framesToStore, kMax);
}

TEST(PlaybackBufferCapacityTest, CombinedDropAndTrim) {
    constexpr size_t kCurrent = 240;
    constexpr size_t kProduced = 300;
    constexpr size_t kMax = 256;
    auto decision = PlaybackBuffer::planCapacityEnforcement(kCurrent, kProduced, kMax);
    EXPECT_EQ(decision.newDataOffset, kProduced - kMax);
    EXPECT_EQ(decision.framesToStore, kMax);
    EXPECT_EQ(decision.dropFromExisting, kCurrent);
}
