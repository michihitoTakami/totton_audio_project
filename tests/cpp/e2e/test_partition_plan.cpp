#include "gpu/partition_plan.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <string>

using ConvolutionEngine::PartitionDescriptor;
using ConvolutionEngine::PartitionPlan;

class PartitionPlanTest : public ::testing::Test {
   protected:
    AppConfig::PartitionedConvolutionConfig defaultConfig_;

    void SetUp() override {
        defaultConfig_.enabled = true;
        defaultConfig_.fastPartitionTaps = 32768;
        defaultConfig_.minPartitionTaps = 16384;
        defaultConfig_.maxPartitions = 3;
    }
};

TEST_F(PartitionPlanTest, DisabledWhenConfigIsOff) {
    defaultConfig_.enabled = false;
    auto plan = ConvolutionEngine::buildPartitionPlan(640000, 16, defaultConfig_);
    EXPECT_FALSE(plan.enabled);
    EXPECT_TRUE(plan.partitions.empty());
}

TEST_F(PartitionPlanTest, SinglePartitionWhenFilterFitsFastPartition) {
    defaultConfig_.fastPartitionTaps = 8192;
    auto plan = ConvolutionEngine::buildPartitionPlan(4000, 16, defaultConfig_);

    ASSERT_TRUE(plan.enabled);
    ASSERT_EQ(plan.partitions.size(), 1u);
    const auto& part = plan.partitions.front();
    EXPECT_TRUE(part.realtime);
    EXPECT_EQ(part.taps, 4000);
    EXPECT_GE(part.fftSize, part.taps * 2);
}

TEST_F(PartitionPlanTest, BuildsFastAndTailPartitions) {
    auto plan = ConvolutionEngine::buildPartitionPlan(131072, 16, defaultConfig_);

    ASSERT_TRUE(plan.enabled);
    ASSERT_EQ(plan.partitions.size(), 3u);

    const auto& fast = plan.partitions[0];
    const auto& tailA = plan.partitions[1];
    const auto& tailB = plan.partitions[2];

    EXPECT_TRUE(fast.realtime);
    EXPECT_FALSE(tailA.realtime);
    EXPECT_FALSE(tailB.realtime);

    EXPECT_EQ(fast.taps, 32768);
    EXPECT_EQ(tailA.taps, 65536);
    EXPECT_EQ(tailB.taps, 32768);

    EXPECT_GE(fast.fftSize, fast.taps * 2);
    EXPECT_GE(tailA.fftSize, tailA.taps * 2);
    EXPECT_GE(tailB.fftSize, tailB.taps * 2);
}

TEST_F(PartitionPlanTest, DescribeIncludesLatencyInfo) {
    auto plan = ConvolutionEngine::buildPartitionPlan(65536, 16, defaultConfig_);
    ASSERT_TRUE(plan.enabled);

    std::string description = plan.describe(705600);
    EXPECT_NE(description.find("fast#0=32768"), std::string::npos);
    EXPECT_NE(description.find("tail#1"), std::string::npos);
    EXPECT_NE(description.find("ms"), std::string::npos);
}

TEST_F(PartitionPlanTest, TailFftMultipleExpandsFftSize) {
    defaultConfig_.tailFftMultiple = 8;
    auto plan = ConvolutionEngine::buildPartitionPlan(200000, 16, defaultConfig_);

    ASSERT_TRUE(plan.enabled);
    ASSERT_GT(plan.partitions.size(), 1u);

    const auto& tail = plan.partitions[1];
    int expectedFft = 1;
    const int target = std::max(tail.taps * defaultConfig_.tailFftMultiple, tail.taps + 1);
    while (expectedFft < target) {
        expectedFft <<= 1;
    }

    EXPECT_EQ(tail.fftSize, expectedFft);
    EXPECT_FALSE(tail.realtime);
}
