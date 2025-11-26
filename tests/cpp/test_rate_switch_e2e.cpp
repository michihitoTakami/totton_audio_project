/**
 * @file test_rate_switch_e2e.cpp
 * @brief E2Eテスト: レート切替シーケンスでの自動交渉挙動を検証 (Issue #221)
 *
 * - negotiate() を連続呼び出しし、同一ファミリー/異なるファミリーの切替時の
 *   requiresReconfiguration フラグと出力レートを検証する
 * - DACが目標レートをサポートしない場合のフォールバック
 * - サポート外レートが拒否されること
 */

#include "auto_negotiation.h"
#include "dac_capability.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace AutoNegotiation;
using DacCapability::Capability;

namespace {

struct ExpectedConfig {
    int inputRate;
    int expectedOutputRate;
    int expectedRatio;
};

constexpr std::array<ExpectedConfig, 6> kExpectedConfigs = {{
    {44100, 705600, 16},
    {48000, 768000, 16},
    {88200, 705600, 8},
    {96000, 768000, 8},
    {176400, 705600, 4},
    {192000, 768000, 4},
}};

}  // namespace

class RateSwitchE2ETest : public ::testing::Test {
   protected:
    Capability createFullCapabilityDac() const {
        Capability cap;
        cap.deviceName = "test:full";
        cap.minSampleRate = 44100;
        cap.maxSampleRate = 768000;
        cap.supportedRates = {44100,  48000,  88200,  96000,  176400,
                              192000, 352800, 384000, 705600, 768000};
        cap.maxChannels = 2;
        cap.isValid = true;
        return cap;
    }

    Capability createLimitedCapabilityDac() const {
        Capability cap;
        cap.deviceName = "test:limited";
        cap.minSampleRate = 44100;
        cap.maxSampleRate = 192000;  // 16x出力はサポートしない
        cap.supportedRates = {44100, 48000, 88200, 96000, 176400, 192000};
        cap.maxChannels = 2;
        cap.isValid = true;
        return cap;
    }
};

TEST_F(RateSwitchE2ETest, NegotiatesExpectedOutputsForSupportedRates) {
    auto dac = createFullCapabilityDac();

    for (const auto& expected : kExpectedConfigs) {
        const auto config = negotiate(expected.inputRate, dac);
        ASSERT_TRUE(config.isValid) << "inputRate=" << expected.inputRate;
        EXPECT_EQ(config.outputRate, expected.expectedOutputRate);
        EXPECT_EQ(config.upsampleRatio, expected.expectedRatio);
        EXPECT_EQ(config.outputRate, config.inputRate * config.upsampleRatio);
    }
}

TEST_F(RateSwitchE2ETest, SameFamilySequenceKeepsCurrentOutputRate) {
    auto dac = createFullCapabilityDac();
    int currentOutputRate = 0;
    std::vector<int> sequence = {44100, 88200, 176400};

    for (int rate : sequence) {
        auto config = negotiate(rate, dac, currentOutputRate);
        ASSERT_TRUE(config.isValid);
        if (currentOutputRate == 0) {
            EXPECT_TRUE(config.requiresReconfiguration);
        } else {
            EXPECT_FALSE(config.requiresReconfiguration)
                << "Same-family switch must reuse existing output rate";
            EXPECT_EQ(config.outputRate, currentOutputRate);
        }
        currentOutputRate = config.outputRate;
    }
}

TEST_F(RateSwitchE2ETest, CrossFamilyTransitionForcesReconfiguration) {
    auto dac = createFullCapabilityDac();

    auto first = negotiate(44100, dac, 0);
    ASSERT_TRUE(first.isValid);
    ASSERT_TRUE(first.requiresReconfiguration);
    EXPECT_EQ(first.outputRate, 705600);

    auto second = negotiate(48000, dac, first.outputRate);
    ASSERT_TRUE(second.isValid);
    EXPECT_TRUE(second.requiresReconfiguration)
        << "44k -> 48k family switch must reconfigure ALSA";
    EXPECT_EQ(second.outputRate, 768000);
    EXPECT_NE(second.outputRate, first.outputRate);

    auto third = negotiate(44100, dac, second.outputRate);
    ASSERT_TRUE(third.isValid);
    EXPECT_TRUE(third.requiresReconfiguration)
        << "48k -> 44k family switch must reconfigure ALSA";
    EXPECT_EQ(third.outputRate, 705600);
}

TEST_F(RateSwitchE2ETest, LimitedDacFallsBackToSupportedOutputs) {
    auto dac = createLimitedCapabilityDac();
    int currentOutputRate = 0;

    auto config44 = negotiate(44100, dac, currentOutputRate);
    ASSERT_TRUE(config44.isValid);
    EXPECT_EQ(config44.outputRate, 176400);
    EXPECT_EQ(config44.upsampleRatio, 4);
    EXPECT_TRUE(config44.requiresReconfiguration);
    currentOutputRate = config44.outputRate;

    auto config48 = negotiate(48000, dac, currentOutputRate);
    ASSERT_TRUE(config48.isValid);
    EXPECT_EQ(config48.outputRate, 192000);
    EXPECT_EQ(config48.upsampleRatio, 4);
    EXPECT_TRUE(config48.requiresReconfiguration)
        << "Output frequency changed from 176.4kHz to 192kHz";
}

TEST_F(RateSwitchE2ETest, UnsupportedInputRatesAreRejected) {
    auto dac = createFullCapabilityDac();
    for (int rate : {22050, 32000, 11025}) {
        auto config = negotiate(rate, dac);
        EXPECT_FALSE(config.isValid) << "rate=" << rate;
        EXPECT_FALSE(config.errorMessage.empty());
    }
}

