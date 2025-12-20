/**
 * @file test_safety_controller.cpp
 * @brief Tests for delimiter safety controller (Issue #1014).
 */

#include "delimiter/safety_controller.h"

#include <gtest/gtest.h>
#include <vector>

TEST(DelimiterSafetyController, CrossfadesToBypass) {
    delimiter::SafetyConfig config;
    config.sampleRate = 1000;
    config.fadeDurationMs = 10;
    delimiter::SafetyController controller(config);

    std::vector<float> processed(20, 1.0f);
    std::vector<float> bypass(20, 0.0f);
    std::vector<float> outL;
    std::vector<float> outR;

    ASSERT_TRUE(controller.requestBypass(delimiter::FallbackReason::InferenceFailure, "fail"));
    ASSERT_TRUE(controller.mixChunk(processed, processed, bypass, bypass, outL, outR));

    ASSERT_EQ(outL.size(), processed.size());
    ASSERT_EQ(outR.size(), processed.size());

    EXPECT_NEAR(outL.front(), 1.0f, 1e-6f);
    EXPECT_NEAR(outL[9], 0.0f, 1e-6f);
    for (std::size_t i = 0; i + 1 < 10; ++i) {
        EXPECT_GE(outL[i], outL[i + 1]) << "i=" << i;
    }
    for (std::size_t i = 10; i < outL.size(); ++i) {
        EXPECT_NEAR(outL[i], 0.0f, 1e-6f) << "i=" << i;
        EXPECT_NEAR(outR[i], 0.0f, 1e-6f) << "i=" << i;
    }
}

TEST(DelimiterSafetyController, BypassLockRequiresUserRequest) {
    delimiter::SafetyConfig config;
    config.failureCountToBypass = 1;
    delimiter::SafetyController controller(config);

    delimiter::InferenceResult result{delimiter::InferenceStatus::Error, "boom"};
    ASSERT_TRUE(controller.observeInferenceResult(result));

    auto status = controller.status();
    EXPECT_TRUE(status.bypassLocked);
    EXPECT_EQ(status.lastFallbackReason, delimiter::FallbackReason::InferenceFailure);

    EXPECT_FALSE(controller.requestActive(false, "auto"));
    EXPECT_TRUE(controller.requestActive(true, "user"));

    status = controller.status();
    EXPECT_FALSE(status.bypassLocked);
}

TEST(DelimiterSafetyController, OverloadTriggersBypassAfterThreshold) {
    delimiter::SafetyConfig config;
    config.maxRealtimeFactor = 1.0;
    config.overloadCountToBypass = 2;
    config.lockOnOverload = false;
    delimiter::SafetyController controller(config);

    EXPECT_FALSE(controller.observeOverload(1.2, 0.0));
    auto status = controller.status();
    EXPECT_FALSE(status.bypassLocked);

    EXPECT_TRUE(controller.observeOverload(1.2, 0.0));
    status = controller.status();
    EXPECT_FALSE(status.bypassLocked);
    EXPECT_EQ(status.lastFallbackReason, delimiter::FallbackReason::Overload);
}

TEST(DelimiterSafetyController, AutoRestoreWhenHealthy) {
    delimiter::SafetyConfig config;
    config.recoveryCountToRestore = 2;
    config.fadeDurationMs = 0;
    delimiter::SafetyController controller(config);

    ASSERT_TRUE(controller.requestBypass(delimiter::FallbackReason::Manual, "manual"));

    controller.observeHealthy();
    controller.observeHealthy();

    auto status = controller.status();
    EXPECT_EQ(status.targetMode, delimiter::ProcessingMode::Active);
}

TEST(DelimiterSafetyController, ForceBypassAndForceActive) {
    delimiter::SafetyConfig config;
    config.sampleRate = 1000;
    config.fadeDurationMs = 10;
    delimiter::SafetyController controller(config);

    controller.forceBypassLock(delimiter::FallbackReason::Manual, "user off");
    auto status = controller.status();
    EXPECT_EQ(status.mode, delimiter::ProcessingMode::Bypass);
    EXPECT_TRUE(status.bypassLocked);
    EXPECT_EQ(status.lastFallbackReason, delimiter::FallbackReason::Manual);

    controller.forceActive("user on");
    status = controller.status();
    EXPECT_EQ(status.mode, delimiter::ProcessingMode::Active);
    EXPECT_FALSE(status.bypassLocked);
    EXPECT_EQ(status.lastFallbackReason, delimiter::FallbackReason::None);
    EXPECT_EQ(status.targetMode, delimiter::ProcessingMode::Active);
}
