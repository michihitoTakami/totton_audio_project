/**
 * @file test_fallback_manager.cpp
 * @brief Unit tests for Fallback Manager (Issue #139)
 *
 * Tests the dynamic fallback functionality for GPU overload and XRUN handling.
 * Note: GPU monitoring tests require NVML, but tests are designed to work
 * without GPU hardware (using stub values when NVML unavailable).
 */

#include "fallback_manager.h"
#include "logging/metrics.h"

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <thread>

using namespace FallbackManager;

class FallbackManagerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        stateChanges_.clear();
        lastState_ = FallbackState::Normal;
    }

    void TearDown() override {
        if (manager_) {
            manager_->shutdown();
            delete manager_;
            manager_ = nullptr;
        }
    }

    // Create manager with test callback. If requireMonitoring is true and GPU monitoring
    // is unavailable (e.g., NVML missing), the test will be skipped.
    Manager* createManager(const FallbackConfig& config, bool requireMonitoring = false) {
        manager_ = new Manager();
        auto callback = [this](FallbackState state) {
            stateChanges_.push_back(state);
            lastState_ = state;
        };
        manager_->initialize(config, callback);
        if (requireMonitoring && manager_ && !manager_->isMonitoringEnabled()) {
            GTEST_SKIP() << "GPU monitoring unavailable - skipping fallback-dependent test";
        }
        return manager_;
    }

    Manager* manager_ = nullptr;
    std::vector<FallbackState> stateChanges_;
    FallbackState lastState_ = FallbackState::Normal;
};

// ============================================================================
// Initial State Tests
// ============================================================================

TEST_F(FallbackManagerTest, InitialState_IsNormal) {
    Manager mgr;
    EXPECT_EQ(mgr.getState(), FallbackState::Normal);
    EXPECT_FALSE(mgr.isInFallback());
}

TEST_F(FallbackManagerTest, InitialState_BeforeInitialize_IsNormal) {
    Manager mgr;
    EXPECT_EQ(mgr.getState(), FallbackState::Normal);
    EXPECT_FLOAT_EQ(mgr.getGpuUtilization(), 0.0f);
}

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(FallbackManagerTest, Initialize_WithDefaultConfig_Succeeds) {
    FallbackConfig config;
    Manager* mgr = createManager(config);
    ASSERT_NE(mgr, nullptr);
    EXPECT_EQ(mgr->getState(), FallbackState::Normal);

    // Give monitor thread a moment to start
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

TEST_F(FallbackManagerTest, Initialize_WithCustomConfig_Succeeds) {
    FallbackConfig config;
    config.gpuThreshold = 90.0f;
    config.gpuThresholdCount = 5;
    config.gpuRecoveryThreshold = 80.0f;
    config.gpuRecoveryCount = 10;
    config.xrunTriggersFallback = false;
    config.monitorIntervalMs = 50;

    Manager* mgr = createManager(config);
    ASSERT_NE(mgr, nullptr);
    EXPECT_EQ(mgr->getState(), FallbackState::Normal);
}

TEST_F(FallbackManagerTest, Shutdown_StopsMonitorThread) {
    FallbackConfig config;
    Manager* mgr = createManager(config);
    ASSERT_NE(mgr, nullptr);

    mgr->shutdown();
    // Thread should be stopped (no crash on second shutdown)
    mgr->shutdown();
}

// ============================================================================
// XRUN Detection Tests
// ============================================================================

TEST_F(FallbackManagerTest, NotifyXrun_IncrementsCount) {
    FallbackConfig config;
    config.xrunTriggersFallback = true;
    Manager* mgr = createManager(config);

    mgr->notifyXrun();
    auto stats = mgr->getStats();
    EXPECT_EQ(stats.xrunCount, 1);

    mgr->notifyXrun();
    stats = mgr->getStats();
    EXPECT_EQ(stats.xrunCount, 2);
}

TEST_F(FallbackManagerTest, NotifyXrun_TriggersFallback_WhenEnabled) {
    FallbackConfig config;
    config.xrunTriggersFallback = true;
    Manager* mgr = createManager(config, true);

    EXPECT_EQ(mgr->getState(), FallbackState::Normal);

    mgr->notifyXrun();

    // Should trigger fallback immediately
    EXPECT_EQ(mgr->getState(), FallbackState::Fallback);
    EXPECT_TRUE(mgr->isInFallback());
    EXPECT_EQ(stateChanges_.size(), 1);
    EXPECT_EQ(stateChanges_[0], FallbackState::Fallback);
}

TEST_F(FallbackManagerTest, NotifyXrun_DoesNotTriggerFallback_WhenDisabled) {
    FallbackConfig config;
    config.xrunTriggersFallback = false;
    Manager* mgr = createManager(config);

    EXPECT_EQ(mgr->getState(), FallbackState::Normal);

    mgr->notifyXrun();

    // Should NOT trigger fallback
    EXPECT_EQ(mgr->getState(), FallbackState::Normal);
    EXPECT_FALSE(mgr->isInFallback());
    EXPECT_EQ(stateChanges_.size(), 0);
}

TEST_F(FallbackManagerTest, NotifyXrun_DoesNotTriggerFallback_WhenAlreadyInFallback) {
    FallbackConfig config;
    config.xrunTriggersFallback = true;
    Manager* mgr = createManager(config);

    // Trigger fallback first
    mgr->notifyXrun();
    EXPECT_EQ(mgr->getState(), FallbackState::Fallback);
    stateChanges_.clear();

    // Another XRUN should not trigger another callback
    mgr->notifyXrun();
    EXPECT_EQ(stateChanges_.size(), 0);  // No new state change
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(FallbackManagerTest, GetStats_ReturnsCorrectValues) {
    FallbackConfig config;
    config.xrunTriggersFallback = true;
    Manager* mgr = createManager(config, true);

    mgr->notifyXrun();  // Triggers fallback

    auto stats = mgr->getStats();
    EXPECT_EQ(stats.xrunCount, 1);
    EXPECT_EQ(stats.fallbackActivations, 1);
    EXPECT_EQ(stats.fallbackRecoveries, 0);
}

TEST_F(FallbackManagerTest, GetStats_TracksMultipleActivations) {
    FallbackConfig config;
    config.xrunTriggersFallback = true;
    Manager* mgr = createManager(config, true);

    mgr->notifyXrun();  // First activation
    mgr->notifyXrun();  // Should not count as new activation

    auto stats = mgr->getStats();
    EXPECT_EQ(stats.fallbackActivations, 1);
}

// ============================================================================
// GPU Utilization Tests
// ============================================================================

TEST_F(FallbackManagerTest, GetGpuUtilization_ReturnsValue) {
    FallbackConfig config;
    Manager* mgr = createManager(config);

    // GPU utilization should be accessible (may be 0 if NVML unavailable)
    float util = mgr->getGpuUtilization();
    EXPECT_GE(util, 0.0f);
    EXPECT_LE(util, 100.0f);
}

// ============================================================================
// State Callback Tests
// ============================================================================

TEST_F(FallbackManagerTest, StateCallback_InvokedOnFallback) {
    FallbackConfig config;
    config.xrunTriggersFallback = true;
    Manager* mgr = createManager(config, true);

    EXPECT_EQ(stateChanges_.size(), 0);

    mgr->notifyXrun();

    EXPECT_EQ(stateChanges_.size(), 1);
    EXPECT_EQ(stateChanges_[0], FallbackState::Fallback);
    EXPECT_EQ(lastState_, FallbackState::Fallback);
}

TEST_F(FallbackManagerTest, StateCallback_NotInvokedOnDuplicateState) {
    FallbackConfig config;
    config.xrunTriggersFallback = true;
    Manager* mgr = createManager(config, true);

    mgr->notifyXrun();
    EXPECT_EQ(stateChanges_.size(), 1);
    stateChanges_.clear();

    // Second XRUN should not trigger callback (already in fallback)
    mgr->notifyXrun();
    EXPECT_EQ(stateChanges_.size(), 0);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(FallbackManagerTest, Config_GpuThreshold_DefaultsTo80) {
    FallbackConfig config;
    EXPECT_FLOAT_EQ(config.gpuThreshold, 80.0f);
}

TEST_F(FallbackManagerTest, Config_GpuThresholdCount_DefaultsTo3) {
    FallbackConfig config;
    EXPECT_EQ(config.gpuThresholdCount, 3);
}

TEST_F(FallbackManagerTest, Config_GpuRecoveryThreshold_DefaultsTo70) {
    FallbackConfig config;
    EXPECT_FLOAT_EQ(config.gpuRecoveryThreshold, 70.0f);
}

TEST_F(FallbackManagerTest, Config_GpuRecoveryCount_DefaultsTo5) {
    FallbackConfig config;
    EXPECT_EQ(config.gpuRecoveryCount, 5);
}

TEST_F(FallbackManagerTest, Config_XrunTriggersFallback_DefaultsToTrue) {
    FallbackConfig config;
    EXPECT_TRUE(config.xrunTriggersFallback);
}

TEST_F(FallbackManagerTest, Config_MonitorIntervalMs_DefaultsTo100) {
    FallbackConfig config;
    EXPECT_EQ(config.monitorIntervalMs, 100);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(FallbackManagerTest, Shutdown_BeforeInitialize_DoesNotCrash) {
    Manager mgr;
    mgr.shutdown();  // Should not crash
}

TEST_F(FallbackManagerTest, GetStats_BeforeInitialize_ReturnsZero) {
    Manager mgr;
    auto stats = mgr.getStats();
    EXPECT_EQ(stats.xrunCount, 0);
    EXPECT_EQ(stats.fallbackActivations, 0);
    EXPECT_EQ(stats.fallbackRecoveries, 0);
}

TEST_F(FallbackManagerTest, NotifyXrun_BeforeInitialize_DoesNotCrash) {
    Manager mgr;
    mgr.notifyXrun();  // Should not crash
    // State should remain Normal
    EXPECT_EQ(mgr.getState(), FallbackState::Normal);
}

TEST_F(FallbackManagerTest, MultipleShutdowns_DoesNotCrash) {
    FallbackConfig config;
    Manager* mgr = createManager(config);
    mgr->shutdown();
    mgr->shutdown();  // Second shutdown should not crash
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(FallbackManagerTest, ConcurrentXrunNotifications_ThreadSafe) {
    FallbackConfig config;
    config.xrunTriggersFallback = true;
    Manager* mgr = createManager(config, true);

    // Simulate concurrent XRUN notifications
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([mgr]() { mgr->notifyXrun(); });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should have at least one activation
    auto stats = mgr->getStats();
    EXPECT_GE(stats.xrunCount, 10);
    EXPECT_GE(stats.fallbackActivations, 1);
}

// ============================================================================
// Integration with Metrics Tests
// ============================================================================

TEST_F(FallbackManagerTest, Integration_WithMetrics_Works) {
    // Test that fallback manager can work with metrics system
    // (even if NVML is not available)
    FallbackConfig config;
    Manager* mgr = createManager(config);

    // Should be able to get GPU metrics (may return stub values)
    auto gpuMetrics = gpu_upsampler::metrics::getGpuMetrics();
    EXPECT_GE(gpuMetrics.utilization, 0.0f);
    EXPECT_LE(gpuMetrics.utilization, 100.0f);

    // Manager should still function
    EXPECT_EQ(mgr->getState(), FallbackState::Normal);
}

