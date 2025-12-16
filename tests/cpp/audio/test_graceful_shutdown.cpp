#include "graceful_shutdown.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace GracefulShutdown;

class GracefulShutdownTest : public ::testing::Test {
   protected:
    void SetUp() override {
        state_.reset();
        controller_.setSignalState(&state_);
        fadeOutCalled_ = false;
        quitLoopCalled_ = false;
        controller_.setFadeOutCallback([this]() { fadeOutCalled_ = true; });
        controller_.setQuitLoopCallback([this]() { quitLoopCalled_ = true; });
        // Set log callback to print to stdout (safe in test context)
        controller_.setLogCallback([](const char* msg) { std::cout << msg << std::endl; });
    }

    SignalState state_;
    Controller controller_;
    bool fadeOutCalled_ = false;
    bool quitLoopCalled_ = false;
};

// ========== Signal State Tests ==========

TEST_F(GracefulShutdownTest, SignalState_InitiallyZero) {
    SignalState s;
    EXPECT_EQ(s.shutdown, 0);
    EXPECT_EQ(s.reload, 0);
    EXPECT_EQ(s.received, 0);
}

TEST_F(GracefulShutdownTest, SignalState_Reset) {
    state_.shutdown = 1;
    state_.reload = 1;
    state_.received = 15;
    state_.reset();
    EXPECT_EQ(state_.shutdown, 0);
    EXPECT_EQ(state_.reload, 0);
    EXPECT_EQ(state_.received, 0);
}

// ========== Controller Initial State Tests ==========

TEST_F(GracefulShutdownTest, Controller_InitialState) {
    EXPECT_TRUE(controller_.isRunning());
    EXPECT_FALSE(controller_.isReloadRequested());
    EXPECT_FALSE(controller_.isMainLoopRunning());
    EXPECT_EQ(controller_.getLastAction(), Controller::Action::NONE);
}

// ========== No Signal Tests ==========

TEST_F(GracefulShutdownTest, ProcessPendingSignals_NoSignal_ReturnsFalse) {
    EXPECT_FALSE(controller_.processPendingSignals());
    EXPECT_EQ(controller_.getLastAction(), Controller::Action::NONE);
    EXPECT_FALSE(fadeOutCalled_);
    EXPECT_FALSE(quitLoopCalled_);
}

// ========== SIGTERM Tests ==========

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGTERM_ReturnsTrue) {
    state_.shutdown = 1;
    state_.received = SIGTERM;

    EXPECT_TRUE(controller_.processPendingSignals());
    EXPECT_EQ(controller_.getLastAction(), Controller::Action::SHUTDOWN);
    EXPECT_EQ(controller_.getLastSignal(), SIGTERM);
}

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGTERM_ClearsShutdownFlag) {
    state_.shutdown = 1;
    state_.received = SIGTERM;

    controller_.processPendingSignals();
    EXPECT_EQ(state_.shutdown, 0);
}

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGTERM_CallsFadeOut) {
    state_.shutdown = 1;
    state_.received = SIGTERM;

    controller_.processPendingSignals();
    EXPECT_TRUE(fadeOutCalled_);
}

TEST_F(GracefulShutdownTest,
       ProcessPendingSignals_SIGTERM_SetsRunningFalse_WhenMainLoopNotRunning) {
    state_.shutdown = 1;
    state_.received = SIGTERM;
    controller_.setMainLoopRunning(false);

    controller_.processPendingSignals();
    EXPECT_FALSE(controller_.isRunning());
    EXPECT_FALSE(quitLoopCalled_);
}

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGTERM_CallsQuitLoop_WhenMainLoopRunning) {
    state_.shutdown = 1;
    state_.received = SIGTERM;
    controller_.setMainLoopRunning(true);

    controller_.processPendingSignals();
    EXPECT_TRUE(quitLoopCalled_);
    // Also verify running_ is set to false (fallback)
    EXPECT_FALSE(controller_.isRunning());
}

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGTERM_ClearsReloadRequested) {
    state_.shutdown = 1;
    state_.received = SIGTERM;
    // Simulate that reload was previously requested
    state_.reload = 1;  // This should also be cleared

    controller_.processPendingSignals();
    EXPECT_FALSE(controller_.isReloadRequested());
}

// ========== SIGINT Tests ==========

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGINT_TreatedAsShutdown) {
    state_.shutdown = 1;
    state_.received = SIGINT;

    EXPECT_TRUE(controller_.processPendingSignals());
    EXPECT_EQ(controller_.getLastAction(), Controller::Action::SHUTDOWN);
    EXPECT_EQ(controller_.getLastSignal(), SIGINT);
}

// ========== SIGHUP Tests ==========

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGHUP_ReturnsTrue) {
    state_.reload = 1;
    state_.received = SIGHUP;

    EXPECT_TRUE(controller_.processPendingSignals());
    EXPECT_EQ(controller_.getLastAction(), Controller::Action::RELOAD);
    EXPECT_EQ(controller_.getLastSignal(), SIGHUP);
}

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGHUP_SetsReloadRequested) {
    state_.reload = 1;
    state_.received = SIGHUP;

    controller_.processPendingSignals();
    EXPECT_TRUE(controller_.isReloadRequested());
}

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGHUP_ClearsReloadFlag) {
    state_.reload = 1;
    state_.received = SIGHUP;

    controller_.processPendingSignals();
    EXPECT_EQ(state_.reload, 0);
}

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGHUP_CallsFadeOut) {
    state_.reload = 1;
    state_.received = SIGHUP;

    controller_.processPendingSignals();
    EXPECT_TRUE(fadeOutCalled_);
}

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGHUP_SetsRunningFalse_Fallback) {
    // Verify running_ is always set to false (even when quitLoopCallback is called)
    state_.reload = 1;
    state_.received = SIGHUP;
    controller_.setMainLoopRunning(true);

    controller_.processPendingSignals();
    EXPECT_TRUE(quitLoopCalled_);
    // Fallback: running_ should be false even if callback was called
    EXPECT_FALSE(controller_.isRunning());
}

// ========== CRITICAL: SIGTERM + SIGHUP Race Condition Tests ==========

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGTERM_TakesPriorityOverSIGHUP) {
    // Both signals arrive in the same cycle
    state_.shutdown = 1;
    state_.reload = 1;
    state_.received = SIGTERM;

    EXPECT_TRUE(controller_.processPendingSignals());

    // SIGTERM (shutdown) should take priority
    EXPECT_EQ(controller_.getLastAction(), Controller::Action::SHUTDOWN);

    // Reload flag should be cleared to prevent restart loop
    EXPECT_FALSE(controller_.isReloadRequested());

    // Both signal flags should be cleared
    EXPECT_EQ(state_.shutdown, 0);
    EXPECT_EQ(state_.reload, 0);
}

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGTERM_ClearsPendingReload) {
    // SIGHUP arrives first, sets reload flag
    state_.reload = 1;
    state_.received = SIGHUP;

    // Then SIGTERM arrives before processPendingSignals is called
    state_.shutdown = 1;
    state_.received = SIGTERM;

    controller_.processPendingSignals();

    // Should shutdown, not reload
    EXPECT_EQ(controller_.getLastAction(), Controller::Action::SHUTDOWN);
    EXPECT_FALSE(controller_.isReloadRequested());
}

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGTERM_PreventsRestartLoop) {
    // This test verifies the fix for the critical bug:
    // If SIGHUP and SIGTERM arrive together, the daemon should NOT restart
    // (i.e., isReloadRequested should be false after shutdown)

    state_.shutdown = 1;
    state_.reload = 1;
    state_.received = SIGTERM;

    controller_.processPendingSignals();

    // The key assertion: reload should NOT be requested after SIGTERM
    // This prevents the do { ... } while (isReloadRequested()) loop from restarting
    EXPECT_FALSE(controller_.isReloadRequested());
}

// ========== Sequential Signal Tests ==========

TEST_F(GracefulShutdownTest, ProcessPendingSignals_SIGHUP_ThenSIGTERM_ShutdownWins) {
    // First call: SIGHUP
    state_.reload = 1;
    state_.received = SIGHUP;
    controller_.processPendingSignals();
    EXPECT_TRUE(controller_.isReloadRequested());

    // Reset for second signal
    controller_.setRunning(true);

    // Second call: SIGTERM
    state_.shutdown = 1;
    state_.received = SIGTERM;
    controller_.processPendingSignals();

    // SIGTERM should clear the reload request
    EXPECT_FALSE(controller_.isReloadRequested());
    EXPECT_EQ(controller_.getLastAction(), Controller::Action::SHUTDOWN);
}

// ========== Null State Tests ==========

TEST_F(GracefulShutdownTest, ProcessPendingSignals_NullState_ReturnsFalse) {
    Controller c;
    // Don't set signal state
    EXPECT_FALSE(c.processPendingSignals());
}

// ========== Global Signal State Tests ==========

TEST_F(GracefulShutdownTest, GetGlobalSignalState_ReturnsSameInstance) {
    SignalState& s1 = getGlobalSignalState();
    SignalState& s2 = getGlobalSignalState();
    EXPECT_EQ(&s1, &s2);
}

// ========== Signal Handler Tests ==========

TEST_F(GracefulShutdownTest, SignalHandler_SIGTERM_SetsShutdownFlag) {
    SignalState& global = getGlobalSignalState();
    global.reset();

    signalHandler(SIGTERM);

    EXPECT_EQ(global.shutdown, 1);
    EXPECT_EQ(global.reload, 0);
    EXPECT_EQ(global.received, SIGTERM);
}

TEST_F(GracefulShutdownTest, SignalHandler_SIGINT_SetsShutdownFlag) {
    SignalState& global = getGlobalSignalState();
    global.reset();

    signalHandler(SIGINT);

    EXPECT_EQ(global.shutdown, 1);
    EXPECT_EQ(global.reload, 0);
    EXPECT_EQ(global.received, SIGINT);
}

TEST_F(GracefulShutdownTest, SignalHandler_SIGHUP_SetsReloadFlag) {
    SignalState& global = getGlobalSignalState();
    global.reset();

    signalHandler(SIGHUP);

    EXPECT_EQ(global.shutdown, 0);
    EXPECT_EQ(global.reload, 1);
    EXPECT_EQ(global.received, SIGHUP);
}
