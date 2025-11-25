/**
 * @file test_soft_mute.cpp
 * @brief Unit tests for Soft Mute controller (CPU-only, no GPU required)
 *
 * Tests the soft mute functionality for glitch-free audio transitions.
 */

#include "soft_mute.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace SoftMute;

class SoftMuteTest : public ::testing::Test {
   protected:
    // Create a test buffer filled with 1.0f
    std::vector<float> createTestBuffer(size_t frames) {
        std::vector<float> buffer(frames * 2, 1.0f);  // Stereo interleaved
        return buffer;
    }

    // Check if buffer is all zeros
    bool isBufferSilent(const std::vector<float>& buffer) {
        for (float sample : buffer) {
            if (std::abs(sample) > 0.0001f) {
                return false;
            }
        }
        return true;
    }

    // Check if buffer is all ones (unchanged)
    bool isBufferUnchanged(const std::vector<float>& buffer) {
        for (float sample : buffer) {
            if (std::abs(sample - 1.0f) > 0.0001f) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Basic State Tests
// ============================================================================

TEST_F(SoftMuteTest, InitialState_IsPlaying) {
    Controller ctrl;
    EXPECT_EQ(ctrl.getState(), MuteState::PLAYING);
    EXPECT_FLOAT_EQ(ctrl.getCurrentGain(), 1.0f);
}

TEST_F(SoftMuteTest, SetMuted_ImmediatelyMutes) {
    Controller ctrl;
    ctrl.setMuted();
    EXPECT_EQ(ctrl.getState(), MuteState::MUTED);
    EXPECT_FLOAT_EQ(ctrl.getCurrentGain(), 0.0f);
}

TEST_F(SoftMuteTest, SetPlaying_ImmediatelyPlays) {
    Controller ctrl;
    ctrl.setMuted();
    ctrl.setPlaying();
    EXPECT_EQ(ctrl.getState(), MuteState::PLAYING);
    EXPECT_FLOAT_EQ(ctrl.getCurrentGain(), 1.0f);
}

// ============================================================================
// Fade-Out Tests
// ============================================================================

TEST_F(SoftMuteTest, StartFadeOut_TransitionsToFadingOut) {
    Controller ctrl;
    ctrl.startFadeOut();
    EXPECT_EQ(ctrl.getState(), MuteState::FADING_OUT);
    EXPECT_TRUE(ctrl.isTransitioning());
}

TEST_F(SoftMuteTest, FadeOut_CompletesToMuted) {
    Controller ctrl(10, 44100);  // 10ms fade at 44.1kHz = 441 samples
    ctrl.startFadeOut();

    auto buffer = createTestBuffer(500);  // More than fade duration
    ctrl.process(buffer.data(), 500);

    EXPECT_EQ(ctrl.getState(), MuteState::MUTED);
    EXPECT_FLOAT_EQ(ctrl.getCurrentGain(), 0.0f);
}

TEST_F(SoftMuteTest, FadeOut_GainDecreasesMonotonically) {
    Controller ctrl(50, 44100);  // 50ms fade
    ctrl.startFadeOut();

    float lastGain = 1.0f;
    auto buffer = createTestBuffer(100);

    for (int i = 0; i < 30; ++i) {
        ctrl.process(buffer.data(), 100);
        float currentGain = ctrl.getCurrentGain();
        EXPECT_LE(currentGain, lastGain) << "Gain should decrease during fade-out";
        lastGain = currentGain;
    }
}

// ============================================================================
// Fade-In Tests
// ============================================================================

TEST_F(SoftMuteTest, StartFadeIn_FromMuted_TransitionsToFadingIn) {
    Controller ctrl;
    ctrl.setMuted();
    ctrl.startFadeIn();
    EXPECT_EQ(ctrl.getState(), MuteState::FADING_IN);
    EXPECT_TRUE(ctrl.isTransitioning());
}

TEST_F(SoftMuteTest, FadeIn_CompletesToPlaying) {
    Controller ctrl(10, 44100);  // 10ms fade
    ctrl.setMuted();
    ctrl.startFadeIn();

    auto buffer = createTestBuffer(500);
    ctrl.process(buffer.data(), 500);

    EXPECT_EQ(ctrl.getState(), MuteState::PLAYING);
    EXPECT_FLOAT_EQ(ctrl.getCurrentGain(), 1.0f);
}

TEST_F(SoftMuteTest, FadeIn_GainIncreasesMonotonically) {
    Controller ctrl(50, 44100);
    ctrl.setMuted();
    ctrl.startFadeIn();

    float lastGain = 0.0f;
    auto buffer = createTestBuffer(100);

    for (int i = 0; i < 30; ++i) {
        ctrl.process(buffer.data(), 100);
        float currentGain = ctrl.getCurrentGain();
        EXPECT_GE(currentGain, lastGain) << "Gain should increase during fade-in";
        lastGain = currentGain;
    }
}

// ============================================================================
// Buffer Processing Tests
// ============================================================================

TEST_F(SoftMuteTest, Process_PlayingState_NoChange) {
    Controller ctrl;
    auto buffer = createTestBuffer(100);

    bool processed = ctrl.process(buffer.data(), 100);

    EXPECT_FALSE(processed);  // No processing needed
    EXPECT_TRUE(isBufferUnchanged(buffer));
}

TEST_F(SoftMuteTest, Process_MutedState_ZerosBuffer) {
    Controller ctrl;
    ctrl.setMuted();
    auto buffer = createTestBuffer(100);

    bool processed = ctrl.process(buffer.data(), 100);

    EXPECT_TRUE(processed);
    EXPECT_TRUE(isBufferSilent(buffer));
}

TEST_F(SoftMuteTest, Process_FadeOut_AppliesDecreasingGain) {
    Controller ctrl(10, 44100);
    ctrl.startFadeOut();
    auto buffer = createTestBuffer(100);

    ctrl.process(buffer.data(), 100);

    // First samples should be close to 1.0, later samples should be lower
    EXPECT_GT(buffer[0], buffer[198]);  // First > Last
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(SoftMuteTest, SetFadeDuration_UpdatesCorrectly) {
    Controller ctrl;
    ctrl.setFadeDuration(100);
    EXPECT_EQ(ctrl.getFadeDuration(), 100);
}

TEST_F(SoftMuteTest, SetSampleRate_UpdatesCorrectly) {
    Controller ctrl;
    ctrl.setSampleRate(48000);
    EXPECT_EQ(ctrl.getSampleRate(), 48000);
}

TEST_F(SoftMuteTest, SetFadeCurve_UpdatesCorrectly) {
    Controller ctrl;
    ctrl.setFadeCurve(FadeCurve::LOGARITHMIC);
    EXPECT_EQ(ctrl.getFadeCurve(), FadeCurve::LOGARITHMIC);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SoftMuteTest, Process_NullBuffer_ReturnsFalse) {
    Controller ctrl;
    bool processed = ctrl.process(nullptr, 100);
    EXPECT_FALSE(processed);
}

TEST_F(SoftMuteTest, Process_ZeroFrames_ReturnsFalse) {
    Controller ctrl;
    auto buffer = createTestBuffer(100);
    bool processed = ctrl.process(buffer.data(), 0);
    EXPECT_FALSE(processed);
}

TEST_F(SoftMuteTest, StartFadeOut_WhileMuted_StaysMuted) {
    Controller ctrl;
    ctrl.setMuted();
    ctrl.startFadeOut();  // Should be ignored
    EXPECT_EQ(ctrl.getState(), MuteState::MUTED);
}

TEST_F(SoftMuteTest, StartFadeIn_WhilePlaying_StaysPlaying) {
    Controller ctrl;
    ctrl.startFadeIn();  // Should be ignored
    EXPECT_EQ(ctrl.getState(), MuteState::PLAYING);
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST_F(SoftMuteTest, CalculateFadeSamples_CorrectCalculation) {
    // 50ms at 44100Hz = 2205 samples
    EXPECT_EQ(calculateFadeSamples(50, 44100), 2205);

    // 100ms at 48000Hz = 4800 samples
    EXPECT_EQ(calculateFadeSamples(100, 48000), 4800);

    // 1ms at 44100Hz = 44 samples
    EXPECT_EQ(calculateFadeSamples(1, 44100), 44);
}

// ============================================================================
// IsSilent Tests
// ============================================================================

TEST_F(SoftMuteTest, IsSilent_WhenMuted) {
    Controller ctrl;
    ctrl.setMuted();
    EXPECT_TRUE(ctrl.isSilent());
}

TEST_F(SoftMuteTest, IsSilent_WhenPlaying) {
    Controller ctrl;
    EXPECT_FALSE(ctrl.isSilent());
}

// ============================================================================
// Filter Switch Tests (Issue #266)
// ============================================================================

TEST_F(SoftMuteTest, FilterSwitch_FadeDuration_CanBeChanged) {
    Controller ctrl(50, 44100);  // Initial: 50ms
    EXPECT_EQ(ctrl.getFadeDuration(), 50);
    
    // Change to filter switch duration (1500ms)
    ctrl.setFadeDuration(1500);
    EXPECT_EQ(ctrl.getFadeDuration(), 1500);
    
    // Reset to default
    ctrl.setFadeDuration(50);
    EXPECT_EQ(ctrl.getFadeDuration(), 50);
}

TEST_F(SoftMuteTest, FilterSwitch_FadeOut_1500ms_Completes) {
    Controller ctrl(1500, 44100);  // 1500ms fade for filter switching
    ctrl.startFadeOut();
    
    EXPECT_EQ(ctrl.getState(), MuteState::FADING_OUT);
    
    // Process enough samples to complete fade-out
    // 1500ms at 44100Hz = 66150 samples
    // Process in chunks to simulate real-time processing
    auto buffer = createTestBuffer(1024);
    size_t totalProcessed = 0;
    const size_t targetSamples = 70000;  // Slightly more than needed
    
    while (totalProcessed < targetSamples && ctrl.isTransitioning()) {
        ctrl.process(buffer.data(), 1024);
        totalProcessed += 1024;
    }
    
    EXPECT_EQ(ctrl.getState(), MuteState::MUTED);
    EXPECT_FLOAT_EQ(ctrl.getCurrentGain(), 0.0f);
}

TEST_F(SoftMuteTest, FilterSwitch_FadeIn_1500ms_Completes) {
    Controller ctrl(1500, 44100);  // 1500ms fade for filter switching
    ctrl.setMuted();
    ctrl.startFadeIn();
    
    EXPECT_EQ(ctrl.getState(), MuteState::FADING_IN);
    
    // Process enough samples to complete fade-in
    auto buffer = createTestBuffer(1024);
    size_t totalProcessed = 0;
    const size_t targetSamples = 70000;  // Slightly more than needed
    
    while (totalProcessed < targetSamples && ctrl.isTransitioning()) {
        ctrl.process(buffer.data(), 1024);
        totalProcessed += 1024;
    }
    
    EXPECT_EQ(ctrl.getState(), MuteState::PLAYING);
    EXPECT_FLOAT_EQ(ctrl.getCurrentGain(), 1.0f);
}

TEST_F(SoftMuteTest, FilterSwitch_CompleteCycle_FadeOutSwitchFadeIn) {
    Controller ctrl(1500, 44100);  // 1500ms fade for filter switching
    
    // Step 1: Fade-out
    ctrl.startFadeOut();
    auto buffer = createTestBuffer(1024);
    size_t processed = 0;
    while (ctrl.isTransitioning() && processed < 70000) {
        ctrl.process(buffer.data(), 1024);
        processed += 1024;
    }
    EXPECT_EQ(ctrl.getState(), MuteState::MUTED);
    
    // Step 2: Filter switch would happen here (simulated by staying muted)
    // In real implementation, filter switch happens between fade-out and fade-in
    
    // Step 3: Fade-in
    ctrl.startFadeIn();
    processed = 0;
    while (ctrl.isTransitioning() && processed < 70000) {
        ctrl.process(buffer.data(), 1024);
        processed += 1024;
    }
    EXPECT_EQ(ctrl.getState(), MuteState::PLAYING);
    EXPECT_FLOAT_EQ(ctrl.getCurrentGain(), 1.0f);
}

TEST_F(SoftMuteTest, FilterSwitch_FadeDuration_ResetAfterCompletion) {
    Controller ctrl(50, 44100);  // Default: 50ms
    
    // Change to filter switch duration
    ctrl.setFadeDuration(1500);
    EXPECT_EQ(ctrl.getFadeDuration(), 1500);
    
    // Perform fade-out and fade-in
    ctrl.startFadeOut();
    auto buffer = createTestBuffer(1024);
    size_t processed = 0;
    while (ctrl.isTransitioning() && processed < 70000) {
        ctrl.process(buffer.data(), 1024);
        processed += 1024;
    }
    
    ctrl.startFadeIn();
    processed = 0;
    while (ctrl.isTransitioning() && processed < 70000) {
        ctrl.process(buffer.data(), 1024);
        processed += 1024;
    }
    
    // After completion, fade duration should still be 1500ms
    // (reset happens in audio processing thread, not in controller)
    EXPECT_EQ(ctrl.getFadeDuration(), 1500);
    
    // Manually reset to default (simulating audio thread behavior)
    ctrl.setFadeDuration(50);
    EXPECT_EQ(ctrl.getFadeDuration(), 50);
}

TEST_F(SoftMuteTest, FilterSwitch_SampleRate_UpdateDuringTransition) {
    Controller ctrl(1500, 44100);  // 1500ms fade at 44.1kHz
    
    // Start fade-out
    ctrl.startFadeOut();
    EXPECT_EQ(ctrl.getSampleRate(), 44100);
    
    // Update sample rate (simulating output rate change during filter switch)
    ctrl.setSampleRate(48000);
    EXPECT_EQ(ctrl.getSampleRate(), 48000);
    
    // Fade should continue with new sample rate
    EXPECT_TRUE(ctrl.isTransitioning());
}
