#ifndef SOFT_MUTE_H
#define SOFT_MUTE_H

#include <atomic>
#include <cstddef>

namespace SoftMute {

// Mute state machine
enum class MuteState {
    PLAYING,     // Normal playback, gain = 1.0
    FADING_OUT,  // Transitioning to muted state
    MUTED,       // Silent, gain = 0.0
    FADING_IN    // Transitioning back to playing
};

// Fade curve type
enum class FadeCurve {
    LINEAR,      // Linear fade (simple but can sound abrupt)
    LOGARITHMIC  // Logarithmic fade (more natural for audio)
};

// Soft mute controller for glitch-free audio transitions
// Thread-safe for use with audio processing threads
class Controller {
   public:
    // Constructor
    // fadeDurationMs: Duration of fade in milliseconds (default: 50ms)
    // sampleRate: Audio sample rate for calculating fade samples
    explicit Controller(int fadeDurationMs = 50, int sampleRate = 44100);

    // Start fade-out transition (PLAYING -> FADING_OUT -> MUTED)
    void startFadeOut();

    // Start fade-in transition (MUTED -> FADING_IN -> PLAYING)
    void startFadeIn();

    // Immediately set to muted state (no fade)
    void setMuted();

    // Immediately set to playing state (no fade)
    void setPlaying();

    // Process audio buffer, applying gain based on current state
    // Returns true if any processing was done (state != PLAYING with gain 1.0)
    // buffer: Interleaved stereo audio samples (modified in place)
    // frames: Number of frames (each frame = 2 samples for stereo)
    bool process(float* buffer, size_t frames);

    // Get current mute state
    MuteState getState() const;

    // Get current gain value (0.0 to 1.0)
    float getCurrentGain() const;

    // Check if currently in a transition state
    bool isTransitioning() const;

    // Check if audio output is silent (MUTED or FADING_OUT near end)
    bool isSilent() const;

    // Configuration
    void setFadeDuration(int durationMs);
    int getFadeDuration() const;

    void setSampleRate(int sampleRate);
    int getSampleRate() const;

    void setFadeCurve(FadeCurve curve);
    FadeCurve getFadeCurve() const;

   private:
    // Calculate gain for current position in fade
    float calculateGain(size_t position, size_t total, bool fadeOut) const;

    // Thread safety model:
    // - All member variables are atomic for safe concurrent access from:
    //   - Control thread
    //   (startFadeOut/startFadeIn/setMuted/setPlaying/setFadeDuration/setSampleRate)
    //   - Audio thread (process)
    // - setFadeDuration/setSampleRate can be called at any time without stopping audio
    std::atomic<MuteState> state_{MuteState::PLAYING};
    std::atomic<float> currentGain_{1.0f};
    std::atomic<size_t> fadePosition_{0};

    // Fade parameters (thread-safe via atomic)
    std::atomic<int> fadeDurationMs_{50};
    std::atomic<int> sampleRate_{44100};
    std::atomic<size_t> fadeSamples_{0};
    std::atomic<FadeCurve> fadeCurve_{FadeCurve::LINEAR};

    // Recalculate fade samples when parameters change
    void updateFadeSamples();
};

// Convenience function: Calculate fade duration in samples
inline size_t calculateFadeSamples(int durationMs, int sampleRate) {
    return static_cast<size_t>(durationMs * sampleRate / 1000);
}

}  // namespace SoftMute

#endif  // SOFT_MUTE_H
