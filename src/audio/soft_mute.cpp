#include "audio/soft_mute.h"

#include <algorithm>
#include <cmath>

namespace SoftMute {

Controller::Controller(int fadeDurationMs, int sampleRate) {
    std::lock_guard<std::mutex> lock(configMutex_);
    fadeDurationMs_.store(fadeDurationMs, std::memory_order_relaxed);
    sampleRate_.store(sampleRate, std::memory_order_relaxed);
    updateFadeSamples();
}

void Controller::startFadeOut() {
    MuteState current = state_.load(std::memory_order_acquire);
    if (current == MuteState::PLAYING || current == MuteState::FADING_IN) {
        fadePosition_.store(0, std::memory_order_relaxed);
        state_.store(MuteState::FADING_OUT, std::memory_order_release);
    }
}

void Controller::startFadeIn() {
    MuteState current = state_.load(std::memory_order_acquire);
    if (current == MuteState::MUTED || current == MuteState::FADING_OUT) {
        fadePosition_.store(0, std::memory_order_relaxed);
        state_.store(MuteState::FADING_IN, std::memory_order_release);
    }
}

void Controller::setMuted() {
    fadePosition_.store(0, std::memory_order_relaxed);
    state_.store(MuteState::MUTED, std::memory_order_release);
    currentGain_.store(0.0f, std::memory_order_relaxed);
}

void Controller::setPlaying() {
    fadePosition_.store(0, std::memory_order_relaxed);
    state_.store(MuteState::PLAYING, std::memory_order_release);
    currentGain_.store(1.0f, std::memory_order_relaxed);
}

bool Controller::process(float* buffer, size_t frames) {
    if (buffer == nullptr || frames == 0) {
        return false;
    }

    MuteState currentState = state_.load(std::memory_order_acquire);

    // Fast path: normal playback, no processing needed
    if (currentState == MuteState::PLAYING) {
        return false;
    }

    // Fast path: fully muted, zero the buffer
    if (currentState == MuteState::MUTED) {
        for (size_t i = 0; i < frames * 2; ++i) {
            buffer[i] = 0.0f;
        }
        currentGain_.store(0.0f, std::memory_order_relaxed);
        return true;
    }

    // Fade processing
    bool isFadeOut = (currentState == MuteState::FADING_OUT);

    for (size_t frame = 0; frame < frames; ++frame) {
        float gain;
        size_t pos = fadePosition_.load(std::memory_order_relaxed);
        size_t fadeSamples = fadeSamples_.load(std::memory_order_relaxed);
        if (pos >= fadeSamples) {
            // Fade complete
            if (isFadeOut) {
                state_.store(MuteState::MUTED, std::memory_order_release);
                gain = 0.0f;
            } else {
                state_.store(MuteState::PLAYING, std::memory_order_release);
                gain = 1.0f;
            }
        } else {
            gain = calculateGain(pos, fadeSamples, isFadeOut);
            fadePosition_.fetch_add(1, std::memory_order_relaxed);
        }

        // Apply gain to both channels (stereo interleaved)
        size_t sampleIndex = frame * 2;
        buffer[sampleIndex] *= gain;
        buffer[sampleIndex + 1] *= gain;

        currentGain_.store(gain, std::memory_order_relaxed);
    }

    return true;
}

MuteState Controller::getState() const {
    return state_.load();
}

float Controller::getCurrentGain() const {
    return currentGain_.load();
}

bool Controller::isTransitioning() const {
    MuteState s = state_.load();
    return s == MuteState::FADING_OUT || s == MuteState::FADING_IN;
}

bool Controller::isSilent() const {
    MuteState s = state_.load();
    return s == MuteState::MUTED || (s == MuteState::FADING_OUT && currentGain_.load() < 0.01f);
}

void Controller::setFadeDuration(int durationMs) {
    std::lock_guard<std::mutex> lock(configMutex_);
    fadeDurationMs_.store(std::max(1, durationMs), std::memory_order_relaxed);
    updateFadeSamples();
}

int Controller::getFadeDuration() const {
    return fadeDurationMs_.load(std::memory_order_relaxed);
}

void Controller::setSampleRate(int sampleRate) {
    std::lock_guard<std::mutex> lock(configMutex_);
    sampleRate_.store(std::max(1, sampleRate), std::memory_order_relaxed);
    updateFadeSamples();
}

int Controller::getSampleRate() const {
    return sampleRate_.load(std::memory_order_relaxed);
}

void Controller::setFadeCurve(FadeCurve curve) {
    std::lock_guard<std::mutex> lock(configMutex_);
    fadeCurve_.store(curve, std::memory_order_relaxed);
}

FadeCurve Controller::getFadeCurve() const {
    return fadeCurve_.load(std::memory_order_relaxed);
}

float Controller::calculateGain(size_t position, size_t total, bool fadeOut) const {
    if (total == 0) {
        return fadeOut ? 0.0f : 1.0f;
    }

    float progress = static_cast<float>(position) / static_cast<float>(total);
    progress = std::clamp(progress, 0.0f, 1.0f);

    float gain;
    FadeCurve curve = fadeCurve_.load(std::memory_order_relaxed);
    switch (curve) {
    case FadeCurve::LINEAR:
        // Linear: gain increases linearly with progress
        // For fade-out: start at 1.0, end at 0.0
        // For fade-in: start at 0.0, end at 1.0
        gain = fadeOut ? (1.0f - progress) : progress;
        break;

    case FadeCurve::LOGARITHMIC:
        // Logarithmic/power curve for more natural audio perception
        // For fade-out: sqrt curve (faster initial drop, slower at end)
        // For fade-in: square curve (slower start, faster at end)
        if (fadeOut) {
            gain = std::sqrt(1.0f - progress);
        } else {
            gain = progress * progress;
        }
        break;

    default:
        gain = fadeOut ? (1.0f - progress) : progress;
        break;
    }

    return std::clamp(gain, 0.0f, 1.0f);
}

void Controller::updateFadeSamples() {
    int durationMs = fadeDurationMs_.load(std::memory_order_relaxed);
    int rate = sampleRate_.load(std::memory_order_relaxed);
    size_t samples = calculateFadeSamples(durationMs, rate);
    if (samples == 0) {
        samples = 1;  // Minimum 1 sample
    }
    fadeSamples_.store(samples, std::memory_order_release);
}

}  // namespace SoftMute
