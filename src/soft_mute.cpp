#include "soft_mute.h"

#include <algorithm>
#include <cmath>

namespace SoftMute {

Controller::Controller(int fadeDurationMs, int sampleRate)
    : fadeDurationMs_(fadeDurationMs), sampleRate_(sampleRate) {
    updateFadeSamples();
}

void Controller::startFadeOut() {
    MuteState current = state_.load();
    if (current == MuteState::PLAYING || current == MuteState::FADING_IN) {
        fadePosition_ = 0;
        state_.store(MuteState::FADING_OUT);
    }
}

void Controller::startFadeIn() {
    MuteState current = state_.load();
    if (current == MuteState::MUTED || current == MuteState::FADING_OUT) {
        fadePosition_ = 0;
        state_.store(MuteState::FADING_IN);
    }
}

void Controller::setMuted() {
    state_.store(MuteState::MUTED);
    currentGain_.store(0.0f);
    fadePosition_ = 0;
}

void Controller::setPlaying() {
    state_.store(MuteState::PLAYING);
    currentGain_.store(1.0f);
    fadePosition_ = 0;
}

bool Controller::process(float* buffer, size_t frames) {
    if (buffer == nullptr || frames == 0) {
        return false;
    }

    MuteState currentState = state_.load();

    // Fast path: normal playback, no processing needed
    if (currentState == MuteState::PLAYING) {
        return false;
    }

    // Fast path: fully muted, zero the buffer
    if (currentState == MuteState::MUTED) {
        for (size_t i = 0; i < frames * 2; ++i) {
            buffer[i] = 0.0f;
        }
        currentGain_.store(0.0f);
        return true;
    }

    // Fade processing
    bool isFadeOut = (currentState == MuteState::FADING_OUT);

    for (size_t frame = 0; frame < frames; ++frame) {
        float gain;

        if (fadePosition_ >= fadeSamples_) {
            // Fade complete
            if (isFadeOut) {
                state_.store(MuteState::MUTED);
                gain = 0.0f;
            } else {
                state_.store(MuteState::PLAYING);
                gain = 1.0f;
            }
        } else {
            gain = calculateGain(fadePosition_, fadeSamples_, isFadeOut);
            fadePosition_++;
        }

        // Apply gain to both channels (stereo interleaved)
        size_t sampleIndex = frame * 2;
        buffer[sampleIndex] *= gain;
        buffer[sampleIndex + 1] *= gain;

        currentGain_.store(gain);
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
    fadeDurationMs_ = std::max(1, durationMs);
    updateFadeSamples();
}

int Controller::getFadeDuration() const {
    return fadeDurationMs_;
}

void Controller::setSampleRate(int sampleRate) {
    sampleRate_ = std::max(1, sampleRate);
    updateFadeSamples();
}

int Controller::getSampleRate() const {
    return sampleRate_;
}

void Controller::setFadeCurve(FadeCurve curve) {
    fadeCurve_ = curve;
}

FadeCurve Controller::getFadeCurve() const {
    return fadeCurve_;
}

float Controller::calculateGain(size_t position, size_t total, bool fadeOut) const {
    if (total == 0) {
        return fadeOut ? 0.0f : 1.0f;
    }

    float progress = static_cast<float>(position) / static_cast<float>(total);
    progress = std::clamp(progress, 0.0f, 1.0f);

    float gain;
    switch (fadeCurve_) {
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
    fadeSamples_ = calculateFadeSamples(fadeDurationMs_, sampleRate_);
    if (fadeSamples_ == 0) {
        fadeSamples_ = 1;  // Minimum 1 sample
    }
}

}  // namespace SoftMute
