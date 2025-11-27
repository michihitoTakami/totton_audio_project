#include "phase_alignment.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kEpsilon = 1e-9f;

double modifiedBesselI0(double x) {
    double result = 1.0;
    double term = 1.0;
    double halfX = x * 0.5;
    double halfXSq = halfX * halfX;
    double m = 1.0;
    while (term > 1e-12) {
        term *= halfXSq / (m * m);
        result += term;
        ++m;
    }
    return result;
}

float kaiserWindow(int n, int length, float beta) {
    if (length <= 1) {
        return 1.0f;
    }
    double ratio = (2.0 * n) / (length - 1) - 1.0;
    double denom = modifiedBesselI0(beta);
    double numer = modifiedBesselI0(beta * std::sqrt(1.0 - ratio * ratio));
    return static_cast<float>(numer / denom);
}

float sinc(float x) {
    if (std::fabs(x) < 1e-6f) {
        return 1.0f;
    }
    float pix = static_cast<float>(M_PI) * x;
    return std::sin(pix) / pix;
}

}  // namespace

namespace PhaseAlignment {

float computeEnergyCentroid(const std::vector<float>& impulse) {
    if (impulse.empty()) {
        return 0.0f;
    }

    long double numerator = 0.0;
    long double denominator = 0.0;
    for (size_t i = 0; i < impulse.size(); ++i) {
        long double sample = impulse[i];
        long double energy = sample * sample;
        numerator += static_cast<long double>(i) * energy;
        denominator += energy;
    }

    if (denominator < kEpsilon) {
        return 0.0f;
    }

    return static_cast<float>(numerator / denominator);
}

FractionalDelayLine::FractionalDelayLine()
    : delaySamples_(0.0f), kernelRadius_(12), beta_(8.6f), kernelDirty_(true) {}

void FractionalDelayLine::configure(float delaySamples, int kernelRadius, float beta) {
    delaySamples_ = std::max(0.0f, delaySamples);
    kernelRadius_ = std::max(1, kernelRadius);
    beta_ = std::max(0.0f, beta);
    kernelDirty_ = true;
}

void FractionalDelayLine::reset() {
    history_.clear();
    history_.shrink_to_fit();
}

bool FractionalDelayLine::isBypassed() const {
    return delaySamples_ < 1e-4f;
}

void FractionalDelayLine::rebuildKernel() {
    int taps = kernelRadius_ * 2 + 1;
    kernel_.resize(taps);
    for (int n = 0; n < taps; ++n) {
        // Positive delaySamples_ shifts output later (adds delay)
        float x = static_cast<float>(n - kernelRadius_) + delaySamples_;
        float win = kaiserWindow(n, taps, beta_);
        kernel_[n] = sinc(x) * win;
    }
    kernelDirty_ = false;

    if (history_.size() != static_cast<size_t>(taps - 1)) {
        history_.assign(taps - 1, 0.0f);
    } else {
        std::fill(history_.begin(), history_.end(), 0.0f);
    }
}

}  // namespace PhaseAlignment
