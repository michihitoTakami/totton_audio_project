#include "hrtf/woodworth_model.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kSpeedOfSound = 343.0f;

float deg2rad(float deg) {
    return deg * static_cast<float>(M_PI) / 180.0f;
}

float clampAzimuth(float az) {
    return std::clamp(az, -80.0f, 80.0f);
}

std::vector<float> makeImpulse(size_t taps, float delaySamples, float gain) {
    std::vector<float> impulse(taps, 0.0f);
    if (taps == 0) {
        return impulse;
    }

    float clampedDelay = std::max(0.0f, delaySamples);
    auto idx = static_cast<size_t>(std::floor(clampedDelay));
    float frac = clampedDelay - static_cast<float>(idx);

    if (idx < taps) {
        impulse[idx] += gain * (1.0f - frac);
    }
    if ((idx + 1) < taps) {
        impulse[idx + 1] += gain * frac;
    }

    return impulse;
}

void applyHighShelf(std::vector<float>& data, float shelfDb) {
    if (data.empty() || std::fabs(shelfDb) < 1e-3f) {
        return;
    }
    float gainLinear = std::pow(10.0f, shelfDb / 20.0f);
    float decay = std::pow(10.0f, -std::fabs(shelfDb) / 80.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        float factor = gainLinear * std::pow(decay, static_cast<float>(i) / data.size());
        data[i] *= factor;
    }
}

std::vector<float> sumResponses(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = std::max(a.size(), b.size());
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        float va = (i < a.size()) ? a[i] : 0.0f;
        float vb = (i < b.size()) ? b[i] : 0.0f;
        out[i] = va + vb;
    }
    return out;
}

}  // namespace

namespace HRTF {

WoodworthIRSet generateWoodworthSet(float azimuthDegrees, const WoodworthParams& params) {
    WoodworthIRSet set;
    set.ll.resize(params.taps, 0.0f);
    set.lr.resize(params.taps, 0.0f);
    set.rl.resize(params.taps, 0.0f);
    set.rr.resize(params.taps, 0.0f);

    float az = clampAzimuth(azimuthDegrees);
    float azRad = deg2rad(az);
    float radius = std::max(0.05f, params.headRadiusMeters);
    float sampleRate = std::max(8000.0f, params.sampleRate);

    bool sourceLeft = az >= 0.0f;
    float itdNear = (radius / kSpeedOfSound) * std::sin(std::fabs(azRad));
    float itdFar = (radius / kSpeedOfSound) * (std::fabs(azRad) + std::sin(std::fabs(azRad)));

    float nearDelaySamples = itdNear * sampleRate;
    float farDelaySamples = itdFar * sampleRate;

    float farShadowGain = std::pow(10.0f, params.farEarShadowDb / 20.0f);

    std::vector<float> nearImpulse = makeImpulse(params.taps, nearDelaySamples, 1.0f);
    std::vector<float> farImpulse = makeImpulse(params.taps, farDelaySamples, farShadowGain);
    applyHighShelf(farImpulse, params.diffuseFieldTiltDb);

    if (sourceLeft) {
        set.ll = nearImpulse;
        set.lr = farImpulse;
        set.rr = nearImpulse;
        set.rl = farImpulse;
    } else {
        set.ll = farImpulse;
        set.lr = nearImpulse;
        set.rr = farImpulse;
        set.rl = nearImpulse;
    }

    float bleedGain = 0.02f;
    auto bleed =
        makeImpulse(params.taps, nearDelaySamples + params.sampleRate * 0.0005f, bleedGain);
    set.ll = sumResponses(set.ll, bleed);
    set.rr = sumResponses(set.rr, bleed);

    return set;
}

}  // namespace HRTF
