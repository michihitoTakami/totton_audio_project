#ifndef HRTF_WOODWORTH_MODEL_H
#define HRTF_WOODWORTH_MODEL_H

#include <cstddef>
#include <vector>

namespace HRTF {

struct WoodworthParams {
    float headRadiusMeters = 0.09f;
    float earSpacingMeters = 0.18f;
    float sampleRate = 48000.0f;
    size_t taps = 1024;
    float farEarShadowDb = -8.0f;
    float diffuseFieldTiltDb = -2.0f;
};

struct WoodworthIRSet {
    std::vector<float> ll;
    std::vector<float> lr;
    std::vector<float> rl;
    std::vector<float> rr;
};

WoodworthIRSet generateWoodworthSet(float azimuthDegrees, const WoodworthParams& params);

}  // namespace HRTF

#endif  // HRTF_WOODWORTH_MODEL_H

