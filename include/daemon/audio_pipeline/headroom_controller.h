#pragma once

#include "audio/filter_headroom.h"
#include "core/config_loader.h"
#include "daemon/api/events.h"

#include <atomic>
#include <functional>
#include <string>

namespace audio_pipeline {

struct HeadroomControllerDependencies {
    daemon_core::api::EventDispatcher* dispatcher = nullptr;
    AppConfig* config = nullptr;
    FilterHeadroomCache* headroomCache = nullptr;
    std::atomic<float>* headroomGain = nullptr;
    std::atomic<float>* outputGain = nullptr;
    std::atomic<float>* effectiveGain = nullptr;
    std::function<ConvolutionEngine::RateFamily()> activeRateFamily;
    std::function<PhaseType()> activePhaseType;
};

class HeadroomController {
   public:
    explicit HeadroomController(HeadroomControllerDependencies deps);

    void start();
    void handle(const daemon_core::api::FilterSwitchRequested& event);

    std::string currentFilterPath() const;
    void refreshCurrentHeadroom(const std::string& reason) const;
    void setTargetPeak(float targetPeak);
    void resetEffectiveGain(const std::string& reason) const;

   private:
    std::string resolveFilterPathFor(ConvolutionEngine::RateFamily family, PhaseType phase) const;
    void updateEffectiveGain(float headroomGain, const std::string& reason) const;
    void applyHeadroomForPath(const std::string& path, const std::string& reason) const;

    HeadroomControllerDependencies deps_;
};

}  // namespace audio_pipeline
