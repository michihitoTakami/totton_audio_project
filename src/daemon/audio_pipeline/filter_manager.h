#pragma once

#include "daemon/api/dependencies.h"
#include "daemon/api/events.h"

#include <string>

namespace audio_pipeline {

struct FilterManagerDependencies {
    daemon_core::api::EventDispatcher* dispatcher = nullptr;
    daemon_core::api::DaemonDependencies deps;
};

class FilterManager {
   public:
    explicit FilterManager(FilterManagerDependencies deps);

    void start();
    void handle(const daemon_core::api::FilterSwitchRequested& event);

    std::string lastRequestedPath() const;
    PhaseType lastRequestedPhase() const;

   private:
    FilterManagerDependencies deps_;
    std::string lastPath_;
    PhaseType lastPhase_{PhaseType::Minimum};
};

}  // namespace audio_pipeline
