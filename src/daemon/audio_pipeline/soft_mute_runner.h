#pragma once

#include "daemon/api/dependencies.h"
#include "daemon/api/events.h"

namespace audio_pipeline {

struct SoftMuteRunnerDependencies {
    daemon_core::api::EventDispatcher* dispatcher = nullptr;
    daemon_core::api::DaemonDependencies deps;
};

class SoftMuteRunner {
   public:
    explicit SoftMuteRunner(SoftMuteRunnerDependencies deps);

    void start();
    void handle(const daemon_core::api::FilterSwitchRequested& event);

    bool wasTriggered() const;

   private:
    SoftMuteRunnerDependencies deps_;
    bool triggered_{false};
};

}  // namespace audio_pipeline
