#pragma once

#include "daemon/api/dependencies.h"
#include "daemon/api/events.h"

namespace audio_pipeline {

struct SoftMuteRunnerDependencies {
    daemon::api::EventDispatcher* dispatcher = nullptr;
    daemon::api::DaemonDependencies deps;
};

class SoftMuteRunner {
   public:
    explicit SoftMuteRunner(SoftMuteRunnerDependencies deps);

    void start();
    void handle(const daemon::api::FilterSwitchRequested& event);

    bool wasTriggered() const;

   private:
    SoftMuteRunnerDependencies deps_;
    bool triggered_{false};
};

}  // namespace audio_pipeline


