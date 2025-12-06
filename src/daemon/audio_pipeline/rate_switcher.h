#pragma once

#include "daemon/api/dependencies.h"
#include "daemon/api/events.h"

#include <atomic>

namespace audio_pipeline {

struct RateSwitcherDependencies {
    daemon::api::EventDispatcher* dispatcher = nullptr;
    daemon::api::DaemonDependencies deps;
    std::atomic<int>* pendingRate = nullptr;
};

class RateSwitcher {
   public:
    explicit RateSwitcher(RateSwitcherDependencies deps);

    void start();
    void handle(const daemon::api::RateChangeRequested& event);

    int lastSeenRate() const;

   private:
    RateSwitcherDependencies deps_;
    std::atomic<int> lastRate_{0};
};

}  // namespace audio_pipeline


