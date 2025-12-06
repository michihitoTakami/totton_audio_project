#pragma once

#include "daemon/api/dependencies.h"
#include "daemon/api/events.h"

#include <string>

namespace daemon_output {

struct AlsaOutputDependencies {
    daemon_core::api::EventDispatcher* dispatcher = nullptr;
    daemon_core::api::DaemonDependencies deps;
};

class AlsaOutput {
   public:
    explicit AlsaOutput(AlsaOutputDependencies deps);

    void start();
    void handle(const daemon_core::api::DeviceChangeRequested& event);
    void markReady(bool ready);

    std::string lastRequestedDevice() const;
    bool ready() const;

   private:
    AlsaOutputDependencies deps_;
    std::string lastDevice_;
    bool ready_{false};
};

}  // namespace daemon_output
