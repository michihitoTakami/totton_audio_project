#pragma once

#include "daemon/api/events.h"

#include <atomic>

namespace daemon_input {

struct PipeWireInputDependencies {
    daemon::api::EventDispatcher* dispatcher = nullptr;
    std::atomic<bool>* runningFlag = nullptr;
    std::atomic<int>* pendingRate = nullptr;
};

class PipeWireInput {
   public:
    explicit PipeWireInput(PipeWireInputDependencies deps);

    void start();
    void stop();
    void publishRateChange(int detectedRate, ConvolutionEngine::RateFamily family) const;

    bool isRunning() const;

   private:
    PipeWireInputDependencies deps_;
    std::atomic<bool> running_{false};
};

}  // namespace daemon_input


