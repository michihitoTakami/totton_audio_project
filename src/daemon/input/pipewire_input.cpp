#include "daemon/input/pipewire_input.h"

namespace daemon_input {

PipeWireInput::PipeWireInput(PipeWireInputDependencies deps) : deps_(std::move(deps)) {}

void PipeWireInput::start() {
    running_.store(true, std::memory_order_release);
    if (deps_.runningFlag) {
        deps_.runningFlag->store(true, std::memory_order_release);
    }
}

void PipeWireInput::stop() {
    running_.store(false, std::memory_order_release);
    if (deps_.runningFlag) {
        deps_.runningFlag->store(false, std::memory_order_release);
    }
}

void PipeWireInput::publishRateChange(int detectedRate, ConvolutionEngine::RateFamily family) const {
    if (deps_.pendingRate) {
        deps_.pendingRate->store(detectedRate, std::memory_order_release);
    }
    if (deps_.dispatcher) {
        daemon::api::RateChangeRequested event;
        event.detectedInputRate = detectedRate;
        event.rateFamily = family;
        deps_.dispatcher->publish(event);
    }
}

bool PipeWireInput::isRunning() const {
    return running_.load(std::memory_order_acquire);
}

}  // namespace daemon_input


