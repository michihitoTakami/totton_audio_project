#include "daemon/output/alsa_output.h"

#include "logging/logger.h"

namespace daemon_output {

AlsaOutput::AlsaOutput(AlsaOutputDependencies deps) : deps_(std::move(deps)) {}

void AlsaOutput::start() {
    if (deps_.dispatcher) {
        deps_.dispatcher->subscribe(
            [this](const daemon_core::api::DeviceChangeRequested& event) { handle(event); });
    }
}

void AlsaOutput::handle(const daemon_core::api::DeviceChangeRequested& event) {
    lastDevice_ = event.preferredDevice;
    if (deps_.deps.outputReady) {
        deps_.deps.outputReady->store(false, std::memory_order_release);
    }
    LOG_INFO("[AlsaOutput] Device change requested: {}", event.preferredDevice);
}

void AlsaOutput::markReady(bool ready) {
    ready_ = ready;
    if (deps_.deps.outputReady) {
        deps_.deps.outputReady->store(ready, std::memory_order_release);
    }
}

std::string AlsaOutput::lastRequestedDevice() const {
    return lastDevice_;
}

bool AlsaOutput::ready() const {
    return ready_;
}

}  // namespace daemon_output
