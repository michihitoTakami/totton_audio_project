#include "daemon/input/rtp_input_adapter.h"

namespace daemon_input {

RtpInputAdapter::RtpInputAdapter(RtpInputAdapterDependencies deps) : deps_(std::move(deps)) {}

void RtpInputAdapter::startDiscovery() {
    lastSessionId_.clear();
}

void RtpInputAdapter::attachStream(const std::string& sessionId) {
    lastSessionId_ = sessionId;
}

void RtpInputAdapter::requestDeviceChange(const std::string& preferredDevice) {
    lastDevice_ = preferredDevice;
    if (deps_.dispatcher) {
        daemon::api::DeviceChangeRequested event;
        event.preferredDevice = preferredDevice;
        deps_.dispatcher->publish(event);
    }
}

std::string RtpInputAdapter::lastSession() const {
    return lastSessionId_;
}

std::string RtpInputAdapter::lastRequestedDevice() const {
    return lastDevice_;
}

}  // namespace daemon_input


