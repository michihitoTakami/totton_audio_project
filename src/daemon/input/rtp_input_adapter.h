#pragma once

#include "daemon/api/events.h"

#include <string>

namespace daemon_input {

struct RtpInputAdapterDependencies {
    daemon_core::api::EventDispatcher* dispatcher = nullptr;
};

class RtpInputAdapter {
   public:
    explicit RtpInputAdapter(RtpInputAdapterDependencies deps);

    void startDiscovery();
    void attachStream(const std::string& sessionId);
    void requestDeviceChange(const std::string& preferredDevice);

    std::string lastSession() const;
    std::string lastRequestedDevice() const;

   private:
    RtpInputAdapterDependencies deps_;
    std::string lastSessionId_;
    std::string lastDevice_;
};

}  // namespace daemon_input
