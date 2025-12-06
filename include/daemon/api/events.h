#pragma once

#include "config_loader.h"
#include "convolution_engine.h"

#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace daemon_core::api {

struct RateChangeRequested {
    int detectedInputRate = 0;
    ConvolutionEngine::RateFamily rateFamily = ConvolutionEngine::RateFamily::RATE_44K;
};

struct DeviceChangeRequested {
    std::string preferredDevice;
    OutputMode mode = OutputMode::Usb;
};

struct FilterSwitchRequested {
    std::string filterPath;
    PhaseType phaseType = PhaseType::Minimum;
    bool reloadHeadroom = false;
};

class EventDispatcher {
   public:
    using RateChangeHandler = std::function<void(const RateChangeRequested&)>;
    using DeviceChangeHandler = std::function<void(const DeviceChangeRequested&)>;
    using FilterSwitchHandler = std::function<void(const FilterSwitchRequested&)>;

    void subscribe(const RateChangeHandler& handler);
    void subscribe(const DeviceChangeHandler& handler);
    void subscribe(const FilterSwitchHandler& handler);

    void publish(const RateChangeRequested& event) const;
    void publish(const DeviceChangeRequested& event) const;
    void publish(const FilterSwitchRequested& event) const;

   private:
    template <typename Event, typename Handler>
    void publishImpl(const Event& event, const std::vector<Handler>& handlers) const;

    void appendRateHandler(const RateChangeHandler& handler);
    void appendDeviceHandler(const DeviceChangeHandler& handler);
    void appendFilterHandler(const FilterSwitchHandler& handler);

    mutable std::mutex mutex_;
    std::vector<RateChangeHandler> rateHandlers_;
    std::vector<DeviceChangeHandler> deviceHandlers_;
    std::vector<FilterSwitchHandler> filterHandlers_;
};

inline void EventDispatcher::subscribe(const RateChangeHandler& handler) {
    appendRateHandler(handler);
}

inline void EventDispatcher::subscribe(const DeviceChangeHandler& handler) {
    appendDeviceHandler(handler);
}

inline void EventDispatcher::subscribe(const FilterSwitchHandler& handler) {
    appendFilterHandler(handler);
}

inline void EventDispatcher::publish(const RateChangeRequested& event) const {
    publishImpl(event, rateHandlers_);
}

inline void EventDispatcher::publish(const DeviceChangeRequested& event) const {
    publishImpl(event, deviceHandlers_);
}

inline void EventDispatcher::publish(const FilterSwitchRequested& event) const {
    publishImpl(event, filterHandlers_);
}

inline void EventDispatcher::appendRateHandler(const RateChangeHandler& handler) {
    std::lock_guard<std::mutex> lock(mutex_);
    rateHandlers_.push_back(handler);
}

inline void EventDispatcher::appendDeviceHandler(const DeviceChangeHandler& handler) {
    std::lock_guard<std::mutex> lock(mutex_);
    deviceHandlers_.push_back(handler);
}

inline void EventDispatcher::appendFilterHandler(const FilterSwitchHandler& handler) {
    std::lock_guard<std::mutex> lock(mutex_);
    filterHandlers_.push_back(handler);
}

template <typename Event, typename Handler>
void EventDispatcher::publishImpl(const Event& event, const std::vector<Handler>& handlers) const {
    std::vector<Handler> copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        copy = handlers;
    }
    for (const auto& handler : copy) {
        if (handler) {
            handler(event);
        }
    }
}

}  // namespace daemon_core::api
