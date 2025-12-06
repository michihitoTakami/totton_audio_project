#include "daemon/audio_pipeline/rate_switcher.h"

#include "logging/logger.h"

#include <utility>

namespace audio_pipeline {

RateSwitcher::RateSwitcher(RateSwitcherDependencies deps) : deps_(std::move(deps)) {}

void RateSwitcher::start() {
    if (deps_.dispatcher) {
        deps_.dispatcher->subscribe(
            [this](const daemon_core::api::RateChangeRequested& event) { handle(event); });
    }
}

void RateSwitcher::handle(const daemon_core::api::RateChangeRequested& event) {
    lastRate_.store(event.detectedInputRate, std::memory_order_release);

    if (deps_.pendingRate) {
        deps_.pendingRate->store(event.detectedInputRate, std::memory_order_release);
    }
    if (deps_.deps.currentInputRate) {
        deps_.deps.currentInputRate->store(event.detectedInputRate, std::memory_order_release);
    }
    if (deps_.deps.currentOutputRate) {
        deps_.deps.currentOutputRate->store(event.detectedInputRate, std::memory_order_release);
    }

    LOG_INFO("[RateSwitcher] Requested rate change: {} Hz (family={})", event.detectedInputRate,
             static_cast<int>(event.rateFamily));
}

int RateSwitcher::lastSeenRate() const {
    return lastRate_.load(std::memory_order_acquire);
}

}  // namespace audio_pipeline
