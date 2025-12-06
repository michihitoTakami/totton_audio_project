#include "daemon/audio_pipeline/filter_manager.h"

#include "logging/logger.h"

#include <utility>

namespace audio_pipeline {

FilterManager::FilterManager(FilterManagerDependencies deps) : deps_(std::move(deps)) {}

void FilterManager::start() {
    if (deps_.dispatcher) {
        deps_.dispatcher->subscribe(
            [this](const daemon_core::api::FilterSwitchRequested& event) { handle(event); });
    }
}

void FilterManager::handle(const daemon_core::api::FilterSwitchRequested& event) {
    lastPath_ = event.filterPath;
    lastPhase_ = event.phaseType;

    if (deps_.deps.refreshHeadroom) {
        deps_.deps.refreshHeadroom(event.filterPath);
    }

    LOG_INFO("[FilterManager] Filter switch requested: path={}, phase={}, reloadHeadroom={}",
             event.filterPath, static_cast<int>(event.phaseType), event.reloadHeadroom);
}

std::string FilterManager::lastRequestedPath() const {
    return lastPath_;
}

PhaseType FilterManager::lastRequestedPhase() const {
    return lastPhase_;
}

}  // namespace audio_pipeline
