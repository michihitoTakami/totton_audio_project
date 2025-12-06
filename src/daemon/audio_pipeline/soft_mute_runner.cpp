#include "daemon/audio_pipeline/soft_mute_runner.h"

#include "logging/logger.h"

#include <utility>

namespace audio_pipeline {

SoftMuteRunner::SoftMuteRunner(SoftMuteRunnerDependencies deps) : deps_(std::move(deps)) {}

void SoftMuteRunner::start() {
    if (deps_.dispatcher) {
        deps_.dispatcher->subscribe([this](const daemon::api::FilterSwitchRequested& event) {
            handle(event);
        });
    }
}

void SoftMuteRunner::handle(const daemon::api::FilterSwitchRequested& event) {
    triggered_ = true;
    auto* controller = (deps_.deps.softMute) ? *deps_.deps.softMute : nullptr;
    if (controller) {
        controller->startFadeOut();
        controller->startFadeIn();
    }
    LOG_INFO("[SoftMuteRunner] Soft-mute triggered for filter switch: {}", event.filterPath);
}

bool SoftMuteRunner::wasTriggered() const {
    return triggered_;
}

}  // namespace audio_pipeline


