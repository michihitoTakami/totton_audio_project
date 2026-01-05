#include "daemon/audio_pipeline/headroom_controller.h"

#include "logging/logger.h"

#include <utility>
#include <vector>

namespace audio_pipeline {

HeadroomController::HeadroomController(HeadroomControllerDependencies deps)
    : deps_(std::move(deps)) {}

void HeadroomController::start() {
    if (deps_.dispatcher) {
        deps_.dispatcher->subscribe(
            [this](const daemon_core::api::FilterSwitchRequested& event) { handle(event); });
    }
}

void HeadroomController::handle(const daemon_core::api::FilterSwitchRequested& event) {
    if (!event.reloadHeadroom) {
        return;
    }

    applyHeadroomForPath(event.filterPath, "filter switch");
}

std::string HeadroomController::currentFilterPath() const {
    if (!deps_.config) {
        return {};
    }

    ConvolutionEngine::RateFamily family = ConvolutionEngine::RateFamily::RATE_44K;
    PhaseType phase = PhaseType::Minimum;
    if (deps_.activeRateFamily) {
        family = deps_.activeRateFamily();
    }
    if (deps_.activePhaseType) {
        phase = deps_.activePhaseType();
    }
    return resolveFilterPathFor(family, phase);
}

void HeadroomController::refreshCurrentHeadroom(const std::string& reason) const {
    applyHeadroomForPath(currentFilterPath(), reason);
}

void HeadroomController::setTargetPeak(float targetPeak) {
    if (deps_.headroomCache) {
        deps_.headroomCache->setTargetPeak(targetPeak);
    }
}

void HeadroomController::setMode(HeadroomMode mode) {
    if (deps_.headroomCache) {
        deps_.headroomCache->setMode(mode);
    }
}

void HeadroomController::resetEffectiveGain(const std::string& reason) const {
    updateEffectiveGain(1.0f, reason);
}

std::string HeadroomController::resolveFilterPathFor(ConvolutionEngine::RateFamily family,
                                                     PhaseType phase) const {
    if (!deps_.config) {
        return {};
    }

    if (phase == PhaseType::Linear) {
        return (family == ConvolutionEngine::RateFamily::RATE_44K)
                   ? deps_.config->filterPath44kLinear
                   : deps_.config->filterPath48kLinear;
    }
    return (family == ConvolutionEngine::RateFamily::RATE_44K) ? deps_.config->filterPath44kMin
                                                               : deps_.config->filterPath48kMin;
}

void HeadroomController::updateEffectiveGain(float headroomGain, const std::string& reason) const {
    if (deps_.headroomGain) {
        deps_.headroomGain->store(headroomGain, std::memory_order_relaxed);
    }
    float userGain = (deps_.config) ? deps_.config->gain : 1.0f;
    float effective = userGain * headroomGain;
    if (deps_.outputGain) {
        deps_.outputGain->store(effective, std::memory_order_relaxed);
    }
    LOG_INFO("Gain [{}]: user {:.4f} * headroom {:.4f} = {:.4f}", reason, userGain, headroomGain,
             effective);
}

void HeadroomController::applyHeadroomForPath(const std::string& path,
                                              const std::string& reason) const {
    if (path.empty()) {
        LOG_WARN("Headroom [{}]: empty filter path, falling back to unity gain", reason);
        updateEffectiveGain(1.0f, reason);
        return;
    }

    if (!deps_.headroomCache) {
        LOG_WARN("Headroom [{}]: headroom cache missing, falling back to unity gain", reason);
        updateEffectiveGain(1.0f, reason);
        return;
    }

    if (deps_.config && deps_.config->headroomMode == HeadroomMode::FamilyMax &&
        deps_.headroomCache) {
        ConvolutionEngine::RateFamily family = ConvolutionEngine::RateFamily::RATE_44K;
        if (deps_.activeRateFamily) {
            family = deps_.activeRateFamily();
        }
        std::vector<std::string> preloadPaths;
        preloadPaths.push_back(resolveFilterPathFor(family, PhaseType::Minimum));
        preloadPaths.push_back(resolveFilterPathFor(family, PhaseType::Linear));
        deps_.headroomCache->preload(preloadPaths);
    }

    FilterHeadroomInfo info = deps_.headroomCache->get(path);
    updateEffectiveGain(info.safeGain, reason);

    const char* metricLabel = info.usedInputBandPeak ? "input_peak" : "max_coef";
    float metricValue = info.usedInputBandPeak ? info.inputBandPeak : info.maxCoefficient;

    if (!info.metadataFound) {
        LOG_WARN("Headroom [{}]: metadata missing for {} (using safe gain {:.4f})", reason, path,
                 info.safeGain);
    } else {
        LOG_INFO(
            "Headroom [{}]: {} {}={:.6f} safeGain={:.4f} target={:.2f} (max_coef={:.6f}, "
            "input_peak={:.6f})",
            reason, path, metricLabel, metricValue, info.safeGain, info.targetPeak,
            info.maxCoefficient, info.inputBandPeak);
    }
}

}  // namespace audio_pipeline
