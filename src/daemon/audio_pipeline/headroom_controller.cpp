#include "daemon/audio_pipeline/headroom_controller.h"

#include "logging/logger.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <utility>

namespace audio_pipeline {
namespace {
inline int64_t debug_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

inline const char* debug_run_id() {
    const char* v = std::getenv("MAGICBOX_DEBUG_RUN_ID");
    return (v && *v) ? v : "pre-fix";
}

inline void debug_ndjson(const char* location, const char* message, const char* hypothesisId,
                         const nlohmann::json& data) {
    nlohmann::json payload;
    payload["sessionId"] = "debug-session";
    payload["runId"] = debug_run_id();
    payload["hypothesisId"] = hypothesisId;
    payload["location"] = location;
    payload["message"] = message;
    payload["data"] = data;
    payload["timestamp"] = debug_now_ms();
    std::ofstream ofs("/home/michihito/Working/gpu_os/.cursor/debug.log", std::ios::app);
    if (!ofs) {
        return;
    }
    ofs << payload.dump() << "\n";
}
}  // namespace

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
    // #region agent log
    debug_ndjson("src/daemon/audio_pipeline/headroom_controller.cpp:updateEffectiveGain",
                 "Updated effective gain", "H1",
                 {{"reason", reason},
                  {"userGain", userGain},
                  {"headroomGain", headroomGain},
                  {"effectiveGain", effective}});
    // #endregion
    LOG_INFO("Gain [{}]: user {:.4f} * headroom {:.4f} = {:.4f}", reason, userGain, headroomGain,
             effective);
}

void HeadroomController::applyHeadroomForPath(const std::string& path,
                                              const std::string& reason) const {
    // #region agent log
    debug_ndjson("src/daemon/audio_pipeline/headroom_controller.cpp:applyHeadroomForPath",
                 "Applying headroom for filter path", "H3",
                 {{"reason", reason},
                  {"path", path},
                  {"userGain", (deps_.config) ? deps_.config->gain : 1.0f}});
    // #endregion
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

    FilterHeadroomInfo info = deps_.headroomCache->get(path);
    // #region agent log
    debug_ndjson("src/daemon/audio_pipeline/headroom_controller.cpp:applyHeadroomForPath",
                 "Headroom cache result", "H1",
                 {{"reason", reason},
                  {"path", path},
                  {"metadataFound", info.metadataFound},
                  {"maxCoefficient", info.maxCoefficient},
                  {"l1Norm", info.l1Norm},
                  {"safeGain", info.safeGain},
                  {"targetPeak", info.targetPeak}});
    // #endregion
    updateEffectiveGain(info.safeGain, reason);

    if (!info.metadataFound) {
        LOG_WARN("Headroom [{}]: metadata missing for {} (using safe gain {:.4f})", reason, path,
                 info.safeGain);
    } else {
        LOG_INFO("Headroom [{}]: {} max_coef={:.6f} safeGain={:.4f} target={:.2f}", reason, path,
                 info.maxCoefficient, info.safeGain, info.targetPeak);
    }
}

}  // namespace audio_pipeline
