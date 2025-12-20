#include "delimiter/safety_controller.h"

#include "audio/overlap_add.h"
#include "logging/logger.h"

#include <algorithm>
#include <utility>

namespace delimiter {
namespace {

const char* fallbackReasonToString(FallbackReason reason) {
    switch (reason) {
    case FallbackReason::None:
        return "none";
    case FallbackReason::InferenceFailure:
        return "inference_failure";
    case FallbackReason::Overload:
        return "overload";
    case FallbackReason::Manual:
        return "manual";
    }
    return "unknown";
}

}  // namespace

SafetyController::SafetyController(SafetyConfig config) : config_(std::move(config)) {
    updateFade();
}

void SafetyController::setSampleRate(int sampleRate) {
    config_.sampleRate = sampleRate;
    updateFade();
}

void SafetyController::setFadeDurationMs(int fadeDurationMs) {
    config_.fadeDurationMs = fadeDurationMs;
    updateFade();
}

SafetyStatus SafetyController::status() const {
    std::lock_guard<std::mutex> lock(statusMutex_);
    SafetyStatus status;
    status.mode = mode_;
    status.targetMode = targetMode_;
    status.lastFallbackReason = lastFallbackReason_;
    status.bypassLocked = bypassLocked_;
    status.detail = lastDetail_;
    return status;
}

bool SafetyController::requestBypass(FallbackReason reason, const std::string& detail) {
    recordDetail(reason, detail);
    if (targetMode_ == ProcessingMode::Bypass && mode_ == ProcessingMode::Bypass) {
        return false;
    }
    startTransition(ProcessingMode::Bypass);
    LOG_WARN("Delimiter: switching to bypass ({})", fallbackReasonToString(reason));
    return true;
}

bool SafetyController::requestActive(bool userRequested, const std::string& detail) {
    if (bypassLocked_ && !userRequested) {
        return false;
    }
    if (targetMode_ == ProcessingMode::Active && mode_ == ProcessingMode::Active) {
        return false;
    }
    startTransition(ProcessingMode::Active);
    if (userRequested) {
        bypassLocked_ = false;
    }
    if (!detail.empty()) {
        LOG_INFO("Delimiter: switching to active (requested={}): {}", userRequested ? "yes" : "no",
                 detail);
    } else {
        LOG_INFO("Delimiter: switching to active (requested={})", userRequested ? "yes" : "no");
    }
    return true;
}

bool SafetyController::observeInferenceResult(const InferenceResult& result) {
    if (result.status == InferenceStatus::Ok) {
        consecutiveFailures_ = 0;
        return false;
    }
    consecutiveFailures_++;
    consecutiveRecovery_ = 0;
    if (consecutiveFailures_ >= config_.failureCountToBypass) {
        std::string detail = result.message.empty() ? "inference error" : result.message;
        if (config_.lockOnFailure) {
            lockBypass(FallbackReason::InferenceFailure, detail);
        } else {
            requestBypass(FallbackReason::InferenceFailure, detail);
        }
        return true;
    }
    return false;
}

bool SafetyController::observeOverload(double realtimeFactor, double queueSeconds) {
    bool rtfOver = (config_.maxRealtimeFactor > 0.0 && realtimeFactor > config_.maxRealtimeFactor);
    bool queueOver = (config_.maxQueueSeconds > 0.0 && queueSeconds > config_.maxQueueSeconds);
    if (!rtfOver && !queueOver) {
        consecutiveOverload_ = 0;
        return false;
    }

    consecutiveOverload_++;
    consecutiveRecovery_ = 0;
    if (consecutiveOverload_ >= config_.overloadCountToBypass) {
        std::string detail = "overload";
        if (rtfOver) {
            detail += " rtf=" + std::to_string(realtimeFactor);
        }
        if (queueOver) {
            detail += " queue_s=" + std::to_string(queueSeconds);
        }
        if (config_.lockOnOverload) {
            lockBypass(FallbackReason::Overload, detail);
        } else {
            requestBypass(FallbackReason::Overload, detail);
        }
        return true;
    }
    return false;
}

void SafetyController::observeHealthy() {
    consecutiveRecovery_++;
    if (mode_ == ProcessingMode::Bypass && !bypassLocked_ &&
        consecutiveRecovery_ >= config_.recoveryCountToRestore) {
        requestActive(false, "recovered");
    }
}

bool SafetyController::mixChunk(const std::vector<float>& processedLeft,
                                const std::vector<float>& processedRight,
                                const std::vector<float>& bypassLeft,
                                const std::vector<float>& bypassRight, std::vector<float>& outLeft,
                                std::vector<float>& outRight) {
    outLeft.clear();
    outRight.clear();
    if (processedLeft.size() != processedRight.size() || bypassLeft.size() != bypassRight.size()) {
        return false;
    }
    if (processedLeft.size() != bypassLeft.size()) {
        return false;
    }
    const std::size_t frames = processedLeft.size();
    if (frames == 0) {
        return true;
    }

    outLeft.resize(frames, 0.0f);
    outRight.resize(frames, 0.0f);

    const float* fromLeft = nullptr;
    const float* fromRight = nullptr;
    const float* toLeft = nullptr;
    const float* toRight = nullptr;
    if (!inTransition_) {
        const bool useProcessed = (mode_ == ProcessingMode::Active);
        const auto& srcLeft = useProcessed ? processedLeft : bypassLeft;
        const auto& srcRight = useProcessed ? processedRight : bypassRight;
        outLeft = srcLeft;
        outRight = srcRight;
        return true;
    }

    if (mode_ == ProcessingMode::Active && targetMode_ == ProcessingMode::Bypass) {
        fromLeft = processedLeft.data();
        fromRight = processedRight.data();
        toLeft = bypassLeft.data();
        toRight = bypassRight.data();
    } else if (mode_ == ProcessingMode::Bypass && targetMode_ == ProcessingMode::Active) {
        fromLeft = bypassLeft.data();
        fromRight = bypassRight.data();
        toLeft = processedLeft.data();
        toRight = processedRight.data();
    } else {
        const bool useProcessed = (targetMode_ == ProcessingMode::Active);
        const auto& srcLeft = useProcessed ? processedLeft : bypassLeft;
        const auto& srcRight = useProcessed ? processedRight : bypassRight;
        outLeft = srcLeft;
        outRight = srcRight;
        inTransition_ = false;
        mode_ = targetMode_;
        return true;
    }

    for (std::size_t i = 0; i < frames; ++i) {
        float fade = fadeValue(fadeIndex_);
        float fromGain = 1.0f - fade;
        outLeft[i] = fromLeft[i] * fromGain + toLeft[i] * fade;
        outRight[i] = fromRight[i] * fromGain + toRight[i] * fade;
        if (fadeIndex_ < fadeSamples_) {
            fadeIndex_++;
        }
        if (fadeIndex_ >= fadeSamples_) {
            inTransition_ = false;
            mode_ = targetMode_;
            for (std::size_t j = i + 1; j < frames; ++j) {
                outLeft[j] = toLeft[j];
                outRight[j] = toRight[j];
            }
            break;
        }
    }
    return true;
}

void SafetyController::startTransition(ProcessingMode targetMode) {
    targetMode_ = targetMode;
    if (mode_ == targetMode_) {
        inTransition_ = false;
        return;
    }
    if (fadeSamples_ <= 1) {
        mode_ = targetMode_;
        inTransition_ = false;
        return;
    }
    inTransition_ = true;
    fadeIndex_ = 0;
}

void SafetyController::updateFade() {
    if (config_.fadeDurationMs <= 0 || config_.sampleRate <= 0) {
        fadeSamples_ = 0;
        fadeCurve_.clear();
        return;
    }
    std::size_t samples =
        static_cast<std::size_t>(config_.fadeDurationMs * config_.sampleRate / 1000);
    fadeSamples_ = std::max<std::size_t>(1, samples);
    fadeCurve_ = AudioUtils::makeRaisedCosineFade(fadeSamples_);
}

float SafetyController::fadeValue(std::size_t index) const {
    if (fadeSamples_ == 0 || fadeCurve_.empty()) {
        return 1.0f;
    }
    if (index >= fadeSamples_) {
        return 1.0f;
    }
    return fadeCurve_[index];
}

void SafetyController::lockBypass(FallbackReason reason, const std::string& detail) {
    bypassLocked_ = true;
    requestBypass(reason, detail);
}

void SafetyController::recordDetail(FallbackReason reason, const std::string& detail) {
    std::lock_guard<std::mutex> lock(statusMutex_);
    lastFallbackReason_ = reason;
    lastDetail_ = detail;
}

}  // namespace delimiter
