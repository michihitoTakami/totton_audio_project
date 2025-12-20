#pragma once

#include "delimiter/inference_backend.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace delimiter {

enum class ProcessingMode {
    Active,
    Bypass,
};

enum class FallbackReason {
    None,
    InferenceFailure,
    Overload,
    Manual,
};

struct SafetyConfig {
    int sampleRate = 44100;
    int fadeDurationMs = 250;
    std::size_t failureCountToBypass = 1;
    std::size_t overloadCountToBypass = 3;
    std::size_t recoveryCountToRestore = 5;
    double maxRealtimeFactor = 1.2;
    double maxQueueSeconds = 0.0;
    bool lockOnFailure = true;
    bool lockOnOverload = false;
};

struct SafetyStatus {
    ProcessingMode mode = ProcessingMode::Active;
    ProcessingMode targetMode = ProcessingMode::Active;
    FallbackReason lastFallbackReason = FallbackReason::None;
    bool bypassLocked = false;
    std::string detail;
};

class SafetyController {
   public:
    explicit SafetyController(SafetyConfig config);

    void setSampleRate(int sampleRate);
    void setFadeDurationMs(int fadeDurationMs);

    SafetyStatus status() const;

    bool requestBypass(FallbackReason reason, const std::string& detail);
    bool requestActive(bool userRequested, const std::string& detail);

    bool observeInferenceResult(const InferenceResult& result);
    bool observeOverload(double realtimeFactor, double queueSeconds);
    void observeHealthy();

    bool mixChunk(const std::vector<float>& processedLeft, const std::vector<float>& processedRight,
                  const std::vector<float>& bypassLeft, const std::vector<float>& bypassRight,
                  std::vector<float>& outLeft, std::vector<float>& outRight);

   private:
    void startTransition(ProcessingMode targetMode);
    void updateFade();
    float fadeValue(std::size_t index) const;
    void lockBypass(FallbackReason reason, const std::string& detail);
    void recordDetail(FallbackReason reason, const std::string& detail);

    SafetyConfig config_;
    ProcessingMode mode_ = ProcessingMode::Active;
    ProcessingMode targetMode_ = ProcessingMode::Active;
    bool inTransition_ = false;
    std::size_t fadeIndex_ = 0;
    std::size_t fadeSamples_ = 0;
    std::vector<float> fadeCurve_;

    std::size_t consecutiveFailures_ = 0;
    std::size_t consecutiveOverload_ = 0;
    std::size_t consecutiveRecovery_ = 0;
    bool bypassLocked_ = false;

    mutable std::mutex statusMutex_;
    FallbackReason lastFallbackReason_ = FallbackReason::None;
    std::string lastDetail_;
};

}  // namespace delimiter
