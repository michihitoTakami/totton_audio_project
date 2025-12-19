#include "delimiter/inference_backend.h"

#include "logging/logger.h"

#include <algorithm>
#include <cctype>

namespace delimiter {
namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

class BypassInferenceBackend final : public InferenceBackend {
   public:
    explicit BypassInferenceBackend(uint32_t expectedSampleRate)
        : expectedSampleRate_(expectedSampleRate) {}

    const char* name() const override {
        return "bypass";
    }

    uint32_t expectedSampleRate() const override {
        return expectedSampleRate_;
    }

    InferenceResult process(const StereoPlanarView& input, std::vector<float>& outLeft,
                            std::vector<float>& outRight) override {
        if (!input.valid() || input.frames == 0) {
            outLeft.clear();
            outRight.clear();
            return {InferenceStatus::InvalidConfig, "invalid input buffer"};
        }

        outLeft.assign(input.left, input.left + input.frames);
        outRight.assign(input.right, input.right + input.frames);
        return {InferenceStatus::Ok, ""};
    }

    void reset() override {}

   private:
    uint32_t expectedSampleRate_;
};

class OrtInferenceBackendPlaceholder final : public InferenceBackend {
   public:
    explicit OrtInferenceBackendPlaceholder(const AppConfig::DelimiterConfig& config)
        : config_(config) {}

    const char* name() const override {
        return "ort";
    }

    uint32_t expectedSampleRate() const override {
        return config_.expectedSampleRate;
    }

    InferenceResult process(const StereoPlanarView& input, std::vector<float>& outLeft,
                            std::vector<float>& outRight) override {
        (void)input;
        outLeft.clear();
        outRight.clear();

        if (config_.ort.modelPath.empty()) {
            return {InferenceStatus::InvalidConfig, "delimiter.ort.modelPath is empty"};
        }

        return {InferenceStatus::Unsupported,
                "ORT backend is not linked in C++ yet. Use an out-of-process backend (follow-up)"};
    }

    void reset() override {}

   private:
    AppConfig::DelimiterConfig config_;
};

}  // namespace

std::unique_ptr<InferenceBackend> createDelimiterInferenceBackend(
    const AppConfig::DelimiterConfig& config) {
    std::string backend = toLower(config.backend);
    if (!config.enabled) {
        return std::make_unique<BypassInferenceBackend>(config.expectedSampleRate);
    }

    if (backend == "bypass") {
        return std::make_unique<BypassInferenceBackend>(config.expectedSampleRate);
    }

    if (backend == "ort") {
        return std::make_unique<OrtInferenceBackendPlaceholder>(config);
    }

    LOG_WARN("Delimiter: Unknown backend '{}' (falling back to bypass)", config.backend);
    return std::make_unique<BypassInferenceBackend>(config.expectedSampleRate);
}

}  // namespace delimiter
