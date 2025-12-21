#pragma once

#include "core/config_loader.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace delimiter {

enum class InferenceStatus {
    Ok,
    Unsupported,
    InvalidConfig,
    Error,
};

struct InferenceResult {
    InferenceStatus status = InferenceStatus::Error;
    std::string message;
};

struct StereoPlanarView {
    const float* left = nullptr;
    const float* right = nullptr;
    std::size_t frames = 0;

    bool valid() const {
        return left != nullptr && right != nullptr;
    }
};

class InferenceBackend {
   public:
    virtual ~InferenceBackend() = default;

    virtual const char* name() const = 0;

    // Backend input/output sample rate expectation (supports 44.1kHz or 48kHz).
    virtual uint32_t expectedSampleRate() const = 0;

    // Process one chunk.
    // Output vectors are resized to input frames.
    virtual InferenceResult process(const StereoPlanarView& input, std::vector<float>& outLeft,
                                    std::vector<float>& outRight) = 0;

    virtual void reset() = 0;
};

std::unique_ptr<InferenceBackend> createDelimiterInferenceBackend(
    const AppConfig::DelimiterConfig& config);

}  // namespace delimiter
