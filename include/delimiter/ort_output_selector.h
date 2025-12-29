#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace delimiter {

// Pure helper for selecting the stereo audio output from multi-output ONNX models.
// This is intentionally independent from onnxruntime headers to keep it easily testable.
//
// Policy:
// - Prefer an output whose name contains "decoded" (case-insensitive) AND has a stereo-like shape.
// - Otherwise, choose the first output that has a stereo-like shape.
// - If none match, return nullopt.
std::optional<std::size_t> pickStereoOutputIndex(
    const std::vector<std::string>& outputNames,
    const std::vector<std::vector<int64_t>>& outputShapes);

// Stereo-like shapes accepted by `OrtInferenceBackend::extractOutputs`:
// - [1, 2, frames] (or with -1 for batch/channel)
// - [2, frames]
// - [frames, 2]
bool isStereoLikeShape(const std::vector<int64_t>& shape);

}  // namespace delimiter
