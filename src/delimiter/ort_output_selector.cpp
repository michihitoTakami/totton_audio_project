#include "delimiter/ort_output_selector.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace delimiter {
namespace {

std::string toLower(std::string_view value) {
    std::string out(value.begin(), value.end());
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

bool containsDecoded(std::string_view name) {
    const std::string lower = toLower(name);
    return lower.find("decoded") != std::string::npos;
}

bool isPositive(int64_t v) {
    return v > 0;
}

bool isTwoOrDynamic(int64_t v) {
    return v == 2 || v == -1;
}

}  // namespace

bool isStereoLikeShape(const std::vector<int64_t>& shape) {
    if (shape.size() == 3) {
        // [N, C, F]
        return (shape[0] == 1 || shape[0] == -1) && isTwoOrDynamic(shape[1]) &&
               isPositive(shape[2]);
    }

    if (shape.size() == 2) {
        // [C, F] or [F, C]
        if (shape[0] == 2 && isPositive(shape[1])) {
            return true;
        }
        if (shape[1] == 2 && isPositive(shape[0])) {
            return true;
        }
        return false;
    }

    return false;
}

std::optional<std::size_t> pickStereoOutputIndex(
    const std::vector<std::string>& outputNames,
    const std::vector<std::vector<int64_t>>& outputShapes) {
    if (outputNames.size() != outputShapes.size()) {
        return std::nullopt;
    }

    // 1) Prefer "decoded" output if it looks stereo.
    for (std::size_t i = 0; i < outputNames.size(); ++i) {
        if (containsDecoded(outputNames[i]) && isStereoLikeShape(outputShapes[i])) {
            return i;
        }
    }

    // 2) Otherwise choose the first stereo-like output.
    for (std::size_t i = 0; i < outputShapes.size(); ++i) {
        if (isStereoLikeShape(outputShapes[i])) {
            return i;
        }
    }

    return std::nullopt;
}

}  // namespace delimiter
