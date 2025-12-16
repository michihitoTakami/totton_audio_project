#include "audio/filter_headroom.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {
constexpr float kEpsilon = 1e-6f;
}

FilterHeadroomCache::FilterHeadroomCache(float targetPeak) : targetPeak_(targetPeak) {}

void FilterHeadroomCache::setTargetPeak(float targetPeak) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::abs(targetPeak_ - targetPeak) < kEpsilon) {
        return;
    }
    targetPeak_ = targetPeak;
    cache_.clear();
}

float FilterHeadroomCache::getTargetPeak() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return targetPeak_;
}

FilterHeadroomInfo FilterHeadroomCache::get(const std::string& coefficientPath) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(coefficientPath);
    if (it != cache_.end()) {
        return it->second;
    }

    FilterHeadroomInfo info = loadMetadata(coefficientPath);
    info.targetPeak = targetPeak_;
    info.safeGain = std::min(1.0f, targetPeak_ / std::max(info.maxCoefficient, kEpsilon));
    cache_[coefficientPath] = info;
    return info;
}

FilterHeadroomInfo FilterHeadroomCache::loadMetadata(const std::string& coefficientPath) const {
    FilterHeadroomInfo info;
    info.coefficientPath = coefficientPath;
    info.metadataPath = deriveMetadataPath(coefficientPath);
    info.maxCoefficient = 1.0f;
    info.safeGain = 1.0f;

    std::filesystem::path metaPath(info.metadataPath);
    if (!std::filesystem::exists(metaPath)) {
        std::cerr << "Headroom: metadata not found for " << coefficientPath << " (expected "
                  << info.metadataPath << ")" << '\n';
        return info;
    }

    try {
        std::ifstream ifs(metaPath);
        if (!ifs) {
            std::cerr << "Headroom: failed to open metadata " << info.metadataPath << '\n';
            return info;
        }

        nlohmann::json metadata;
        ifs >> metadata;

        const auto& validation = metadata.at("validation_results");
        const auto& normalization = validation.at("normalization");

        if (normalization.contains("max_coefficient_amplitude")) {
            info.maxCoefficient = normalization["max_coefficient_amplitude"].get<float>();
        }
        if (normalization.contains("l1_norm") && normalization["l1_norm"].is_number()) {
            info.l1Norm = normalization["l1_norm"].get<float>();
        }
        info.metadataFound = true;

        std::cout << "Headroom: metadata loaded for " << coefficientPath << " (max coeff "
                  << info.maxCoefficient << ", L1 " << info.l1Norm << ")" << '\n';

    } catch (const std::exception& e) {
        std::cerr << "Headroom: failed to parse metadata " << info.metadataPath << " - " << e.what()
                  << '\n';
    }

    return info;
}

std::string FilterHeadroomCache::deriveMetadataPath(const std::string& coefficientPath) {
    std::filesystem::path path(coefficientPath);
    std::filesystem::path extension = path.extension();
    if (!extension.empty()) {
        path.replace_extension(".json");
    } else {
        path += ".json";
    }
    return path.string();
}
