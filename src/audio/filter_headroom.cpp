#include "audio/filter_headroom.h"

#include "logging/logger.h"

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
    float headroomMetric =
        (info.inputBandPeak > kEpsilon) ? info.inputBandPeak : info.maxCoefficient;
    info.usedInputBandPeak = info.inputBandPeak > kEpsilon;
    info.safeGain = std::min(1.0f, targetPeak_ / std::max(headroomMetric, kEpsilon));
    cache_[coefficientPath] = info;
    return info;
}

FilterHeadroomInfo FilterHeadroomCache::loadMetadata(const std::string& coefficientPath) const {
    FilterHeadroomInfo info;
    info.coefficientPath = coefficientPath;
    info.metadataPath = deriveMetadataPath(coefficientPath);
    info.maxCoefficient = 1.0f;
    info.inputBandPeak = 0.0f;
    info.upsampleRatio = 0.0f;
    info.normalizedDcGain = 0.0f;
    info.safeGain = 1.0f;

    std::filesystem::path metaPath(info.metadataPath);
    if (!std::filesystem::exists(metaPath)) {
        LOG_WARN("Headroom: metadata not found for {} (expected {})", coefficientPath,
                 info.metadataPath);
        return info;
    }

    try {
        std::ifstream ifs(metaPath);
        if (!ifs) {
            LOG_WARN("Headroom: failed to open metadata {}", info.metadataPath);
            return info;
        }

        nlohmann::json metadata;
        ifs >> metadata;

        if (metadata.contains("upsample_ratio") && metadata["upsample_ratio"].is_number()) {
            info.upsampleRatio = metadata["upsample_ratio"].get<float>();
        }

        const auto& validation = metadata.at("validation_results");
        const auto& normalization = validation.at("normalization");

        if (normalization.contains("max_coefficient_amplitude")) {
            info.maxCoefficient = normalization["max_coefficient_amplitude"].get<float>();
        }
        if (normalization.contains("l1_norm") && normalization["l1_norm"].is_number()) {
            info.l1Norm = normalization["l1_norm"].get<float>();
        }
        if (normalization.contains("normalized_dc_gain") &&
            normalization["normalized_dc_gain"].is_number()) {
            info.normalizedDcGain = normalization["normalized_dc_gain"].get<float>();
        }

        float inputBandPeakRaw = 0.0f;
        if (validation.contains("input_band_peak_normalized") &&
            validation["input_band_peak_normalized"].is_number()) {
            info.inputBandPeak = validation["input_band_peak_normalized"].get<float>();
        } else if (validation.contains("input_band_peak") &&
                   validation["input_band_peak"].is_number()) {
            inputBandPeakRaw = validation["input_band_peak"].get<float>();
        }

        if (info.inputBandPeak < kEpsilon) {
            float derivedDcGain = info.normalizedDcGain;
            if ((derivedDcGain < kEpsilon) && normalization.contains("target_dc_gain") &&
                normalization.contains("dc_gain_factor") &&
                normalization["target_dc_gain"].is_number() &&
                normalization["dc_gain_factor"].is_number()) {
                derivedDcGain = normalization["target_dc_gain"].get<float>() *
                                normalization["dc_gain_factor"].get<float>();
            }

            if (info.upsampleRatio > kEpsilon) {
                if (inputBandPeakRaw > kEpsilon) {
                    info.inputBandPeak = inputBandPeakRaw / info.upsampleRatio;
                } else if (derivedDcGain > kEpsilon) {
                    info.inputBandPeak = derivedDcGain / info.upsampleRatio;
                }
            }
        }

        info.metadataFound = true;

        std::cout << "Headroom: metadata loaded for " << coefficientPath << " (max coeff "
                  << info.maxCoefficient << ", L1 " << info.l1Norm << ", input peak "
                  << info.inputBandPeak << ")" << '\n';

    } catch (const std::exception& e) {
        LOG_WARN("Headroom: failed to parse metadata {} - {}", info.metadataPath, e.what());
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
