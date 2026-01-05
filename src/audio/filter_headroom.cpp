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

std::string familyFromRate(int rate) {
    if (rate > 0 && rate % 44100 == 0) {
        return "44k";
    }
    if (rate > 0 && rate % 48000 == 0) {
        return "48k";
    }
    return "";
}
}  // namespace

FilterHeadroomCache::FilterHeadroomCache(float targetPeak)
    : targetPeak_(targetPeak), mode_(HeadroomMode::FamilyMax) {}

void FilterHeadroomCache::setTargetPeak(float targetPeak) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::abs(targetPeak_ - targetPeak) < kEpsilon) {
        return;
    }
    targetPeak_ = targetPeak;
    familyMaxMetric_.clear();
    cache_.clear();
}

void FilterHeadroomCache::setMode(HeadroomMode mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ == mode) {
        return;
    }
    mode_ = mode;
    familyMaxMetric_.clear();
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

    return loadAndCacheLocked(coefficientPath);
}

void FilterHeadroomCache::preload(const std::vector<std::string>& coefficientPaths) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& path : coefficientPaths) {
        if (path.empty()) {
            continue;
        }
        if (cache_.find(path) != cache_.end()) {
            continue;
        }
        loadAndCacheLocked(path);
    }
}

void FilterHeadroomCache::preloadDirectory(const std::string& directory) {
    namespace fs = std::filesystem;
    std::vector<std::string> paths;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(directory, ec)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() != ".json") {
            continue;
        }
        fs::path coeffPath = entry.path();
        coeffPath.replace_extension(".bin");
        paths.push_back(coeffPath.string());
    }
    if (!paths.empty()) {
        preload(paths);
    }
}

FilterHeadroomInfo FilterHeadroomCache::loadAndCacheLocked(const std::string& coefficientPath) {
    FilterHeadroomInfo info = loadMetadata(coefficientPath);
    info.targetPeak = targetPeak_;
    info.rateFamily = deriveRateFamily(info);
    info.usedInputBandPeak = info.inputBandPeak > kEpsilon;
    info.headroomMetric =
        (info.inputBandPeak > kEpsilon) ? info.inputBandPeak : info.maxCoefficient;
    info.familyMetric = info.headroomMetric;
    info.familyGainApplied = false;

    if (mode_ == HeadroomMode::FamilyMax && !info.rateFamily.empty()) {
        float familyMetric = updateFamilyMetric(info.rateFamily, info.headroomMetric);
        info.familyMetric = familyMetric;
        info.familyGainApplied = true;
        updateCachedFamilyGains(info.rateFamily, familyMetric);
    }

    info.safeGain = computeSafeGain(info.familyMetric);
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
    info.inputSampleRate = 0.0f;
    info.outputSampleRate = 0.0f;
    info.headroomMetric = 1.0f;
    info.familyMetric = 1.0f;
    info.familyGainApplied = false;
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
        if (metadata.contains("sample_rate_input") && metadata["sample_rate_input"].is_number()) {
            info.inputSampleRate = metadata["sample_rate_input"].get<float>();
        }
        if (metadata.contains("sample_rate_output") && metadata["sample_rate_output"].is_number()) {
            info.outputSampleRate = metadata["sample_rate_output"].get<float>();
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

float FilterHeadroomCache::computeSafeGain(float metric) const {
    return std::min(1.0f, targetPeak_ / std::max(metric, kEpsilon));
}

std::string FilterHeadroomCache::deriveRateFamily(const FilterHeadroomInfo& info) const {
    int inputRate = static_cast<int>(std::lround(info.inputSampleRate));
    int outputRate = static_cast<int>(std::lround(info.outputSampleRate));

    std::string family = familyFromRate(inputRate);
    if (!family.empty()) {
        return family;
    }
    family = familyFromRate(outputRate);
    if (!family.empty()) {
        return family;
    }
    if (info.upsampleRatio > kEpsilon && inputRate > 0) {
        int derivedOutput =
            static_cast<int>(std::lround(info.upsampleRatio * info.inputSampleRate));
        family = familyFromRate(derivedOutput);
    }
    return family;
}

float FilterHeadroomCache::updateFamilyMetric(const std::string& familyKey, float headroomMetric) {
    float& maxMetric = familyMaxMetric_[familyKey];
    if (headroomMetric > maxMetric) {
        maxMetric = headroomMetric;
    }
    return maxMetric;
}

void FilterHeadroomCache::updateCachedFamilyGains(const std::string& familyKey,
                                                  float familyMetric) {
    float familySafeGain = computeSafeGain(familyMetric);
    for (auto& [_, cached] : cache_) {
        if (cached.rateFamily == familyKey) {
            cached.familyMetric = familyMetric;
            cached.familyGainApplied = (mode_ == HeadroomMode::FamilyMax);
            cached.safeGain = familySafeGain;
            cached.targetPeak = targetPeak_;
        }
    }
}
