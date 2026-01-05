#ifndef FILTER_HEADROOM_H
#define FILTER_HEADROOM_H

#include "core/config_loader.h"

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct FilterHeadroomInfo {
    std::string coefficientPath;
    std::string metadataPath;
    std::string rateFamily;
    float maxCoefficient = 1.0f;
    float l1Norm = 0.0f;
    float normalizedDcGain = 0.0f;
    float upsampleRatio = 0.0f;
    float inputSampleRate = 0.0f;
    float outputSampleRate = 0.0f;
    // Peak gain seen in the input band (normalized by upsample ratio).
    float inputBandPeak = 0.0f;
    float headroomMetric = 1.0f;
    float familyMetric = 1.0f;
    float safeGain = 1.0f;
    float targetPeak = 1.0f;
    bool metadataFound = false;
    bool usedInputBandPeak = false;
    bool familyGainApplied = false;
};

class FilterHeadroomCache {
   public:
    explicit FilterHeadroomCache(float targetPeak = 0.97f);

    void setTargetPeak(float targetPeak);
    void setMode(HeadroomMode mode);
    float getTargetPeak() const;

    // Returns cached info (loads/derives on first request per path)
    FilterHeadroomInfo get(const std::string& coefficientPath);
    void preload(const std::vector<std::string>& coefficientPaths);
    void preloadDirectory(const std::string& directory);

   private:
    FilterHeadroomInfo loadAndCacheLocked(const std::string& coefficientPath);
    FilterHeadroomInfo loadMetadata(const std::string& coefficientPath) const;
    static std::string deriveMetadataPath(const std::string& coefficientPath);
    float computeSafeGain(float metric) const;
    std::string deriveRateFamily(const FilterHeadroomInfo& info) const;
    float updateFamilyMetric(const std::string& familyKey, float headroomMetric);
    void updateCachedFamilyGains(const std::string& familyKey, float familyMetric);

    float targetPeak_;
    HeadroomMode mode_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, FilterHeadroomInfo> cache_;
    std::unordered_map<std::string, float> familyMaxMetric_;
};

#endif  // FILTER_HEADROOM_H
