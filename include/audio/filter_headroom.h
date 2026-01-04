#ifndef FILTER_HEADROOM_H
#define FILTER_HEADROOM_H

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

struct FilterHeadroomInfo {
    std::string coefficientPath;
    std::string metadataPath;
    float maxCoefficient = 1.0f;
    float l1Norm = 0.0f;
    float normalizedDcGain = 0.0f;
    float upsampleRatio = 0.0f;
    // Peak gain seen in the input band (normalized by upsample ratio).
    float inputBandPeak = 0.0f;
    float safeGain = 1.0f;
    float targetPeak = 1.0f;
    bool metadataFound = false;
    bool usedInputBandPeak = false;
};

class FilterHeadroomCache {
   public:
    explicit FilterHeadroomCache(float targetPeak = 0.97f);

    void setTargetPeak(float targetPeak);
    float getTargetPeak() const;

    // Returns cached info (loads/derives on first request per path)
    FilterHeadroomInfo get(const std::string& coefficientPath);

   private:
    FilterHeadroomInfo loadMetadata(const std::string& coefficientPath) const;
    static std::string deriveMetadataPath(const std::string& coefficientPath);

    float targetPeak_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, FilterHeadroomInfo> cache_;
};

#endif  // FILTER_HEADROOM_H
