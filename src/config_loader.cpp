#include "config_loader.h"

#include "daemon_constants.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

PhaseType parsePhaseType(const std::string& str) {
    if (str == "linear") {
        return PhaseType::Linear;
    }
    // Default to Minimum for "minimum" or any invalid value
    return PhaseType::Minimum;
}

const char* phaseTypeToString(PhaseType type) {
    switch (type) {
    case PhaseType::Linear:
        return "linear";
    case PhaseType::Minimum:
    default:
        return "minimum";
    }
}

// Validate headSize string, returns default "m" for invalid values
static std::string validateHeadSize(const std::string& str) {
    if (str == "s" || str == "m" || str == "l" || str == "xl") {
        return str;
    }
    return "m";  // Default to medium for invalid values
}

bool loadAppConfig(const std::filesystem::path& configPath, AppConfig& outConfig, bool verbose) {
    outConfig = AppConfig{};

    std::ifstream file(configPath);
    if (!file.is_open()) {
        if (verbose) {
            std::cout << "Config: " << configPath << " not found, using defaults" << std::endl;
        }
        return false;
    }

    try {
        nlohmann::json j;
        file >> j;

        if (j.contains("alsaDevice"))
            outConfig.alsaDevice = j["alsaDevice"].get<std::string>();
        if (j.contains("bufferSize"))
            outConfig.bufferSize = j["bufferSize"].get<int>();
        if (j.contains("periodSize"))
            outConfig.periodSize = j["periodSize"].get<int>();
        if (j.contains("upsampleRatio"))
            outConfig.upsampleRatio = j["upsampleRatio"].get<int>();
        if (j.contains("blockSize"))
            outConfig.blockSize = j["blockSize"].get<int>();
        if (j.contains("gain"))
            outConfig.gain = j["gain"].get<float>();
        if (j.contains("headroomTarget"))
            outConfig.headroomTarget = j["headroomTarget"].get<float>();
        if (j.contains("filterPath"))
            outConfig.filterPath = j["filterPath"].get<std::string>();
        if (j.contains("phaseType"))
            outConfig.phaseType = parsePhaseType(j["phaseType"].get<std::string>());

        // Quad-phase mode settings
        if (j.contains("quadPhaseEnabled"))
            outConfig.quadPhaseEnabled = j["quadPhaseEnabled"].get<bool>();
        if (j.contains("filterPath44kMin"))
            outConfig.filterPath44kMin = j["filterPath44kMin"].get<std::string>();
        if (j.contains("filterPath48kMin"))
            outConfig.filterPath48kMin = j["filterPath48kMin"].get<std::string>();
        if (j.contains("filterPath44kLinear"))
            outConfig.filterPath44kLinear = j["filterPath44kLinear"].get<std::string>();
        if (j.contains("filterPath48kLinear"))
            outConfig.filterPath48kLinear = j["filterPath48kLinear"].get<std::string>();

        // Multi-rate mode settings (Issue #219)
        if (j.contains("multiRateEnabled"))
            outConfig.multiRateEnabled = j["multiRateEnabled"].get<bool>();
        if (j.contains("coefficientDir"))
            outConfig.coefficientDir = j["coefficientDir"].get<std::string>();

        // Partitioned convolution settings (Issue #351)
        if (j.contains("partitionedConvolution") && j["partitionedConvolution"].is_object()) {
            auto pc = j["partitionedConvolution"];
            try {
                if (pc.contains("enabled") && pc["enabled"].is_boolean())
                    outConfig.partitionedConvolution.enabled = pc["enabled"].get<bool>();
                if (pc.contains("fastPartitionTaps") && pc["fastPartitionTaps"].is_number_integer())
                    outConfig.partitionedConvolution.fastPartitionTaps =
                        std::max(1024, pc["fastPartitionTaps"].get<int>());
                if (pc.contains("minPartitionTaps") && pc["minPartitionTaps"].is_number_integer())
                    outConfig.partitionedConvolution.minPartitionTaps =
                        std::max(1024, pc["minPartitionTaps"].get<int>());
                if (pc.contains("maxPartitions") && pc["maxPartitions"].is_number_integer())
                    outConfig.partitionedConvolution.maxPartitions =
                        std::max(1, pc["maxPartitions"].get<int>());
            } catch (const std::exception& e) {
                if (verbose) {
                    std::cerr << "Config: Invalid partitionedConvolution settings, using defaults: "
                              << e.what() << std::endl;
                }
                outConfig.partitionedConvolution = AppConfig::PartitionedConvolutionConfig{};
            }
        }

        // EQ settings
        if (j.contains("eqEnabled"))
            outConfig.eqEnabled = j["eqEnabled"].get<bool>();
        if (j.contains("eqProfilePath"))
            outConfig.eqProfilePath = j["eqProfilePath"].get<std::string>();

        // Crossfeed settings (with type error handling)
        if (j.contains("crossfeed") && j["crossfeed"].is_object()) {
            auto cf = j["crossfeed"];
            try {
                if (cf.contains("enabled") && cf["enabled"].is_boolean())
                    outConfig.crossfeed.enabled = cf["enabled"].get<bool>();
                if (cf.contains("headSize") && cf["headSize"].is_string())
                    outConfig.crossfeed.headSize =
                        validateHeadSize(cf["headSize"].get<std::string>());
                if (cf.contains("hrtfPath") && cf["hrtfPath"].is_string())
                    outConfig.crossfeed.hrtfPath = cf["hrtfPath"].get<std::string>();
            } catch (const std::exception& e) {
                // On type error, keep defaults (already set in AppConfig{})
                if (verbose) {
                    std::cerr << "Config: Invalid crossfeed settings, using defaults: " << e.what()
                              << std::endl;
                }
            }
        }

        // Fallback settings (Issue #139)
        if (j.contains("fallback") && j["fallback"].is_object()) {
            auto fb = j["fallback"];
            try {
                if (fb.contains("enabled") && fb["enabled"].is_boolean())
                    outConfig.fallback.enabled = fb["enabled"].get<bool>();
                if (fb.contains("gpuThreshold") && fb["gpuThreshold"].is_number())
                    outConfig.fallback.gpuThreshold = fb["gpuThreshold"].get<float>();
                if (fb.contains("gpuThresholdCount") && fb["gpuThresholdCount"].is_number_integer())
                    outConfig.fallback.gpuThresholdCount = fb["gpuThresholdCount"].get<int>();
                if (fb.contains("gpuRecoveryThreshold") && fb["gpuRecoveryThreshold"].is_number())
                    outConfig.fallback.gpuRecoveryThreshold =
                        fb["gpuRecoveryThreshold"].get<float>();
                if (fb.contains("gpuRecoveryCount") && fb["gpuRecoveryCount"].is_number_integer())
                    outConfig.fallback.gpuRecoveryCount = fb["gpuRecoveryCount"].get<int>();
                if (fb.contains("xrunTriggersFallback") && fb["xrunTriggersFallback"].is_boolean())
                    outConfig.fallback.xrunTriggersFallback =
                        fb["xrunTriggersFallback"].get<bool>();
                if (fb.contains("monitorIntervalMs") && fb["monitorIntervalMs"].is_number_integer())
                    outConfig.fallback.monitorIntervalMs = fb["monitorIntervalMs"].get<int>();

                // Validate fallback configuration values
                // GPU threshold: clamp to 0-100%
                outConfig.fallback.gpuThreshold =
                    std::clamp(outConfig.fallback.gpuThreshold, 0.0f, 100.0f);
                // Recovery threshold: clamp to 0-threshold (must be <= threshold for hysteresis)
                outConfig.fallback.gpuRecoveryThreshold = std::clamp(
                    outConfig.fallback.gpuRecoveryThreshold, 0.0f, outConfig.fallback.gpuThreshold);
                // Count values: ensure at least 1
                outConfig.fallback.gpuThresholdCount =
                    std::max(1, outConfig.fallback.gpuThresholdCount);
                outConfig.fallback.gpuRecoveryCount =
                    std::max(1, outConfig.fallback.gpuRecoveryCount);
                // Monitor interval: ensure at least 10ms (smaller values cause high CPU load)
                outConfig.fallback.monitorIntervalMs =
                    std::max(10, outConfig.fallback.monitorIntervalMs);
            } catch (const std::exception& e) {
                // On type error, keep defaults (already set in AppConfig{})
                if (verbose) {
                    std::cerr << "Config: Invalid fallback settings, using defaults: " << e.what()
                              << std::endl;
                }
            }
        }

        // Clamp derived floating-point values after parsing (ensures sane bounds)
        outConfig.gain = std::max(0.0f, outConfig.gain);
        outConfig.headroomTarget =
            std::clamp(outConfig.headroomTarget, DaemonConstants::MIN_HEADROOM_TARGET,
                       DaemonConstants::MAX_HEADROOM_TARGET);

        if (verbose) {
            std::cout << "Config: Loaded from " << std::filesystem::absolute(configPath)
                      << std::endl;
        }

        // Clamp derived floating-point values after parsing (ensures sane bounds)
        outConfig.headroomTarget = std::clamp(outConfig.headroomTarget, 0.01f, 1.0f);

        return true;
    } catch (const std::exception& e) {
        if (verbose) {
            std::cerr << "Config: Failed to parse " << configPath << ": " << e.what() << std::endl;
        }
        return false;
    }
}
