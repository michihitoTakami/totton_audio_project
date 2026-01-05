#include "core/config_loader.h"

#include "core/daemon_constants.h"
#include "logging/logger.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

static std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

PhaseType parsePhaseType(const std::string& str) {
    if (str == "linear" || str == "hybrid") {
        return PhaseType::Linear;
    }
    // Default to Minimum for "minimum" or any invalid value
    return PhaseType::Minimum;
}

const char* phaseTypeToString(PhaseType type) {
    switch (type) {
    case PhaseType::Linear:
        return "hybrid";
    case PhaseType::Minimum:
    default:
        return "minimum";
    }
}

HeadroomMode parseHeadroomMode(const std::string& str) {
    std::string lower = toLower(str);
    if (lower == "per_filter" || lower == "per-filter" || lower == "legacy") {
        return HeadroomMode::PerFilter;
    }
    return HeadroomMode::FamilyMax;
}

const char* headroomModeToString(HeadroomMode mode) {
    switch (mode) {
    case HeadroomMode::PerFilter:
        return "per_filter";
    case HeadroomMode::FamilyMax:
    default:
        return "family_max";
    }
}

OutputMode parseOutputMode(const std::string& str) {
    std::string lower = toLower(str);
    if (lower == "usb") {
        return OutputMode::Usb;
    }
    return OutputMode::Usb;
}

const char* outputModeToString(OutputMode mode) {
    switch (mode) {
    case OutputMode::Usb:
    default:
        return "usb";
    }
}

GpuBackend parseGpuBackend(const std::string& str) {
    std::string lower = toLower(str);
    if (lower == "vulkan") {
        return GpuBackend::Vulkan;
    }
    return GpuBackend::Cuda;
}

const char* gpuBackendToString(GpuBackend backend) {
    switch (backend) {
    case GpuBackend::Vulkan:
        return "vulkan";
    case GpuBackend::Cuda:
    default:
        return "cuda";
    }
}

// Validate headSize string, returns default "m" for invalid values
static std::string validateHeadSize(const std::string& str) {
    if (str == "xs" || str == "s" || str == "m" || str == "l" || str == "xl") {
        return str;
    }
    return "m";  // Default to medium for invalid values
}

static std::string validateDelimiterBackend(const std::string& str) {
    std::string lower = toLower(str);
    if (lower == "bypass") {
        return lower;
    }
    // Accept various ONNX Runtime aliases, normalize to "ort"
    if (lower == "ort" || lower == "onnx" || lower == "onnxruntime") {
        return "ort";
    }
    return "bypass";
}

static std::string validateOrtProvider(const std::string& str) {
    std::string lower = toLower(str);
    if (lower == "cpu" || lower == "cuda" || lower == "tensorrt" || lower == "trt") {
        return (lower == "trt") ? "tensorrt" : lower;
    }
    return "cpu";
}

static uint32_t sanitizeDelimiterSampleRate(int sr, bool verbose) {
    constexpr uint32_t kRate44k = 44100;
    constexpr uint32_t kRate48k = 48000;
    if (sr == static_cast<int>(kRate44k) || sr == static_cast<int>(kRate48k)) {
        return static_cast<uint32_t>(sr);
    }
    if (verbose) {
        LOG_WARN(
            "Config: delimiter.expectedSampleRate must be 44100 or 48000 (got {}), using 44100",
            sr);
    }
    return kRate44k;
}

bool loadAppConfig(const std::filesystem::path& configPath, AppConfig& outConfig, bool verbose) {
    outConfig = AppConfig{};

    std::ifstream file(configPath);
    if (!file.is_open()) {
        if (verbose) {
            std::cout << "Config: " << configPath << " not found, using defaults" << '\n';
        }
        return false;
    }

    try {
        nlohmann::json j;
        file >> j;

        if (j.contains("alsaDevice")) {
            outConfig.alsaDevice = j["alsaDevice"].get<std::string>();
        }
        if (j.contains("bufferSize")) {
            outConfig.bufferSize = j["bufferSize"].get<int>();
        }
        if (j.contains("periodSize")) {
            outConfig.periodSize = j["periodSize"].get<int>();
        }
        if (j.contains("upsampleRatio")) {
            outConfig.upsampleRatio = j["upsampleRatio"].get<int>();
        }
        if (j.contains("blockSize")) {
            outConfig.blockSize = j["blockSize"].get<int>();
        }
        if (j.contains("gain")) {
            outConfig.gain = j["gain"].get<float>();
        }
        if (j.contains("headroomTarget")) {
            outConfig.headroomTarget = j["headroomTarget"].get<float>();
        }
        if (j.contains("headroomMode") && j["headroomMode"].is_string()) {
            outConfig.headroomMode = parseHeadroomMode(j["headroomMode"].get<std::string>());
        }
        if (j.contains("filterPath")) {
            outConfig.filterPath = j["filterPath"].get<std::string>();
        }
        if (j.contains("gpuBackend") && j["gpuBackend"].is_string()) {
            outConfig.gpuBackend = parseGpuBackend(j["gpuBackend"].get<std::string>());
        }
        if (j.contains("phaseType")) {
            outConfig.phaseType = parsePhaseType(j["phaseType"].get<std::string>());
        }

        // Keep output config aligned with legacy alsaDevice field by default
        outConfig.output.mode = OutputMode::Usb;
        outConfig.output.usb.preferredDevice = outConfig.alsaDevice;

        if (j.contains("output") && j["output"].is_object()) {
            auto output = j["output"];
            try {
                if (output.contains("mode") && output["mode"].is_string()) {
                    std::string modeStr = output["mode"].get<std::string>();
                    OutputMode parsed = parseOutputMode(modeStr);
                    std::string normalized = toLower(modeStr);
                    if (normalized != "usb" && verbose) {
                        LOG_WARN("Config: Unsupported output.mode '{}', falling back to 'usb'",
                                 modeStr);
                    }
                    outConfig.output.mode = parsed;
                }

                if (output.contains("options") && output["options"].is_object()) {
                    auto options = output["options"];
                    if (options.contains("usb") && options["usb"].is_object()) {
                        auto usb = options["usb"];
                        if (usb.contains("preferredDevice") && usb["preferredDevice"].is_string()) {
                            outConfig.output.usb.preferredDevice =
                                usb["preferredDevice"].get<std::string>();
                        }
                    }
                }
            } catch (const std::exception& e) {
                if (verbose) {
                    LOG_WARN("Config: Invalid output settings, using defaults: {}", e.what());
                }
                outConfig.output = OutputConfig{};
            }
        }

        if (j.contains("filterPath44kMin")) {
            outConfig.filterPath44kMin = j["filterPath44kMin"].get<std::string>();
        }
        if (j.contains("filterPath48kMin")) {
            outConfig.filterPath48kMin = j["filterPath48kMin"].get<std::string>();
        }
        if (j.contains("filterPath44kLinear")) {
            outConfig.filterPath44kLinear = j["filterPath44kLinear"].get<std::string>();
        }
        if (j.contains("filterPath48kLinear")) {
            outConfig.filterPath48kLinear = j["filterPath48kLinear"].get<std::string>();
        }
        // Synchronize legacy alsaDevice with structured output config
        if (outConfig.output.mode == OutputMode::Usb) {
            if (!outConfig.output.usb.preferredDevice.empty()) {
                outConfig.alsaDevice = outConfig.output.usb.preferredDevice;
            } else if (!outConfig.alsaDevice.empty()) {
                outConfig.output.usb.preferredDevice = outConfig.alsaDevice;
            }
        }

        // Multi-rate mode settings (Issue #219)
        if (j.contains("multiRateEnabled")) {
            outConfig.multiRateEnabled = j["multiRateEnabled"].get<bool>();
        }
        if (j.contains("coefficientDir")) {
            outConfig.coefficientDir = j["coefficientDir"].get<std::string>();
        }

        if (j.contains("loopback") && j["loopback"].is_object()) {
            auto lb = j["loopback"];
            try {
                if (lb.contains("enabled") && lb["enabled"].is_boolean()) {
                    outConfig.loopback.enabled = lb["enabled"].get<bool>();
                }
                if (lb.contains("device") && lb["device"].is_string()) {
                    outConfig.loopback.device = lb["device"].get<std::string>();
                }
                if (lb.contains("sampleRate") && lb["sampleRate"].is_number_integer()) {
                    outConfig.loopback.sampleRate = lb["sampleRate"].get<uint32_t>();
                }
                if (lb.contains("channels") && lb["channels"].is_number_integer()) {
                    outConfig.loopback.channels = static_cast<uint8_t>(lb["channels"].get<int>());
                }
                if (lb.contains("format") && lb["format"].is_string()) {
                    outConfig.loopback.format = lb["format"].get<std::string>();
                }
                if (lb.contains("periodFrames") && lb["periodFrames"].is_number_integer()) {
                    outConfig.loopback.periodFrames =
                        static_cast<uint32_t>(lb["periodFrames"].get<int>());
                }
            } catch (const std::exception& e) {
                if (verbose) {
                    LOG_WARN("Config: Invalid loopback settings, using defaults: {}", e.what());
                }
                outConfig.loopback = AppConfig::LoopbackInputConfig{};
            }

            if (outConfig.loopback.sampleRate == 0) {
                outConfig.loopback.sampleRate = 44100;
            }
            if (outConfig.loopback.channels == 0) {
                outConfig.loopback.channels = 2;
            }
            outConfig.loopback.channels =
                static_cast<uint8_t>(std::clamp<int>(outConfig.loopback.channels, 1, 8));
            if (outConfig.loopback.periodFrames == 0) {
                outConfig.loopback.periodFrames = 1024;
            }
        }

        if (j.contains("i2s") && j["i2s"].is_object()) {
            auto i2s = j["i2s"];
            try {
                if (i2s.contains("enabled") && i2s["enabled"].is_boolean()) {
                    outConfig.i2s.enabled = i2s["enabled"].get<bool>();
                }
                if (i2s.contains("device") && i2s["device"].is_string()) {
                    outConfig.i2s.device = i2s["device"].get<std::string>();
                }
                if (i2s.contains("sampleRate") && i2s["sampleRate"].is_number_integer()) {
                    outConfig.i2s.sampleRate = i2s["sampleRate"].get<uint32_t>();
                }
                if (i2s.contains("channels") && i2s["channels"].is_number_integer()) {
                    outConfig.i2s.channels = static_cast<uint8_t>(i2s["channels"].get<int>());
                }
                if (i2s.contains("format") && i2s["format"].is_string()) {
                    outConfig.i2s.format = i2s["format"].get<std::string>();
                }
                if (i2s.contains("periodFrames") && i2s["periodFrames"].is_number_integer()) {
                    outConfig.i2s.periodFrames =
                        static_cast<uint32_t>(i2s["periodFrames"].get<int>());
                }
            } catch (const std::exception& e) {
                if (verbose) {
                    LOG_WARN("Config: Invalid i2s settings, using defaults: {}", e.what());
                }
                outConfig.i2s = AppConfig::I2sInputConfig{};
            }

            if (outConfig.i2s.channels == 0) {
                outConfig.i2s.channels = 2;
            }
            outConfig.i2s.channels =
                static_cast<uint8_t>(std::clamp<int>(outConfig.i2s.channels, 1, 8));
            if (outConfig.i2s.periodFrames == 0) {
                outConfig.i2s.periodFrames = 1024;
            }
            // sampleRate=0 is allowed (follow runtime/negotiated rate in daemon)
        }

        // Partitioned convolution settings (Issue #351)
        if (j.contains("partitionedConvolution") && j["partitionedConvolution"].is_object()) {
            auto pc = j["partitionedConvolution"];
            try {
                if (pc.contains("enabled") && pc["enabled"].is_boolean()) {
                    outConfig.partitionedConvolution.enabled = pc["enabled"].get<bool>();
                }
                if (pc.contains("fastPartitionTaps") &&
                    pc["fastPartitionTaps"].is_number_integer()) {
                    outConfig.partitionedConvolution.fastPartitionTaps =
                        std::max(1024, pc["fastPartitionTaps"].get<int>());
                }
                if (pc.contains("minPartitionTaps") && pc["minPartitionTaps"].is_number_integer()) {
                    outConfig.partitionedConvolution.minPartitionTaps =
                        std::max(1024, pc["minPartitionTaps"].get<int>());
                }
                if (pc.contains("maxPartitions") && pc["maxPartitions"].is_number_integer()) {
                    outConfig.partitionedConvolution.maxPartitions =
                        std::max(1, pc["maxPartitions"].get<int>());
                }
                if (pc.contains("tailFftMultiple") && pc["tailFftMultiple"].is_number_integer()) {
                    int tailMultiple = pc["tailFftMultiple"].get<int>();
                    outConfig.partitionedConvolution.tailFftMultiple =
                        std::clamp(tailMultiple, 2, 16);
                }
            } catch (const std::exception& e) {
                if (verbose) {
                    LOG_WARN("Config: Invalid partitionedConvolution settings, using defaults: {}",
                             e.what());
                }
                outConfig.partitionedConvolution = AppConfig::PartitionedConvolutionConfig{};
            }
        }

        // EQ settings (type-safe parse to avoid null/number crashes)
        if (j.contains("eqEnabled") && j["eqEnabled"].is_boolean()) {
            outConfig.eqEnabled = j["eqEnabled"].get<bool>();
        }
        if (j.contains("eqProfilePath") && j["eqProfilePath"].is_string()) {
            outConfig.eqProfilePath = j["eqProfilePath"].get<std::string>();
        }

        // Crossfeed settings (with type error handling)
        if (j.contains("crossfeed") && j["crossfeed"].is_object()) {
            auto cf = j["crossfeed"];
            try {
                if (cf.contains("enabled") && cf["enabled"].is_boolean()) {
                    outConfig.crossfeed.enabled = cf["enabled"].get<bool>();
                }
                if (cf.contains("headSize") && cf["headSize"].is_string()) {
                    outConfig.crossfeed.headSize =
                        validateHeadSize(cf["headSize"].get<std::string>());
                }
                if (cf.contains("hrtfPath") && cf["hrtfPath"].is_string()) {
                    outConfig.crossfeed.hrtfPath = cf["hrtfPath"].get<std::string>();
                }
            } catch (const std::exception& e) {
                // On type error, keep defaults (already set in AppConfig{})
                if (verbose) {
                    LOG_WARN("Config: Invalid crossfeed settings, using defaults: {}", e.what());
                }
            }
        }

        // Fallback settings (Issue #139)
        if (j.contains("fallback") && j["fallback"].is_object()) {
            auto fb = j["fallback"];
            try {
                if (fb.contains("enabled") && fb["enabled"].is_boolean()) {
                    outConfig.fallback.enabled = fb["enabled"].get<bool>();
                }
                if (fb.contains("gpuThreshold") && fb["gpuThreshold"].is_number()) {
                    outConfig.fallback.gpuThreshold = fb["gpuThreshold"].get<float>();
                }
                if (fb.contains("gpuThresholdCount") &&
                    fb["gpuThresholdCount"].is_number_integer()) {
                    outConfig.fallback.gpuThresholdCount = fb["gpuThresholdCount"].get<int>();
                }
                if (fb.contains("gpuRecoveryThreshold") && fb["gpuRecoveryThreshold"].is_number()) {
                    outConfig.fallback.gpuRecoveryThreshold =
                        fb["gpuRecoveryThreshold"].get<float>();
                }
                if (fb.contains("gpuRecoveryCount") && fb["gpuRecoveryCount"].is_number_integer()) {
                    outConfig.fallback.gpuRecoveryCount = fb["gpuRecoveryCount"].get<int>();
                }
                if (fb.contains("xrunTriggersFallback") &&
                    fb["xrunTriggersFallback"].is_boolean()) {
                    outConfig.fallback.xrunTriggersFallback =
                        fb["xrunTriggersFallback"].get<bool>();
                }
                if (fb.contains("monitorIntervalMs") &&
                    fb["monitorIntervalMs"].is_number_integer()) {
                    outConfig.fallback.monitorIntervalMs = fb["monitorIntervalMs"].get<int>();
                }

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
                    LOG_WARN("Config: Invalid fallback settings, using defaults: {}", e.what());
                }
            }
        }

        // De-limiter inference backend settings (Issue #1017)
        if (j.contains("delimiter") && j["delimiter"].is_object()) {
            auto dl = j["delimiter"];
            try {
                if (dl.contains("enabled") && dl["enabled"].is_boolean()) {
                    outConfig.delimiter.enabled = dl["enabled"].get<bool>();
                }
                if (dl.contains("backend") && dl["backend"].is_string()) {
                    outConfig.delimiter.backend =
                        validateDelimiterBackend(dl["backend"].get<std::string>());
                }
                if (dl.contains("expectedSampleRate") &&
                    dl["expectedSampleRate"].is_number_integer()) {
                    int sr = dl["expectedSampleRate"].get<int>();
                    if (sr > 0) {
                        outConfig.delimiter.expectedSampleRate =
                            sanitizeDelimiterSampleRate(sr, verbose);
                    }
                }
                if (dl.contains("chunkSec") && dl["chunkSec"].is_number()) {
                    float sec = dl["chunkSec"].get<float>();
                    if (std::isfinite(sec) && sec > 0.0f) {
                        outConfig.delimiter.chunkSec = sec;
                    }
                }
                if (dl.contains("overlapSec") && dl["overlapSec"].is_number()) {
                    float sec = dl["overlapSec"].get<float>();
                    if (std::isfinite(sec) && sec > 0.0f) {
                        outConfig.delimiter.overlapSec = sec;
                    }
                }

                if (dl.contains("ort") && dl["ort"].is_object()) {
                    auto ort = dl["ort"];
                    if (ort.contains("modelPath") && ort["modelPath"].is_string()) {
                        outConfig.delimiter.ort.modelPath = ort["modelPath"].get<std::string>();
                    }
                    if (ort.contains("provider") && ort["provider"].is_string()) {
                        outConfig.delimiter.ort.provider =
                            validateOrtProvider(ort["provider"].get<std::string>());
                    }
                    if (ort.contains("intraOpThreads") &&
                        ort["intraOpThreads"].is_number_integer()) {
                        int threads = ort["intraOpThreads"].get<int>();
                        outConfig.delimiter.ort.intraOpThreads = std::max(0, threads);
                    }
                }
            } catch (const std::exception& e) {
                if (verbose) {
                    LOG_WARN("Config: Invalid delimiter settings, using defaults: {}", e.what());
                }
            }
        }

        // RTP configuration removed (legacy path). No RTP fields are parsed.

        // Clamp derived floating-point values after parsing (ensures sane bounds)
        outConfig.gain = std::max(0.0f, outConfig.gain);
        outConfig.headroomTarget =
            std::clamp(outConfig.headroomTarget, DaemonConstants::MIN_HEADROOM_TARGET,
                       DaemonConstants::MAX_HEADROOM_TARGET);

        // Ensure per-family filters fall back to the generic path if not explicitly set
        if (outConfig.filterPath44kMin.empty()) {
            outConfig.filterPath44kMin = outConfig.filterPath;
        }
        if (outConfig.filterPath48kMin.empty()) {
            outConfig.filterPath48kMin = outConfig.filterPath;
        }
        if (outConfig.filterPath44kLinear.empty()) {
            outConfig.filterPath44kLinear = outConfig.filterPath;
        }
        if (outConfig.filterPath48kLinear.empty()) {
            outConfig.filterPath48kLinear = outConfig.filterPath;
        }
        if (verbose) {
            std::cout << "Config: Loaded from " << std::filesystem::absolute(configPath) << '\n';
        }

        // Clamp derived floating-point values after parsing (ensures sane bounds)
        outConfig.headroomTarget = std::clamp(outConfig.headroomTarget, 0.01f, 1.0f);

        return true;
    } catch (const std::exception& e) {
        if (verbose) {
            LOG_ERROR("Config: Failed to parse {}: {}", configPath.string(), e.what());
        }
        return false;
    }
}
