#include "daemon/runtime_stats.h"

#include "convolution_engine.h"
#include "daemon_constants.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

namespace runtime_stats {

namespace {

constexpr double kPeakDbfsFloor = -200.0;

double linearToDbfs(double value) {
    if (value <= 0.0) {
        return kPeakDbfsFloor;
    }
    return 20.0 * std::log10(value);
}

nlohmann::json makePeakJson(float linearValue) {
    nlohmann::json peak;
    peak["linear"] = linearValue;
    peak["dbfs"] = linearToDbfs(linearValue);
    return peak;
}

void updatePeakLevel(std::atomic<float>& peak, float candidate) {
    float absValue = std::fabs(candidate);
    float current = peak.load(std::memory_order_relaxed);
    while (absValue > current &&
           !peak.compare_exchange_weak(current, absValue, std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {}
}

int getInputRate(const Dependencies& deps) {
    if (deps.inputSampleRate) {
        return *deps.inputSampleRate;
    }
    return 0;
}

int getUpsampleRatio(const Dependencies& deps) {
    if (deps.config) {
        return deps.config->upsampleRatio;
    }
    return 0;
}

std::string determinePhaseType(const Dependencies& deps) {
    if (deps.upsampler) {
        return (deps.upsampler->getPhaseType() == PhaseType::Minimum) ? "minimum" : "linear";
    }
    if (deps.config) {
        return (deps.config->phaseType == PhaseType::Minimum) ? "minimum" : "linear";
    }
    return "minimum";
}

nlohmann::json buildFallbackJson(const Dependencies& deps) {
    nlohmann::json fallback;
    bool enabled = (deps.config) ? deps.config->fallback.enabled : false;
    fallback["enabled"] = enabled;
    fallback["active"] =
        (deps.fallbackActive && deps.fallbackActive->load(std::memory_order_relaxed));
    fallback["gpu_utilization"] = 0.0;
    fallback["monitoring_enabled"] =
        (deps.fallbackManager ? deps.fallbackManager->isMonitoringEnabled() : false);

    if (enabled && deps.fallbackManager) {
        fallback["gpu_utilization"] = deps.fallbackManager->getGpuUtilization();
        auto stats = deps.fallbackManager->getStats();
        fallback["xrun_count"] = stats.xrunCount;
        fallback["activations"] = stats.fallbackActivations;
        fallback["recoveries"] = stats.fallbackRecoveries;
    } else {
        fallback["xrun_count"] = 0;
        fallback["activations"] = 0;
        fallback["recoveries"] = 0;
    }

    return fallback;
}

}  // namespace

static std::atomic<size_t> s_clipCount{0};
static std::atomic<size_t> s_totalSamples{0};
static std::atomic<size_t> s_bufferDropped{0};

static std::atomic<float> s_peakInput{0.0f};
static std::atomic<float> s_peakUpsampler{0.0f};
static std::atomic<float> s_peakPostCrossfeed{0.0f};
static std::atomic<float> s_peakPostGain{0.0f};

void reset() {
    s_clipCount.store(0, std::memory_order_relaxed);
    s_totalSamples.store(0, std::memory_order_relaxed);
    s_bufferDropped.store(0, std::memory_order_relaxed);
    s_peakInput.store(0.0f, std::memory_order_relaxed);
    s_peakUpsampler.store(0.0f, std::memory_order_relaxed);
    s_peakPostCrossfeed.store(0.0f, std::memory_order_relaxed);
    s_peakPostGain.store(0.0f, std::memory_order_relaxed);
}

void recordClip() {
    s_clipCount.fetch_add(1, std::memory_order_relaxed);
}

void addSamples(std::size_t count) {
    s_totalSamples.fetch_add(count, std::memory_order_relaxed);
}

void addDroppedFrames(std::size_t count) {
    s_bufferDropped.fetch_add(count, std::memory_order_relaxed);
}

void updateInputPeak(float peak) {
    updatePeakLevel(s_peakInput, peak);
}

void updateUpsamplerPeak(float peak) {
    updatePeakLevel(s_peakUpsampler, peak);
}

void updatePostCrossfeedPeak(float peak) {
    updatePeakLevel(s_peakPostCrossfeed, peak);
}

void updatePostGainPeak(float peak) {
    updatePeakLevel(s_peakPostGain, peak);
}

std::size_t clipCount() {
    return s_clipCount.load(std::memory_order_relaxed);
}

std::size_t totalSamples() {
    return s_totalSamples.load(std::memory_order_relaxed);
}

std::size_t droppedFrames() {
    return s_bufferDropped.load(std::memory_order_relaxed);
}

nlohmann::json collect(const Dependencies& deps, std::size_t bufferCapacityFrames) {
    std::size_t clips = clipCount();
    std::size_t total = totalSamples();
    double clipRate = (total > 0) ? (static_cast<double>(clips) / total) : 0.0;

    int inputRate = getInputRate(deps);
    int upsampleRatio = getUpsampleRatio(deps);
    int outputRate = (inputRate > 0 && upsampleRatio > 0) ? inputRate * upsampleRatio : 0;

    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    std::string phaseTypeStr = determinePhaseType(deps);

    nlohmann::json stats;
    stats["clip_count"] = clips;
    stats["total_samples"] = total;
    stats["clip_rate"] = clipRate;
    stats["input_rate"] = inputRate;
    stats["output_rate"] = outputRate;
    stats["upsample_ratio"] = upsampleRatio;
    stats["eq_enabled"] = (deps.config ? deps.config->eqEnabled : false);
    stats["phase_type"] = phaseTypeStr;
    stats["last_updated"] = epoch;

    nlohmann::json gain;
    gain["user"] = (deps.config ? deps.config->gain : 0.0f);
    gain["headroom"] =
        (deps.headroomGain ? deps.headroomGain->load(std::memory_order_relaxed) : 0.0f);
    gain["headroom_effective"] =
        (deps.outputGain ? deps.outputGain->load(std::memory_order_relaxed) : 0.0f);
    gain["limiter"] = (deps.limiterGain ? deps.limiterGain->load(std::memory_order_relaxed) : 0.0f);
    gain["effective"] =
        (deps.effectiveGain ? deps.effectiveGain->load(std::memory_order_relaxed) : 0.0f);
    gain["target_peak"] = (deps.config ? deps.config->headroomTarget : 0.0f);
    gain["metadata_peak"] = (deps.headroomCache ? deps.headroomCache->getTargetPeak() : 0.0f);
    stats["gain"] = gain;

    nlohmann::json peaks;
    peaks["input"] = makePeakJson(s_peakInput.load(std::memory_order_relaxed));
    peaks["upsampler"] = makePeakJson(s_peakUpsampler.load(std::memory_order_relaxed));
    peaks["post_mix"] = makePeakJson(s_peakPostCrossfeed.load(std::memory_order_relaxed));
    peaks["post_gain"] = makePeakJson(s_peakPostGain.load(std::memory_order_relaxed));
    stats["peaks"] = peaks;

    stats["fallback"] = buildFallbackJson(deps);

    if (deps.dacManager) {
        stats["dac"] = deps.dacManager->buildStatsSummaryJson();
    } else {
        stats["dac"] = nlohmann::json::object();
    }

    nlohmann::json buffer;
    buffer["max_seconds"] = DaemonConstants::MAX_OUTPUT_BUFFER_SECONDS;
    buffer["capacity_frames"] = static_cast<std::size_t>(bufferCapacityFrames);
    buffer["dropped_frames"] = droppedFrames();
    stats["buffer"] = buffer;

    return stats;
}

void writeStatsFile(const Dependencies& deps, std::size_t bufferCapacityFrames,
                    const std::string& path) {
    auto stats = collect(deps, bufferCapacityFrames);
    std::string tmpPath = path + ".tmp";
    std::ofstream ofs(tmpPath);
    if (!ofs) {
        return;
    }
    ofs << stats.dump(2) << std::endl;
    ofs.close();
    std::rename(tmpPath.c_str(), path.c_str());
}

}  // namespace runtime_stats
