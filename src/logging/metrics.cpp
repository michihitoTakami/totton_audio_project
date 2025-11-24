/**
 * @file metrics.cpp
 * @brief Implementation of metrics collection for GPU Audio Upsampler (Issue #43)
 */

#include "logging/metrics.h"

#include "logging/logger.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>

#ifdef HAVE_NVML
#include <nvml.h>
#endif

namespace gpu_upsampler {
namespace metrics {

namespace {

// Global metrics storage
AudioStats g_audioStats;
BufferMetrics g_bufferMetrics;
std::mutex g_bufferMutex;

// Timing metrics with mutex protection
struct TimingState {
    double avgProcessingTime{0.0};
    double maxProcessingTime{0.0};
    double minProcessingTime{std::numeric_limits<double>::max()};
    uint64_t blockCount{0};
    double totalProcessingTime{0.0};
    std::mutex mutex;
};
TimingState g_timingState;

// Audio configuration
struct AudioConfig {
    std::atomic<uint32_t> inputRate{0};
    std::atomic<uint32_t> outputRate{0};
    std::atomic<uint32_t> upsampleRatio{0};
};
AudioConfig g_audioConfig;

// NVML state
#ifdef HAVE_NVML
bool g_nvmlInitialized = false;
nvmlDevice_t g_nvmlDevice = nullptr;
std::mutex g_nvmlMutex;
#endif

// GPU metrics cache (to avoid slow NVML calls)
GpuMetrics g_gpuMetricsCache;
std::mutex g_gpuMetricsCacheMutex;
std::chrono::steady_clock::time_point g_lastGpuMetricsUpdate;
constexpr std::chrono::milliseconds GPU_METRICS_CACHE_DURATION{1000};  // 1 second cache

}  // namespace

// ============================================================
// Global metrics accessors
// ============================================================

AudioStats& getAudioStats() {
    return g_audioStats;
}

GpuMetrics getGpuMetrics() {
    auto now = std::chrono::steady_clock::now();

    // Fast path: return cached metrics if still fresh
    {
        std::lock_guard<std::mutex> lock(g_gpuMetricsCacheMutex);
        if (now - g_lastGpuMetricsUpdate < GPU_METRICS_CACHE_DURATION) {
            return g_gpuMetricsCache;
        }
    }

    // Slow path: update metrics from NVML
    GpuMetrics metrics;

#ifdef HAVE_NVML
    std::lock_guard<std::mutex> nvmlLock(g_nvmlMutex);

    if (!g_nvmlInitialized) {
        initializeNvml();
    }

    if (g_nvmlInitialized && g_nvmlDevice != nullptr) {
        metrics.available = true;

        // Get utilization
        nvmlUtilization_t utilization;
        if (nvmlDeviceGetUtilizationRates(g_nvmlDevice, &utilization) == NVML_SUCCESS) {
            metrics.utilization = static_cast<float>(utilization.gpu);
        }

        // Get memory info
        nvmlMemory_t memory;
        if (nvmlDeviceGetMemoryInfo(g_nvmlDevice, &memory) == NVML_SUCCESS) {
            metrics.memoryUsed = static_cast<float>(memory.used) / (1024.0f * 1024.0f);
            metrics.memoryTotal = static_cast<float>(memory.total) / (1024.0f * 1024.0f);
        }

        // Get temperature
        unsigned int temp;
        if (nvmlDeviceGetTemperature(g_nvmlDevice, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            metrics.temperature = static_cast<float>(temp);
        }
    }
#else
    metrics.available = false;
#endif

    // Update cache
    {
        std::lock_guard<std::mutex> lock(g_gpuMetricsCacheMutex);
        g_gpuMetricsCache = metrics;
        g_lastGpuMetricsUpdate = now;
    }

    return metrics;
}

BufferMetrics getBufferMetrics() {
    std::lock_guard<std::mutex> lock(g_bufferMutex);
    return g_bufferMetrics;
}

TimingMetrics getTimingMetrics() {
    std::lock_guard<std::mutex> lock(g_timingState.mutex);
    TimingMetrics metrics;
    metrics.avgProcessingTimeMs = g_timingState.avgProcessingTime;
    metrics.maxProcessingTimeMs = g_timingState.maxProcessingTime;
    metrics.minProcessingTimeMs = g_timingState.minProcessingTime;
    metrics.blockCount = g_timingState.blockCount;

    // Handle case where no blocks have been processed yet
    if (metrics.blockCount == 0) {
        metrics.minProcessingTimeMs = 0.0;
    }

    return metrics;
}

MetricsSnapshot getSnapshot() {
    MetricsSnapshot snapshot;

    // Copy audio stats atomically into non-atomic snapshot
    snapshot.audio.totalSamples = g_audioStats.totalSamples.load(std::memory_order_relaxed);
    snapshot.audio.clipCount = g_audioStats.clipCount.load(std::memory_order_relaxed);
    snapshot.audio.xrunCount = g_audioStats.xrunCount.load(std::memory_order_relaxed);
    snapshot.audio.bufferUnderflows = g_audioStats.bufferUnderflows.load(std::memory_order_relaxed);
    snapshot.audio.bufferOverflows = g_audioStats.bufferOverflows.load(std::memory_order_relaxed);

    snapshot.gpu = getGpuMetrics();
    snapshot.buffer = getBufferMetrics();
    snapshot.timing = getTimingMetrics();
    snapshot.timestamp = std::chrono::system_clock::now();

    snapshot.inputRate = g_audioConfig.inputRate.load(std::memory_order_relaxed);
    snapshot.outputRate = g_audioConfig.outputRate.load(std::memory_order_relaxed);
    snapshot.upsampleRatio = g_audioConfig.upsampleRatio.load(std::memory_order_relaxed);

    return snapshot;
}

// ============================================================
// Metric recording functions
// ============================================================

void recordClip() {
    g_audioStats.clipCount.fetch_add(1, std::memory_order_relaxed);
}

void recordClips(uint64_t count) {
    g_audioStats.clipCount.fetch_add(count, std::memory_order_relaxed);
}

void recordSamples(uint64_t count) {
    g_audioStats.totalSamples.fetch_add(count, std::memory_order_relaxed);
}

void recordXrun() {
    g_audioStats.xrunCount.fetch_add(1, std::memory_order_relaxed);
    // Note: Logging removed to avoid I/O in audio thread
    // Use metrics snapshot to monitor XRUN count
}

void recordBufferUnderflow() {
    g_audioStats.bufferUnderflows.fetch_add(1, std::memory_order_relaxed);
    // Note: Logging removed to avoid I/O in audio thread
}

void recordBufferOverflow() {
    g_audioStats.bufferOverflows.fetch_add(1, std::memory_order_relaxed);
    // Note: Logging removed to avoid I/O in audio thread
}

void updateBufferLevels(size_t inputFill, size_t inputCap, size_t outputFill, size_t outputCap) {
    std::lock_guard<std::mutex> lock(g_bufferMutex);
    g_bufferMetrics.inputFillLevel = inputFill;
    g_bufferMetrics.inputCapacity = inputCap;
    g_bufferMetrics.outputFillLevel = outputFill;
    g_bufferMetrics.outputCapacity = outputCap;

    if (inputCap > 0) {
        g_bufferMetrics.inputFillPercent =
            100.0f * static_cast<float>(inputFill) / static_cast<float>(inputCap);
    }
    if (outputCap > 0) {
        g_bufferMetrics.outputFillPercent =
            100.0f * static_cast<float>(outputFill) / static_cast<float>(outputCap);
    }
}

void recordProcessingTime(double timeMs) {
    std::lock_guard<std::mutex> lock(g_timingState.mutex);

    g_timingState.blockCount++;
    g_timingState.totalProcessingTime += timeMs;
    g_timingState.avgProcessingTime =
        g_timingState.totalProcessingTime / static_cast<double>(g_timingState.blockCount);

    if (timeMs > g_timingState.maxProcessingTime) {
        g_timingState.maxProcessingTime = timeMs;
    }

    if (timeMs < g_timingState.minProcessingTime) {
        g_timingState.minProcessingTime = timeMs;
    }
}

void setAudioConfig(uint32_t inputRate, uint32_t outputRate, uint32_t upsampleRatio) {
    g_audioConfig.inputRate.store(inputRate, std::memory_order_relaxed);
    g_audioConfig.outputRate.store(outputRate, std::memory_order_relaxed);
    g_audioConfig.upsampleRatio.store(upsampleRatio, std::memory_order_relaxed);
}

// ============================================================
// Stats file output
// ============================================================

nlohmann::json toJson() {
    auto snapshot = getSnapshot();

    nlohmann::json json;

    // Audio stats (snapshot is non-atomic)
    json["audio"]["totalSamples"] = snapshot.audio.totalSamples;
    json["audio"]["clipCount"] = snapshot.audio.clipCount;
    json["audio"]["xrunCount"] = snapshot.audio.xrunCount;
    json["audio"]["bufferUnderflows"] = snapshot.audio.bufferUnderflows;
    json["audio"]["bufferOverflows"] = snapshot.audio.bufferOverflows;

    // Calculate clip rate
    uint64_t total = snapshot.audio.totalSamples;
    uint64_t clips = snapshot.audio.clipCount;
    double clipRate = (total > 0) ? (static_cast<double>(clips) / static_cast<double>(total)) : 0.0;
    json["audio"]["clipRate"] = clipRate;

    // GPU metrics
    json["gpu"]["available"] = snapshot.gpu.available;
    if (snapshot.gpu.available) {
        json["gpu"]["utilization"] = snapshot.gpu.utilization;
        json["gpu"]["memoryUsedMB"] = snapshot.gpu.memoryUsed;
        json["gpu"]["memoryTotalMB"] = snapshot.gpu.memoryTotal;
        json["gpu"]["temperatureC"] = snapshot.gpu.temperature;
    }

    // Buffer metrics
    json["buffer"]["inputFillLevel"] = snapshot.buffer.inputFillLevel;
    json["buffer"]["inputCapacity"] = snapshot.buffer.inputCapacity;
    json["buffer"]["inputFillPercent"] = snapshot.buffer.inputFillPercent;
    json["buffer"]["outputFillLevel"] = snapshot.buffer.outputFillLevel;
    json["buffer"]["outputCapacity"] = snapshot.buffer.outputCapacity;
    json["buffer"]["outputFillPercent"] = snapshot.buffer.outputFillPercent;

    // Timing metrics
    json["timing"]["avgProcessingTimeMs"] = snapshot.timing.avgProcessingTimeMs;
    json["timing"]["maxProcessingTimeMs"] = snapshot.timing.maxProcessingTimeMs;
    json["timing"]["minProcessingTimeMs"] = snapshot.timing.minProcessingTimeMs;
    json["timing"]["blockCount"] = snapshot.timing.blockCount;

    // Audio configuration
    json["config"]["inputRate"] = snapshot.inputRate;
    json["config"]["outputRate"] = snapshot.outputRate;
    json["config"]["upsampleRatio"] = snapshot.upsampleRatio;

    // Timestamp
    auto time_t = std::chrono::system_clock::to_time_t(snapshot.timestamp);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    json["lastUpdated"] = oss.str();

    return json;
}

std::string toJsonString() {
    return toJson().dump(2);
}

bool writeStatsFile(const std::string& filePath) {
    try {
        std::ofstream file(filePath);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open stats file for writing: {}", filePath);
            return false;
        }

        file << toJson().dump(2);
        file.close();

        LOG_DEBUG("Stats file written: {}", filePath);
        return true;
    } catch (const std::exception& ex) {
        LOG_ERROR("Failed to write stats file: {}", ex.what());
        return false;
    }
}

// ============================================================
// Reset functions
// ============================================================

void reset() {
    resetAudioStats();
    resetTimingMetrics();

    {
        std::lock_guard<std::mutex> lock(g_bufferMutex);
        g_bufferMetrics = BufferMetrics{};
    }

    LOG_INFO("All metrics reset");
}

void resetAudioStats() {
    g_audioStats.totalSamples.store(0, std::memory_order_relaxed);
    g_audioStats.clipCount.store(0, std::memory_order_relaxed);
    g_audioStats.xrunCount.store(0, std::memory_order_relaxed);
    g_audioStats.bufferUnderflows.store(0, std::memory_order_relaxed);
    g_audioStats.bufferOverflows.store(0, std::memory_order_relaxed);
}

void resetTimingMetrics() {
    std::lock_guard<std::mutex> lock(g_timingState.mutex);
    g_timingState.avgProcessingTime = 0.0;
    g_timingState.maxProcessingTime = 0.0;
    g_timingState.minProcessingTime = std::numeric_limits<double>::max();
    g_timingState.blockCount = 0;
    g_timingState.totalProcessingTime = 0.0;
}

// ============================================================
// NVML support
// ============================================================

bool isNvmlAvailable() {
#ifdef HAVE_NVML
    return g_nvmlInitialized;
#else
    return false;
#endif
}

bool initializeNvml() {
#ifdef HAVE_NVML
    std::lock_guard<std::mutex> lock(g_nvmlMutex);

    if (g_nvmlInitialized) {
        return true;
    }

    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        LOG_WARN("Failed to initialize NVML: {}", nvmlErrorString(result));
        return false;
    }

    // Get the first GPU device
    result = nvmlDeviceGetHandleByIndex(0, &g_nvmlDevice);
    if (result != NVML_SUCCESS) {
        LOG_WARN("Failed to get NVML device handle: {}", nvmlErrorString(result));
        nvmlShutdown();
        return false;
    }

    g_nvmlInitialized = true;
    LOG_INFO("NVML initialized successfully");
    return true;
#else
    return false;
#endif
}

void shutdownNvml() {
#ifdef HAVE_NVML
    std::lock_guard<std::mutex> lock(g_nvmlMutex);

    if (g_nvmlInitialized) {
        nvmlShutdown();
        g_nvmlInitialized = false;
        g_nvmlDevice = nullptr;
        LOG_INFO("NVML shutdown");
    }
#endif
}

}  // namespace metrics
}  // namespace gpu_upsampler
