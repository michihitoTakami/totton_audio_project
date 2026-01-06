/**
 * @file metrics.h
 * @brief Metrics collection API for GPU Audio Upsampler (Issue #43)
 *
 * Provides unified metrics collection for monitoring:
 * - GPU utilization (via NVML when available)
 * - Audio buffer states
 * - XRUN detection
 * - Clipping statistics
 * - Processing latency
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

namespace gpu_upsampler {
namespace metrics {

/**
 * @brief Audio processing statistics (atomic for thread-safe updates)
 */
struct AudioStats {
    std::atomic<uint64_t> totalSamples{0};         // Total samples processed
    std::atomic<uint64_t> clipCount{0};            // Number of clipped samples
    std::atomic<uint64_t> xrunCount{0};            // Number of XRUNs detected
    std::atomic<uint64_t> captureXrunCount{0};     // XRUNs at capture/input stage
    std::atomic<uint64_t> processingXrunCount{0};  // XRUNs at processing/ring buffer
    std::atomic<uint64_t> outputXrunCount{0};      // XRUNs at output/DAC stage
    std::atomic<uint64_t> bufferUnderflows{0};     // Buffer underflow events
    std::atomic<uint64_t> bufferOverflows{0};      // Buffer overflow events
};

/**
 * @brief Audio statistics snapshot (non-atomic, for reporting)
 */
struct AudioStatsSnapshot {
    uint64_t totalSamples{0};
    uint64_t clipCount{0};
    uint64_t xrunCount{0};
    uint64_t captureXrunCount{0};
    uint64_t processingXrunCount{0};
    uint64_t outputXrunCount{0};
    uint64_t bufferUnderflows{0};
    uint64_t bufferOverflows{0};
};

/**
 * @brief GPU utilization metrics
 */
struct GpuMetrics {
    float utilization{0.0f};  // GPU utilization percentage (0-100)
    float memoryUsed{0.0f};   // Memory used in MB
    float memoryTotal{0.0f};  // Total memory in MB
    float temperature{0.0f};  // GPU temperature in Celsius
    bool available{false};    // Whether NVML is available
};

/**
 * @brief Buffer state metrics
 */
struct BufferMetrics {
    size_t inputFillLevel{0};       // Input buffer fill level (samples)
    size_t inputCapacity{0};        // Input buffer capacity (samples)
    size_t outputFillLevel{0};      // Output buffer fill level (samples)
    size_t outputCapacity{0};       // Output buffer capacity (samples)
    float inputFillPercent{0.0f};   // Input buffer fill percentage
    float outputFillPercent{0.0f};  // Output buffer fill percentage
};

/**
 * @brief Processing timing metrics
 */
struct TimingMetrics {
    double avgProcessingTimeMs{0.0};  // Average processing time per block
    double maxProcessingTimeMs{0.0};  // Maximum processing time
    double minProcessingTimeMs{0.0};  // Minimum processing time
    uint64_t blockCount{0};           // Number of blocks processed
};

/**
 * @brief Combined metrics snapshot
 */
struct MetricsSnapshot {
    AudioStatsSnapshot audio;  // Non-atomic snapshot
    GpuMetrics gpu;
    BufferMetrics buffer;
    TimingMetrics timing;
    std::chrono::system_clock::time_point timestamp;

    // Audio configuration info
    uint32_t inputRate{0};
    uint32_t outputRate{0};
    uint32_t upsampleRatio{0};
};

// ============================================================
// Global metrics instance
// ============================================================

/**
 * @brief Get the global audio statistics
 *
 * Thread-safe atomic counters for audio statistics.
 * Designed for lock-free updates from audio callback.
 */
AudioStats& getAudioStats();

/**
 * @brief Get current GPU metrics
 *
 * Queries NVML if available, returns stub values otherwise.
 * Note: This call may be slow, don't call from audio thread.
 */
GpuMetrics getGpuMetrics();

/**
 * @brief Get current buffer metrics
 *
 * Returns current buffer fill levels.
 */
BufferMetrics getBufferMetrics();

/**
 * @brief Get timing metrics
 */
TimingMetrics getTimingMetrics();

/**
 * @brief Get complete metrics snapshot
 */
MetricsSnapshot getSnapshot();

// ============================================================
// Metric recording functions (for use by audio processing code)
// ============================================================

/**
 * @brief Record a clipped sample
 *
 * Thread-safe, lock-free. Safe to call from audio callback.
 */
void recordClip();

/**
 * @brief Record multiple clipped samples
 *
 * Thread-safe, lock-free. Safe to call from audio callback.
 */
void recordClips(uint64_t count);

/**
 * @brief Record samples processed
 *
 * Thread-safe, lock-free. Safe to call from audio callback.
 */
void recordSamples(uint64_t count);

/**
 * @brief Record an XRUN event
 *
 * Thread-safe, lock-free.
 */
void recordXrun();

/**
 * @brief Record an XRUN at the capture/input stage (I2S/loopback)
 */
void recordCaptureXrun();

/**
 * @brief Record an XRUN in processing/ring buffers
 */
void recordProcessingXrun();

/**
 * @brief Record an XRUN at the output/DAC stage
 */
void recordOutputXrun();

/**
 * @brief Record a buffer underflow
 *
 * Thread-safe, lock-free.
 */
void recordBufferUnderflow();

/**
 * @brief Record a buffer overflow
 *
 * Thread-safe, lock-free.
 */
void recordBufferOverflow();

/**
 * @brief Update buffer fill levels
 *
 * @param inputFill Input buffer fill level (samples)
 * @param inputCap Input buffer capacity (samples)
 * @param outputFill Output buffer fill level (samples)
 * @param outputCap Output buffer capacity (samples)
 */
void updateBufferLevels(size_t inputFill, size_t inputCap, size_t outputFill, size_t outputCap);

/**
 * @brief Record processing time for a block
 *
 * @param timeMs Processing time in milliseconds
 */
void recordProcessingTime(double timeMs);

/**
 * @brief Set audio configuration info
 */
void setAudioConfig(uint32_t inputRate, uint32_t outputRate, uint32_t upsampleRatio);

// ============================================================
// Stats file output (for Web UI integration)
// ============================================================

/**
 * @brief Write metrics to JSON file
 *
 * Writes current metrics to the specified file path.
 * Used for Web UI integration (Issue #48).
 *
 * @param filePath Output file path
 * @return true if write succeeded
 */
bool writeStatsFile(const std::string& filePath);

/**
 * @brief Get metrics as JSON object
 *
 * @return JSON object containing all metrics
 */
nlohmann::json toJson();

/**
 * @brief Get metrics as JSON string
 *
 * @return JSON string containing all metrics
 */
std::string toJsonString();

// ============================================================
// Reset functions
// ============================================================

/**
 * @brief Reset all metrics to zero
 */
void reset();

/**
 * @brief Reset audio statistics only
 */
void resetAudioStats();

/**
 * @brief Reset timing metrics only
 */
void resetTimingMetrics();

// ============================================================
// NVML support
// ============================================================

/**
 * @brief Check if NVML is available
 *
 * @return true if NVML is available for GPU monitoring
 */
bool isNvmlAvailable();

/**
 * @brief Initialize NVML
 *
 * Called automatically when needed.
 *
 * @return true if initialization succeeded
 */
bool initializeNvml();

/**
 * @brief Shutdown NVML
 */
void shutdownNvml();

}  // namespace metrics
}  // namespace gpu_upsampler
