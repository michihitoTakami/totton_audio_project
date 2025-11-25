/**
 * @file fallback_manager.h
 * @brief Dynamic Fallback Manager for GPU overload and XRUN handling (Issue #139)
 *
 * Provides automatic fallback when:
 * - GPU utilization exceeds threshold (80% default)
 * - XRUN events are detected
 *
 * Features:
 * - Hysteresis for stable state transitions
 * - ZeroMQ notification of fallback state
 * - Configurable thresholds
 */

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>

namespace FallbackManager {

/**
 * @brief Fallback state
 */
enum class FallbackState {
    Normal,   // Normal operation (GPU processing enabled)
    Fallback  // Fallback mode (GPU processing disabled/bypassed)
};

/**
 * @brief Configuration for fallback manager
 */
struct FallbackConfig {
    float gpuThreshold{80.0f};          // GPU utilization threshold (%)
    int gpuThresholdCount{3};           // Consecutive threshold exceedances to trigger fallback
    float gpuRecoveryThreshold{70.0f};  // Recovery threshold (threshold - 10%)
    int gpuRecoveryCount{5};            // Consecutive recovery measurements to return to normal
    bool xrunTriggersFallback{true};    // Whether XRUN should trigger immediate fallback
    int monitorIntervalMs{100};         // GPU monitoring interval (milliseconds)
};

/**
 * @brief Fallback manager class
 *
 * Monitors GPU utilization and XRUN events, automatically switching
 * to fallback mode when thresholds are exceeded.
 */
class Manager {
   public:
    Manager();
    ~Manager();

    /**
     * @brief Initialize the fallback manager
     *
     * @param config Configuration parameters
     * @param stateCallback Callback invoked when fallback state changes
     * @return true if initialization succeeded
     */
    bool initialize(const FallbackConfig& config, std::function<void(FallbackState)> stateCallback);

    /**
     * @brief Shutdown the fallback manager
     */
    void shutdown();

    /**
     * @brief Get current fallback state
     */
    FallbackState getState() const {
        return state_.load();
    }

    /**
     * @brief Check if currently in fallback mode
     */
    bool isInFallback() const {
        return state_.load() == FallbackState::Fallback;
    }

    /**
     * @brief Notify manager of an XRUN event
     *
     * Thread-safe, can be called from ALSA callback.
     */
    void notifyXrun();

    /**
     * @brief Get current GPU utilization
     */
    float getGpuUtilization() const {
        return gpuUtilization_.load();
    }

    /**
     * @brief Get fallback statistics
     */
    struct Stats {
        uint64_t xrunCount{0};
        uint64_t fallbackActivations{0};
        uint64_t fallbackRecoveries{0};
        std::chrono::steady_clock::time_point lastFallbackTime;
        std::chrono::steady_clock::time_point lastRecoveryTime;
    };
    Stats getStats() const;

   private:
    void monitorThread();
    void checkGpuUtilization();
    void triggerFallback();
    void triggerRecovery();

    FallbackConfig config_;
    std::function<void(FallbackState)> stateCallback_;

    std::atomic<FallbackState> state_{FallbackState::Normal};
    std::atomic<float> gpuUtilization_{0.0f};
    std::atomic<bool> running_{false};

    // GPU threshold tracking
    int consecutiveThresholdExceedances_{0};
    int consecutiveRecoveryMeasurements_{0};
    std::mutex thresholdMutex_;

    // Statistics
    std::atomic<uint64_t> xrunCount_{0};
    std::atomic<uint64_t> fallbackActivations_{0};
    std::atomic<uint64_t> fallbackRecoveries_{0};
    std::chrono::steady_clock::time_point lastFallbackTime_;
    std::chrono::steady_clock::time_point lastRecoveryTime_;
    mutable std::mutex statsMutex_;

    std::thread monitorThread_;
};

}  // namespace FallbackManager
