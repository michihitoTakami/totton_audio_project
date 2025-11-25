/**
 * @file fallback_manager.cpp
 * @brief Implementation of Dynamic Fallback Manager (Issue #139)
 */

#include "fallback_manager.h"

#include "logging/logger.h"
#include "logging/metrics.h"

#include <chrono>

namespace FallbackManager {

Manager::Manager() = default;

Manager::~Manager() {
    shutdown();
}

bool Manager::initialize(const FallbackConfig& config,
                         std::function<void(FallbackState)> stateCallback) {
    if (running_.load()) {
        LOG_WARN("Fallback manager already initialized");
        return false;
    }

    config_ = config;
    stateCallback_ = std::move(stateCallback);

    bool monitoringReady = gpu_upsampler::metrics::isNvmlAvailable();
    if (!monitoringReady) {
        monitoringReady = gpu_upsampler::metrics::initializeNvml();
    }
    gpuMonitoringEnabled_.store(monitoringReady, std::memory_order_relaxed);
    if (!monitoringReady) {
        LOG_WARN(
            "NVML not available - GPU utilization monitoring disabled (XRUN fallback still "
            "active)");
    }

    running_.store(true);
    monitorThread_ = std::thread(&Manager::monitorThread, this);

    LOG_INFO("Fallback manager initialized (GPU threshold: {}%, XRUN fallback: {})",
             config_.gpuThreshold, config_.xrunTriggersFallback);

    return true;
}

void Manager::shutdown() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    if (monitorThread_.joinable()) {
        monitorThread_.join();
    }

    LOG_INFO("Fallback manager shutdown");
}

void Manager::notifyXrun() {
    xrunCount_.fetch_add(1, std::memory_order_relaxed);

    // XRUN-triggered fallback works regardless of NVML availability
    // NVML is only required for GPU utilization-based hysteresis (threshold monitoring)
    // XRUN detection itself does not depend on NVML
    if (running_.load() && config_.xrunTriggersFallback && state_.load() == FallbackState::Normal) {
        LOG_WARN("XRUN detected - triggering fallback");
        triggerFallback();
    }
}

FallbackManager::Manager::Stats Manager::getStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    Stats stats;
    stats.xrunCount = xrunCount_.load();
    stats.fallbackActivations = fallbackActivations_.load();
    stats.fallbackRecoveries = fallbackRecoveries_.load();
    stats.lastFallbackTime = lastFallbackTime_;
    stats.lastRecoveryTime = lastRecoveryTime_;
    return stats;
}

void Manager::monitorThread() {
    if (!gpuMonitoringEnabled_.load(std::memory_order_relaxed)) {
        LOG_INFO("Fallback monitor thread started (GPU monitoring disabled)");
    } else {
        LOG_INFO("Fallback monitor thread started");
    }

    while (running_.load()) {
        checkGpuUtilization();

        // Sleep for monitoring interval
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.monitorIntervalMs));
    }

    LOG_INFO("Fallback monitor thread stopped");
}

void Manager::checkGpuUtilization() {
    if (!gpuMonitoringEnabled_.load(std::memory_order_relaxed)) {
        return;
    }

    // Get current GPU utilization
    auto gpuMetrics = gpu_upsampler::metrics::getGpuMetrics();
    float utilization = gpuMetrics.utilization;
    gpuUtilization_.store(utilization, std::memory_order_relaxed);

    if (!gpuMetrics.available) {
        // NVML not available - skip GPU-based fallback
        return;
    }

    // Determine action under lock, but execute callback outside lock
    // to prevent blocking and potential deadlock
    enum class Action { None, TriggerFallback, TriggerRecovery };
    Action action = Action::None;

    {
        std::lock_guard<std::mutex> lock(thresholdMutex_);

        FallbackState currentState = state_.load();

        if (currentState == FallbackState::Normal) {
            // Check if threshold exceeded
            if (utilization >= config_.gpuThreshold) {
                consecutiveThresholdExceedances_++;
                consecutiveRecoveryMeasurements_ = 0;

                if (consecutiveThresholdExceedances_ >= config_.gpuThresholdCount) {
                    LOG_WARN(
                        "GPU utilization {}% exceeds threshold {}% ({} consecutive) - triggering "
                        "fallback",
                        utilization, config_.gpuThreshold, consecutiveThresholdExceedances_);
                    action = Action::TriggerFallback;
                }
            } else {
                consecutiveThresholdExceedances_ = 0;
            }
        } else {
            // In fallback mode - check for recovery
            if (utilization <= config_.gpuRecoveryThreshold) {
                consecutiveRecoveryMeasurements_++;
                consecutiveThresholdExceedances_ = 0;

                if (consecutiveRecoveryMeasurements_ >= config_.gpuRecoveryCount) {
                    LOG_INFO(
                        "GPU utilization {}% below recovery threshold {}% ({} consecutive) - "
                        "returning to normal",
                        utilization, config_.gpuRecoveryThreshold,
                        consecutiveRecoveryMeasurements_);
                    action = Action::TriggerRecovery;
                }
            } else {
                consecutiveRecoveryMeasurements_ = 0;
            }
        }
    }  // Release thresholdMutex_ before calling trigger functions

    // Execute action outside the lock to prevent blocking
    if (action == Action::TriggerFallback) {
        triggerFallback();
    } else if (action == Action::TriggerRecovery) {
        triggerRecovery();
    }
}

void Manager::triggerFallback() {
    FallbackState oldState = state_.exchange(FallbackState::Fallback);
    if (oldState == FallbackState::Normal) {
        fallbackActivations_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(statsMutex_);
            lastFallbackTime_ = std::chrono::steady_clock::now();
        }

        if (stateCallback_) {
            stateCallback_(FallbackState::Fallback);
        }

        LOG_WARN("Fallback mode activated");
    }
}

void Manager::triggerRecovery() {
    FallbackState oldState = state_.exchange(FallbackState::Normal);
    if (oldState == FallbackState::Fallback) {
        fallbackRecoveries_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(statsMutex_);
            lastRecoveryTime_ = std::chrono::steady_clock::now();
        }

        if (stateCallback_) {
            stateCallback_(FallbackState::Normal);
        }

        LOG_INFO("Recovered from fallback mode - returning to normal operation");
    }
}

}  // namespace FallbackManager
