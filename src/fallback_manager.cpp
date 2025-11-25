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

    // Initialize NVML if available
    if (!gpu_upsampler::metrics::isNvmlAvailable()) {
        if (!gpu_upsampler::metrics::initializeNvml()) {
            LOG_WARN("NVML not available - GPU monitoring disabled");
        }
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

    if (config_.xrunTriggersFallback && state_.load() == FallbackState::Normal) {
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
    LOG_INFO("Fallback monitor thread started");

    while (running_.load()) {
        checkGpuUtilization();

        // Sleep for monitoring interval
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config_.monitorIntervalMs));
    }

    LOG_INFO("Fallback monitor thread stopped");
}

void Manager::checkGpuUtilization() {
    // Get current GPU utilization
    auto gpuMetrics = gpu_upsampler::metrics::getGpuMetrics();
    float utilization = gpuMetrics.utilization;
    gpuUtilization_.store(utilization, std::memory_order_relaxed);

    if (!gpuMetrics.available) {
        // NVML not available - skip GPU-based fallback
        return;
    }

    std::lock_guard<std::mutex> lock(thresholdMutex_);

    FallbackState currentState = state_.load();

    if (currentState == FallbackState::Normal) {
        // Check if threshold exceeded
        if (utilization >= config_.gpuThreshold) {
            consecutiveThresholdExceedances_++;
            consecutiveRecoveryMeasurements_ = 0;

            if (consecutiveThresholdExceedances_ >= config_.gpuThresholdCount) {
                LOG_WARN("GPU utilization {}% exceeds threshold {}% ({} consecutive) - triggering fallback",
                         utilization, config_.gpuThreshold, consecutiveThresholdExceedances_);
                triggerFallback();
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
                LOG_INFO("GPU utilization {}% below recovery threshold {}% ({} consecutive) - returning to normal",
                         utilization, config_.gpuRecoveryThreshold, consecutiveRecoveryMeasurements_);
                triggerRecovery();
            }
        } else {
            consecutiveRecoveryMeasurements_ = 0;
        }
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

