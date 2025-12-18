#ifndef STREAMING_CACHE_MANAGER_H
#define STREAMING_CACHE_MANAGER_H

#include "audio/input_stall_detector.h"
#include "logging/logger.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>

namespace streaming_cache {

struct StreamingCacheDependencies {
    // 非RTスレッドで実行される。重いリセット/ロック取得はここで行う。
    // 戻り値=false の場合、flush は未完了として次の drain で再試行される。
    std::function<bool(std::chrono::nanoseconds gap)> flushAction;
};

class StreamingCacheManager {
   public:
    explicit StreamingCacheManager(const StreamingCacheDependencies& deps) : deps_(deps) {}

    void handleInputBlock() {
        auto now = std::chrono::steady_clock::now();
        std::int64_t nowNs = AudioInput::toNanoseconds(now);
        std::int64_t previousNs = lastInputTimestampNs_.exchange(nowNs, std::memory_order_acq_rel);
        if (previousNs <= 0) {
            // 初回サンプルで開始時刻を記録（ギャップ判定のグレース適用のため）
            std::int64_t expected = 0;
            (void)firstInputTimestampNs_.compare_exchange_strong(expected, nowNs,
                                                                 std::memory_order_acq_rel);
            return;
        }

        std::int64_t startNs = firstInputTimestampNs_.load(std::memory_order_acquire);
        if (startNs > 0 && (nowNs - startNs) < kInitialStallGraceNs) {
            // 起動直後のギャップは無視してポップノイズを回避
            return;
        }

        if (AudioInput::shouldResetAfterStall(previousNs, nowNs)) {
            std::chrono::nanoseconds gap(nowNs - previousNs);
            flushCaches(gap);
            return;
        }

        // Non-RT thread will drain pending flush requests.
    }

    void flushCaches(std::chrono::nanoseconds gap = std::chrono::nanoseconds::zero()) {
        pendingGapNs_.store(static_cast<std::int64_t>(gap.count()), std::memory_order_release);
        externalFlushRequested_.store(true, std::memory_order_release);
    }

    bool hasPendingFlush() const {
        return externalFlushRequested_.load(std::memory_order_acquire);
    }

    // Call from a non-RT thread (e.g., main loop).
    bool drainFlushRequests() {
        if (!externalFlushRequested_.load(std::memory_order_acquire)) {
            return false;
        }

        std::int64_t gapNs = pendingGapNs_.exchange(0, std::memory_order_acq_rel);
        std::chrono::nanoseconds gap(gapNs > 0 ? gapNs : 0);

        // Reset timestamps so a fresh grace window applies after flush.
        lastInputTimestampNs_.store(0, std::memory_order_release);
        firstInputTimestampNs_.store(0, std::memory_order_release);

        bool ok = true;
        if (deps_.flushAction) {
            ok = deps_.flushAction(gap);
        }
        if (ok) {
            externalFlushRequested_.store(false, std::memory_order_release);
            double gapMs = static_cast<double>(gap.count()) / 1'000'000.0;
            LOG_INFO("[Stream] Flush requested (gap {:.3f}ms) → completed", gapMs);
        } else {
            externalFlushRequested_.store(true, std::memory_order_release);
            pendingGapNs_.store(static_cast<std::int64_t>(gap.count()), std::memory_order_release);
            LOG_WARN("[Stream] Flush requested but could not complete (will retry)");
        }
        return true;
    }

   private:
    StreamingCacheDependencies deps_;
    std::atomic<std::int64_t> lastInputTimestampNs_{0};
    std::atomic<std::int64_t> firstInputTimestampNs_{0};
    std::atomic<bool> externalFlushRequested_{false};
    std::atomic<std::int64_t> pendingGapNs_{0};

    static constexpr std::int64_t kInitialStallGraceNs = 1'500'000'000;  // 1.5s
};

}  // namespace streaming_cache

#endif  // STREAMING_CACHE_MANAGER_H
