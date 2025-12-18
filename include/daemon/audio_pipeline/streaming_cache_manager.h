#ifndef STREAMING_CACHE_MANAGER_H
#define STREAMING_CACHE_MANAGER_H

#include "audio/input_stall_detector.h"
#include "logging/logger.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>

namespace streaming_cache {

struct StreamingCacheDependencies {
    // RT スレッドで実行される。ブロッキング/長時間処理は避けること。
    std::function<void(std::chrono::nanoseconds gap)> flushAction;
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
            // 外部フラッシュ要求だけは初回でも処理できる
            maybeFlushExternal(std::chrono::nanoseconds::zero());
            return;
        }

        std::int64_t startNs = firstInputTimestampNs_.load(std::memory_order_acquire);
        if (startNs > 0 && (nowNs - startNs) < kInitialStallGraceNs) {
            // 起動直後のギャップは無視してポップノイズを回避
            maybeFlushExternal(std::chrono::nanoseconds::zero());
            return;
        }

        if (AudioInput::shouldResetAfterStall(previousNs, nowNs)) {
            std::chrono::nanoseconds gap(nowNs - previousNs);
            flushInternal(gap);
            return;
        }

        maybeFlushExternal(std::chrono::nanoseconds::zero());
    }

    void flushCaches(std::chrono::nanoseconds gap = std::chrono::nanoseconds::zero()) {
        pendingGapNs_.store(static_cast<std::int64_t>(gap.count()), std::memory_order_release);
        externalFlushRequested_.store(true, std::memory_order_release);
    }

   private:
    void maybeFlushExternal(std::chrono::nanoseconds fallbackGap) {
        if (!externalFlushRequested_.exchange(false, std::memory_order_acq_rel)) {
            return;
        }
        std::int64_t gapNs = pendingGapNs_.exchange(0, std::memory_order_acq_rel);
        std::chrono::nanoseconds gap(gapNs > 0 ? gapNs : fallbackGap.count());
        flushInternal(gap);
    }

    void flushInternal(std::chrono::nanoseconds gap) {
        // リセット後に再度グレース期間を適用するため、入力タイムスタンプを初期化
        lastInputTimestampNs_.store(0, std::memory_order_release);
        firstInputTimestampNs_.store(0, std::memory_order_release);

        if (deps_.flushAction) {
            deps_.flushAction(gap);
        }

        double gapMs = static_cast<double>(gap.count()) / 1'000'000.0;
        LOG_INFO("[Stream] Input gap {:.3f}ms detected → flushing streaming cache", gapMs);
    }

    StreamingCacheDependencies deps_;
    std::atomic<std::int64_t> lastInputTimestampNs_{0};
    std::atomic<std::int64_t> firstInputTimestampNs_{0};
    std::atomic<bool> externalFlushRequested_{false};
    std::atomic<std::int64_t> pendingGapNs_{0};

    static constexpr std::int64_t kInitialStallGraceNs = 1'500'000'000;  // 1.5s
};

}  // namespace streaming_cache

#endif  // STREAMING_CACHE_MANAGER_H
