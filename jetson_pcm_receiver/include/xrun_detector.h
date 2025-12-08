#pragma once

#include <chrono>
#include <optional>

// XRUNの連続時間を監視し、一定時間継続した場合にストームと判定する単純な監視クラス。
class XrunDetector {
   public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    explicit XrunDetector(std::chrono::milliseconds window);

    // XRUN発生を記録し、ストームが検知されたらtrueを返す。
    bool onXrun(TimePoint now);

    // 正常書き込みができたらストーム状態と連続時間をリセットする。
    void onSuccess(TimePoint now);

    // 手動リセット（open直後など）。
    void reset();

    bool storm() const {
        return storm_;
    }

    std::chrono::milliseconds window() const {
        return window_;
    }

   private:
    std::chrono::milliseconds window_;
    std::optional<TimePoint> streakStart_{};
    bool storm_{false};
};
