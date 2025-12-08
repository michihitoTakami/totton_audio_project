#include "xrun_detector.h"

#include <chrono>
#include <gtest/gtest.h>

using namespace std::chrono_literals;

TEST(XrunDetector, DetectsStormAfterWindow) {
    XrunDetector detector(1000ms);
    const auto t0 = XrunDetector::Clock::now();

    EXPECT_FALSE(detector.onXrun(t0));
    EXPECT_FALSE(detector.onXrun(t0 + 500ms));
    EXPECT_TRUE(detector.onXrun(t0 + 1001ms));
}

TEST(XrunDetector, SuccessResetsStreak) {
    XrunDetector detector(1000ms);
    const auto t0 = XrunDetector::Clock::now();

    EXPECT_FALSE(detector.onXrun(t0));
    EXPECT_FALSE(detector.onXrun(t0 + 900ms));

    detector.onSuccess(t0 + 950ms);

    EXPECT_FALSE(detector.onXrun(t0 + 1100ms));
    EXPECT_TRUE(detector.onXrun(t0 + 2101ms));
}

TEST(XrunDetector, ResetClearsStorm) {
    XrunDetector detector(1000ms);
    const auto t0 = XrunDetector::Clock::now();

    detector.onXrun(t0);
    detector.onXrun(t0 + 1200ms);
    EXPECT_TRUE(detector.storm());

    detector.reset();
    EXPECT_FALSE(detector.storm());
    EXPECT_FALSE(detector.onXrun(t0 + 1500ms));
}
