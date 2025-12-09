#include "xrun_detector.h"

XrunDetector::XrunDetector(std::chrono::milliseconds window) : window_(window) {}

bool XrunDetector::onXrun(TimePoint now) {
    if (!streakStart_) {
        streakStart_ = now;
    }
    if (storm_) {
        return true;
    }
    const auto elapsed = now - *streakStart_;
    if (elapsed >= window_) {
        storm_ = true;
    }
    return storm_;
}

void XrunDetector::onSuccess(TimePoint /*now*/) {
    streakStart_.reset();
    storm_ = false;
}

void XrunDetector::reset() {
    streakStart_.reset();
    storm_ = false;
}
