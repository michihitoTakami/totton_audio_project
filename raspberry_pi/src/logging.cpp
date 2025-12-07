#include "logging.h"

#include <algorithm>
#include <cctype>
#include <iostream>

namespace {
std::atomic<LogLevel> gLogLevel{LogLevel::Info};
}  // namespace

LogLevel parseLogLevel(const std::string &name) {
    std::string lower;
    lower.reserve(name.size());
    std::transform(name.begin(), name.end(), std::back_inserter(lower),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (lower == "error" || lower == "err" || lower == "e") {
        return LogLevel::Error;
    }
    if (lower == "warn" || lower == "warning" || lower == "w") {
        return LogLevel::Warn;
    }
    if (lower == "debug" || lower == "d") {
        return LogLevel::Debug;
    }
    return LogLevel::Info;
}

void setLogLevel(LogLevel level) {
    gLogLevel.store(level, std::memory_order_relaxed);
}

bool shouldLog(LogLevel level) {
    return static_cast<int>(level) <= static_cast<int>(gLogLevel.load(std::memory_order_relaxed));
}

void logMessage(LogLevel level, const std::string &message) {
    if (!shouldLog(level)) {
        return;
    }
    const char *prefix = "";
    switch (level) {
    case LogLevel::Error:
        prefix = "[ERROR] ";
        break;
    case LogLevel::Warn:
        prefix = "[WARN ] ";
        break;
    case LogLevel::Info:
        prefix = "[INFO ] ";
        break;
    case LogLevel::Debug:
        prefix = "[DEBUG] ";
        break;
    }
    std::ostream &os =
        (level == LogLevel::Error || level == LogLevel::Warn) ? std::cerr : std::cout;
    os << prefix << message << std::endl;
}
