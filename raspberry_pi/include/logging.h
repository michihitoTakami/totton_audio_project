#pragma once

#include <atomic>
#include <string>

enum class LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
};

LogLevel parseLogLevel(const std::string &name);

void setLogLevel(LogLevel level);

bool shouldLog(LogLevel level);

void logMessage(LogLevel level, const std::string &message);

// Convenience shorthands
inline void logError(const std::string &msg) {
    logMessage(LogLevel::Error, msg);
}
inline void logWarn(const std::string &msg) {
    logMessage(LogLevel::Warn, msg);
}
inline void logInfo(const std::string &msg) {
    logMessage(LogLevel::Info, msg);
}
inline void logDebug(const std::string &msg) {
    logMessage(LogLevel::Debug, msg);
}
