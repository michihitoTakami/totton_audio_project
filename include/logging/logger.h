/**
 * @file logger.h
 * @brief Structured logging API for GPU Audio Upsampler (Issue #43)
 *
 * Provides a unified logging interface using spdlog.
 * Supports console output, rotating file output, and configurable log levels.
 */

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <string_view>

// Forward declare spdlog logger
namespace spdlog {
class logger;
}  // namespace spdlog

namespace gpu_upsampler {
namespace logging {

/**
 * @brief Log level enumeration
 */
enum class LogLevel : std::uint8_t {
    Trace,     // Very detailed debugging information
    Debug,     // Debug information
    Info,      // General information
    Warn,      // Warnings
    Error,     // Errors
    Critical,  // Critical errors
    Off        // Disable logging
};

/**
 * @brief Logging configuration
 */
struct LogConfig {
    LogLevel level = LogLevel::Info;
    std::string filePath = "";                                   // Empty = no file output
    size_t maxFileSize = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    size_t maxBackups = 5;
    bool consoleOutput = true;
    bool coloredOutput = true;
    std::string pattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v";
};

/**
 * @brief Initialize the logging system
 *
 * @param config Logging configuration
 * @return true if initialization succeeded, false otherwise
 */
bool initialize(const LogConfig& config = LogConfig{});

/**
 * @brief Early initialization with stderr output only
 *
 * Lightweight initialization for logging before config file is available.
 * Outputs to stderr only. Should be called before PID lock acquisition.
 * Can be followed by initializeFromConfig() for full initialization.
 *
 * @return true if initialization succeeded, false otherwise
 */
bool initializeEarly();

/**
 * @brief Initialize logging from JSON config file
 *
 * Reads the "logging" section from the config file.
 * Falls back to defaults if section is missing.
 *
 * @param configPath Path to JSON config file
 * @return true if initialization succeeded, false otherwise
 */
bool initializeFromConfig(const std::string& configPath);

/**
 * @brief Shutdown the logging system
 *
 * Flushes all pending log messages and releases resources.
 */
void shutdown();

/**
 * @brief Set the global log level
 *
 * @param level The log level to set
 */
void setLevel(LogLevel level);

/**
 * @brief Get the current log level
 *
 * @return Current log level
 */
LogLevel getLevel();

/**
 * @brief Flush all pending log messages
 */
void flush();

/**
 * @brief Get the underlying spdlog logger
 *
 * For advanced use cases only.
 *
 * @return Shared pointer to spdlog logger
 */
std::shared_ptr<spdlog::logger> getLogger();

/**
 * @brief Convert LogLevel to string
 */
std::string_view levelToString(LogLevel level);

/**
 * @brief Convert string to LogLevel
 *
 * @param str Level name (case-insensitive)
 * @return Corresponding LogLevel, defaults to Info if unknown
 */
LogLevel stringToLevel(std::string_view str);

// ============================================================
// Logging macros - Use these instead of calling spdlog directly
// ============================================================

}  // namespace logging
}  // namespace gpu_upsampler

// Include spdlog for macro usage
#include <spdlog/spdlog.h>

/**
 * @brief Log a trace message
 *
 * Use for very detailed debugging information.
 * Disabled in release builds by default.
 */
#define LOG_TRACE(...)                                     \
    do {                                                   \
        auto logger = gpu_upsampler::logging::getLogger(); \
        if (logger)                                        \
            SPDLOG_LOGGER_TRACE(logger, __VA_ARGS__);      \
    } while (0)

/**
 * @brief Log a debug message
 *
 * Use for debugging information during development.
 */
#define LOG_DEBUG(...)                                     \
    do {                                                   \
        auto logger = gpu_upsampler::logging::getLogger(); \
        if (logger)                                        \
            SPDLOG_LOGGER_DEBUG(logger, __VA_ARGS__);      \
    } while (0)

/**
 * @brief Log an info message
 *
 * Use for general operational information.
 */
#define LOG_INFO(...)                                      \
    do {                                                   \
        auto logger = gpu_upsampler::logging::getLogger(); \
        if (logger)                                        \
            SPDLOG_LOGGER_INFO(logger, __VA_ARGS__);       \
    } while (0)

/**
 * @brief Log a warning message
 *
 * Use for potentially problematic situations.
 */
#define LOG_WARN(...)                                      \
    do {                                                   \
        auto logger = gpu_upsampler::logging::getLogger(); \
        if (logger)                                        \
            SPDLOG_LOGGER_WARN(logger, __VA_ARGS__);       \
    } while (0)

/**
 * @brief Log an error message
 *
 * Use for error conditions that are recoverable.
 */
#define LOG_ERROR(...)                                     \
    do {                                                   \
        auto logger = gpu_upsampler::logging::getLogger(); \
        if (logger)                                        \
            SPDLOG_LOGGER_ERROR(logger, __VA_ARGS__);      \
    } while (0)

/**
 * @brief Log a critical error message
 *
 * Use for critical errors that require immediate attention.
 */
#define LOG_CRITICAL(...)                                  \
    do {                                                   \
        auto logger = gpu_upsampler::logging::getLogger(); \
        if (logger)                                        \
            SPDLOG_LOGGER_CRITICAL(logger, __VA_ARGS__);   \
    } while (0)

// ============================================================
// Conditional logging macros (for performance-critical paths)
// ============================================================

/**
 * @brief Log if condition is true
 *
 * Useful for conditional logging to avoid string formatting overhead.
 */
#define LOG_IF(level, condition, ...) \
    do {                              \
        if (condition)                \
            LOG_##level(__VA_ARGS__); \
    } while (0)

/**
 * @brief Log every N occurrences
 *
 * Useful for rate-limiting logs in hot paths.
 */
#define LOG_EVERY_N(level, n, ...)                            \
    do {                                                      \
        static std::atomic<uint64_t> log_count_##__LINE__{0}; \
        if (log_count_##__LINE__.fetch_add(1) % (n) == 0) {   \
            LOG_##level(__VA_ARGS__);                         \
        }                                                     \
    } while (0)

/**
 * @brief Log at most once
 *
 * Useful for one-time warnings or informational messages.
 */
#define LOG_ONCE(level, ...)                                             \
    do {                                                                 \
        static std::atomic<bool> logged_##__LINE__{false};               \
        bool expected = false;                                           \
        if (logged_##__LINE__.compare_exchange_strong(expected, true)) { \
            LOG_##level(__VA_ARGS__);                                    \
        }                                                                \
    } while (0)
