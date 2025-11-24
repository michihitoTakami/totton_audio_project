/**
 * @file logger.cpp
 * @brief Implementation of structured logging for GPU Audio Upsampler (Issue #43)
 */

#include "logging/logger.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace gpu_upsampler {
namespace logging {

namespace {

// Global logger instance
std::shared_ptr<spdlog::logger> g_logger;
std::mutex g_init_mutex;
std::atomic<bool> g_initialized{false};

// Convert our LogLevel to spdlog::level::level_enum
spdlog::level::level_enum toSpdlogLevel(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
        return spdlog::level::trace;
    case LogLevel::Debug:
        return spdlog::level::debug;
    case LogLevel::Info:
        return spdlog::level::info;
    case LogLevel::Warn:
        return spdlog::level::warn;
    case LogLevel::Error:
        return spdlog::level::err;
    case LogLevel::Critical:
        return spdlog::level::critical;
    case LogLevel::Off:
        return spdlog::level::off;
    default:
        return spdlog::level::info;
    }
}

// Convert spdlog::level to our LogLevel
LogLevel fromSpdlogLevel(spdlog::level::level_enum level) {
    switch (level) {
    case spdlog::level::trace:
        return LogLevel::Trace;
    case spdlog::level::debug:
        return LogLevel::Debug;
    case spdlog::level::info:
        return LogLevel::Info;
    case spdlog::level::warn:
        return LogLevel::Warn;
    case spdlog::level::err:
        return LogLevel::Error;
    case spdlog::level::critical:
        return LogLevel::Critical;
    case spdlog::level::off:
        return LogLevel::Off;
    default:
        return LogLevel::Info;
    }
}

}  // namespace

bool initialize(const LogConfig& config) {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    if (g_initialized.load(std::memory_order_acquire)) {
        // Already initialized, just update settings
        if (g_logger) {
            g_logger->set_level(toSpdlogLevel(config.level));
            g_logger->set_pattern(config.pattern);
        }
        return true;
    }

    try {
        std::vector<spdlog::sink_ptr> sinks;

        // Console sink
        if (config.consoleOutput) {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(toSpdlogLevel(config.level));
            if (!config.coloredOutput) {
                console_sink->set_color_mode(spdlog::color_mode::never);
            }
            sinks.push_back(console_sink);
        }

        // Rotating file sink
        if (!config.filePath.empty()) {
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                config.filePath, config.maxFileSize, config.maxBackups);
            file_sink->set_level(toSpdlogLevel(config.level));
            sinks.push_back(file_sink);
        }

        // Create logger with multiple sinks
        g_logger = std::make_shared<spdlog::logger>("gpu_upsampler", sinks.begin(), sinks.end());
        g_logger->set_level(toSpdlogLevel(config.level));
        g_logger->set_pattern(config.pattern);

        // Enable backtrace for debugging (stores last 32 messages)
        g_logger->enable_backtrace(32);

        // Register as default logger
        spdlog::set_default_logger(g_logger);

        // Flush on error level and above
        g_logger->flush_on(spdlog::level::err);

        g_initialized.store(true, std::memory_order_release);

        LOG_INFO("Logging initialized (level={})", levelToString(config.level));
        if (!config.filePath.empty()) {
            LOG_INFO("Log file: {} (max {}MB x {} backups)", config.filePath,
                     config.maxFileSize / (1024 * 1024), config.maxBackups);
        }

        return true;
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        return false;
    }
}

bool initializeFromConfig(const std::string& configPath) {
    LogConfig config;

    try {
        std::ifstream file(configPath);
        if (!file.is_open()) {
            // Config file doesn't exist, use defaults
            return initialize(config);
        }

        nlohmann::json json;
        file >> json;

        // Read logging section if it exists
        if (json.contains("logging")) {
            const auto& logSection = json["logging"];

            if (logSection.contains("level")) {
                config.level = stringToLevel(logSection["level"].get<std::string>());
            }

            if (logSection.contains("filePath")) {
                config.filePath = logSection["filePath"].get<std::string>();
            }

            if (logSection.contains("maxFileSize")) {
                config.maxFileSize = logSection["maxFileSize"].get<size_t>();
            }

            if (logSection.contains("maxBackups")) {
                config.maxBackups = logSection["maxBackups"].get<size_t>();
            }

            if (logSection.contains("consoleOutput")) {
                config.consoleOutput = logSection["consoleOutput"].get<bool>();
            }

            if (logSection.contains("coloredOutput")) {
                config.coloredOutput = logSection["coloredOutput"].get<bool>();
            }

            if (logSection.contains("pattern")) {
                config.pattern = logSection["pattern"].get<std::string>();
            }
        }
    } catch (const nlohmann::json::exception& ex) {
        std::cerr << "Failed to parse logging config: " << ex.what() << std::endl;
        // Continue with default config
    }

    return initialize(config);
}

void shutdown() {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    if (g_logger) {
        LOG_INFO("Logging shutdown");
        g_logger->flush();
    }

    g_initialized.store(false, std::memory_order_release);
    spdlog::shutdown();
    g_logger.reset();
}

void setLevel(LogLevel level) {
    if (g_logger) {
        g_logger->set_level(toSpdlogLevel(level));
        LOG_INFO("Log level changed to {}", levelToString(level));
    }
}

LogLevel getLevel() {
    if (g_logger) {
        return fromSpdlogLevel(g_logger->level());
    }
    return LogLevel::Info;
}

void flush() {
    if (g_logger) {
        g_logger->flush();
    }
}

std::shared_ptr<spdlog::logger> getLogger() {
    // Fast path: already initialized (no lock)
    if (g_initialized.load(std::memory_order_acquire)) {
        return g_logger;
    }
    // Slow path: need initialization
    initialize();
    return g_logger;
}

std::string_view levelToString(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
        return "trace";
    case LogLevel::Debug:
        return "debug";
    case LogLevel::Info:
        return "info";
    case LogLevel::Warn:
        return "warn";
    case LogLevel::Error:
        return "error";
    case LogLevel::Critical:
        return "critical";
    case LogLevel::Off:
        return "off";
    default:
        return "info";
    }
}

LogLevel stringToLevel(std::string_view str) {
    // Convert to lowercase for case-insensitive comparison
    std::string lower;
    lower.reserve(str.size());
    for (char c : str) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (lower == "trace")
        return LogLevel::Trace;
    if (lower == "debug")
        return LogLevel::Debug;
    if (lower == "info")
        return LogLevel::Info;
    if (lower == "warn" || lower == "warning")
        return LogLevel::Warn;
    if (lower == "error" || lower == "err")
        return LogLevel::Error;
    if (lower == "critical" || lower == "fatal")
        return LogLevel::Critical;
    if (lower == "off" || lower == "none")
        return LogLevel::Off;

    return LogLevel::Info;  // Default
}

}  // namespace logging
}  // namespace gpu_upsampler
