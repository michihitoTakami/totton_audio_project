#include "core/config_loader.h"
#include "core/daemon_constants.h"
#include "daemon/app/app.h"
#include "daemon/app/process_resources.h"
#include "logging/logger.h"

#include <unistd.h>

// PID file path (also serves as lock file)
constexpr const char* PID_FILE_PATH = "/tmp/gpu_upsampler_alsa.pid";

// Stats file path (JSON format for Web API)
constexpr const char* STATS_FILE_PATH = "/tmp/gpu_upsampler_stats.json";

constexpr const char* CONFIG_FILE_PATH = DEFAULT_CONFIG_FILE;

int main(int argc, char* argv[]) {
    // Early initialization with stderr output only (before PID lock)
    // This allows logging during PID lock acquisition
    gpu_upsampler::logging::initializeEarly();

    daemon_app::ProcessResources::Options resourceOptions;
    resourceOptions.pidFilePath = PID_FILE_PATH;
    resourceOptions.statsFilePath = STATS_FILE_PATH;
    auto processResources = daemon_app::ProcessResources::acquire(resourceOptions);
    if (!processResources) {
        return 1;
    }

    // Full initialization from config file (after PID lock acquired)
    // This replaces stderr-only logger with configured logger (file + console)
    gpu_upsampler::logging::initializeFromConfig(CONFIG_FILE_PATH);

    LOG_INFO("========================================");
    LOG_INFO("  GPU Audio Upsampler - ALSA Direct Output");
    LOG_INFO("========================================");
    LOG_INFO("PID: {} (file: {})", getpid(), processResources->pidLock().path());

    daemon_app::App app;
    daemon_app::AppOptions options;
    options.configFilePath = CONFIG_FILE_PATH;
    options.statsFilePath = STATS_FILE_PATH;
    return app.run(options, argc, argv);
}
