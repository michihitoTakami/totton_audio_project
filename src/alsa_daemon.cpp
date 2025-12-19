#include "core/config_loader.h"
#include "daemon/api/events.h"
#include "daemon/app/app.h"
#include "daemon/app/process_resources.h"
#include "daemon/app/runtime_state.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/audio_pipeline/filter_manager.h"
#include "daemon/audio_pipeline/headroom_controller.h"
#include "daemon/audio_pipeline/rate_switcher.h"
#include "daemon/audio_pipeline/soft_mute_runner.h"
#include "daemon/audio_pipeline/streaming_cache_manager.h"
#include "daemon/control/handlers/handler_registry.h"
#include "daemon/output/alsa_output.h"
#include "daemon/output/playback_buffer_manager.h"
#include "daemon/pcm/dac_manager.h"
#include "logging/logger.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>

constexpr const char* PID_FILE_PATH = "/tmp/gpu_upsampler_alsa.pid";
constexpr const char* STATS_FILE_PATH = "/tmp/gpu_upsampler_stats.json";
constexpr const char* CONFIG_FILE_PATH = DEFAULT_CONFIG_FILE;

static daemon_app::RuntimeState g_state;

int main(int argc, char* argv[]) {
    gpu_upsampler::logging::initializeEarly();

    daemon_app::ProcessResources::Options resourceOptions;
    resourceOptions.pidFilePath = PID_FILE_PATH;
    resourceOptions.statsFilePath = STATS_FILE_PATH;
    auto processResources = daemon_app::ProcessResources::acquire(resourceOptions);
    if (!processResources) {
        return 1;
    }

    gpu_upsampler::logging::initializeFromConfig(CONFIG_FILE_PATH);

    LOG_INFO("========================================");
    LOG_INFO("  GPU Audio Upsampler - ALSA Direct Output");
    LOG_INFO("========================================");
    LOG_INFO("PID: {} (file: {})", getpid(), processResources->pidLock().path());

    daemon_app::AppOverrides overrides;
    if (const char* env_dev = std::getenv("ALSA_DEVICE")) {
        overrides.alsaDevice = env_dev;
        std::cout << "Config: ALSA_DEVICE env override: " << env_dev << '\n';
    }
    if (argc > 1) {
        overrides.filterPath = argv[1];
        std::cout << "Config: CLI filter path override: " << argv[1] << '\n';
    }

    daemon_app::App app(g_state, CONFIG_FILE_PATH, STATS_FILE_PATH);
    return app.run(overrides);
}
