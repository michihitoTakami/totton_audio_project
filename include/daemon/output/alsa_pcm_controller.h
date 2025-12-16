#pragma once

#include "audio/fallback_manager.h"
#include "core/config_loader.h"
#include "daemon/audio_pipeline/streaming_cache_manager.h"
#include "daemon/pcm/dac_manager.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace daemon_output {

struct AlsaPcmControllerDependencies {
    const AppConfig* config = nullptr;
    dac::DacManager* dacManager = nullptr;
    streaming_cache::StreamingCacheManager* streamingCacheManager = nullptr;
    FallbackManager::Manager* fallbackManager = nullptr;
    std::atomic<bool>* running = nullptr;
    std::atomic<bool>* outputReady = nullptr;
    std::function<int()> currentOutputRate;
};

class AlsaPcmController {
   public:
    explicit AlsaPcmController(AlsaPcmControllerDependencies deps);
    ~AlsaPcmController();

    // Open PCM for the currently selected device (blocks/retries while running).
    bool openSelected();
    void close();

    // PCM state / params.
    bool alive() const;
    bool isOpen() const {
        return pcmHandle_ != nullptr;
    }
    const std::string& device() const {
        return currentDevice_;
    }
    std::uint32_t channels() const {
        return channels_;
    }
    std::uint64_t periodFrames() const {
        return periodFrames_;
    }

    // Re-open with a new sample rate (used when output rate changes).
    bool reconfigure(int newSampleRate);

    // Switch to another output device (drop/close/flush + open).
    bool switchDevice(const std::string& nextDevice);

    // Blocking write of interleaved int32 frames.
    long writeInterleaved(const std::int32_t* interleaved, std::size_t frames);

   private:
    bool openForDevice(const std::string& device, int forcedSampleRate);
    void markConnected(const std::string& device, const char* logMessage);
    void markDisconnected(const std::string& device, const char* logMessage);
    void refreshPeriodFrames();

    AlsaPcmControllerDependencies deps_;
    std::string currentDevice_;
    void* pcmHandle_{nullptr};
    std::uint32_t channels_{2};
    std::uint64_t periodFrames_{0};
};

}  // namespace daemon_output
