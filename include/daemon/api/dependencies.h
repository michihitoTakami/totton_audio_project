#pragma once

#include "config_loader.h"
#include "daemon/api/events.h"
#include "soft_mute.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace audio_pipeline {
class AudioPipeline;
}

namespace daemon_control {
class ControlPlane;
}

namespace daemon_output {
class AlsaOutput;
}

namespace dac {
class DacManager;
}

namespace daemon_core::api {

struct DaemonDependencies {
    AppConfig* config = nullptr;
    std::atomic<bool>* running = nullptr;
    std::atomic<bool>* outputReady = nullptr;
    std::atomic<bool>* crossfeedEnabled = nullptr;
    std::atomic<int>* currentInputRate = nullptr;
    std::atomic<int>* currentOutputRate = nullptr;
    SoftMute::Controller** softMute = nullptr;
    ConvolutionEngine::GPUUpsampler** upsampler = nullptr;
    audio_pipeline::AudioPipeline** audioPipeline = nullptr;
    dac::DacManager** dacManager = nullptr;
    std::mutex* streamingMutex = nullptr;
    std::function<void(const std::string&)> refreshHeadroom;
};

struct DaemonContext {
    EventDispatcher* dispatcher = nullptr;
    DaemonDependencies deps;
};

}  // namespace daemon_core::api
