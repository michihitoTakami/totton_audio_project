#pragma once

#include "audio/filter_headroom.h"
#include "audio/soft_mute.h"
#include "convolution_engine.h"
#include "core/config_loader.h"
#include "core/daemon_constants.h"
#include "daemon/api/dependencies.h"

#include <alsa/asoundlib.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace audio_pipeline {
class AudioPipeline;
class RateSwitcher;
class FilterManager;
class SoftMuteRunner;
}  // namespace audio_pipeline

namespace streaming_cache {
class StreamingCacheManager;
}  // namespace streaming_cache

namespace daemon_output {
class PlaybackBufferManager;
class AlsaOutput;
}  // namespace daemon_output

namespace daemon_control::handlers {
class HandlerRegistry;
}  // namespace daemon_control::handlers

namespace dac {
class DacManager;
}  // namespace dac

namespace FallbackManager {
class Manager;
}  // namespace FallbackManager

namespace daemon_app {

using StreamFloatVector = ConvolutionEngine::StreamFloatVector;

struct GainState {
    std::atomic<float> output{1.0f};
    std::atomic<float> headroom{1.0f};
    std::atomic<float> limiter{1.0f};
    std::atomic<float> effective{1.0f};
};

struct RateState {
    std::atomic<int> currentInputRate{DaemonConstants::DEFAULT_INPUT_SAMPLE_RATE};
    std::atomic<int> currentOutputRate{DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE};
    std::atomic<int> currentRateFamilyInt{
        static_cast<int>(ConvolutionEngine::RateFamily::RATE_44K)};
    std::atomic<int> pendingRateChange{0};
    std::atomic<bool> alsaReconfigureNeeded{false};
    std::atomic<int> alsaNewOutputRate{0};

    ConvolutionEngine::RateFamily activeRateFamily = ConvolutionEngine::RateFamily::RATE_44K;
    PhaseType activePhaseType = PhaseType::Minimum;
    int inputSampleRate = DaemonConstants::DEFAULT_INPUT_SAMPLE_RATE;
};

struct ControlFlags {
    std::atomic<bool> running{true};
    std::atomic<bool> reloadRequested{false};
    std::atomic<bool> zmqBindFailed{false};
    std::atomic<bool> outputReady{false};
};

struct SoftMuteState {
    SoftMute::Controller* controller = nullptr;
    std::mutex opMutex;
    std::atomic<bool> restorePending{false};
    std::atomic<int> restoreFadeMs{0};
    std::atomic<int> restoreSampleRate{0};
};

struct PlaybackState {
    std::unique_ptr<daemon_output::PlaybackBufferManager> buffer;
};

struct StreamingState {
    StreamFloatVector streamInputLeft;
    StreamFloatVector streamInputRight;
    size_t streamAccumulatedLeft = 0;
    size_t streamAccumulatedRight = 0;
    StreamFloatVector upsamplerOutputLeft;
    StreamFloatVector upsamplerOutputRight;

    std::mutex streamingMutex;
};

struct CrossfeedState {
    ConvolutionEngine::FourChannelFIR* processor = nullptr;
    std::atomic<bool> enabled{false};
    ConvolutionEngine::StreamFloatVector cfStreamInputLeft;
    ConvolutionEngine::StreamFloatVector cfStreamInputRight;
    size_t cfStreamAccumulatedLeft = 0;
    size_t cfStreamAccumulatedRight = 0;
    ConvolutionEngine::StreamFloatVector cfOutputLeft;
    ConvolutionEngine::StreamFloatVector cfOutputRight;
    ConvolutionEngine::StreamFloatVector cfOutputBufferLeft;
    ConvolutionEngine::StreamFloatVector cfOutputBufferRight;
    std::mutex crossfeedMutex;
};

struct LoopbackState {
    std::mutex handleMutex;
    snd_pcm_t* handle = nullptr;
    std::atomic<bool> captureRunning{false};
    std::atomic<bool> captureReady{false};
};

struct I2sState {
    std::mutex handleMutex;
    snd_pcm_t* handle = nullptr;
    std::atomic<bool> captureRunning{false};
    std::atomic<bool> captureReady{false};
};

struct ManagerState {
    std::unique_ptr<streaming_cache::StreamingCacheManager> streamingCacheManager;
    std::unique_ptr<daemon_core::api::EventDispatcher> eventDispatcher;
    daemon_core::api::DaemonDependencies daemonDependencies{};

    audio_pipeline::AudioPipeline* audioPipelineRaw = nullptr;
    dac::DacManager* dacManagerRaw = nullptr;

    std::unique_ptr<audio_pipeline::RateSwitcher> rateSwitcher;
    std::unique_ptr<audio_pipeline::FilterManager> filterManager;
    std::unique_ptr<audio_pipeline::SoftMuteRunner> softMuteRunner;
    std::unique_ptr<daemon_output::AlsaOutput> alsaOutputInterface;
    std::unique_ptr<daemon_control::handlers::HandlerRegistry> handlerRegistry;
    std::unique_ptr<dac::DacManager> dacManager;
};

struct RuntimeState {
    AppConfig config;
    GainState gains;
    FilterHeadroomCache headroomCache;

    ControlFlags flags;
    RateState rates;

    ConvolutionEngine::GPUUpsampler* upsampler = nullptr;
    std::mutex inputProcessMutex;
    std::unique_ptr<audio_pipeline::AudioPipeline> audioPipeline;

    SoftMuteState softMute;
    PlaybackState playback;
    StreamingState streaming;
    CrossfeedState crossfeed;
    ManagerState managers;
    LoopbackState loopback;
    I2sState i2s;

    FallbackManager::Manager* fallbackManager = nullptr;
    std::atomic<bool> fallbackActive{false};
    int pidLockFd = -1;
};

}  // namespace daemon_app
