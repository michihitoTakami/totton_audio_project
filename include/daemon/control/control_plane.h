#pragma once

#include "audio/soft_mute.h"
#include "core/config_loader.h"
#include "daemon/api/events.h"
#include "daemon/control/zmq_server.h"
#include "daemon/metrics/runtime_stats.h"
#include "daemon/pcm/dac_manager.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace ConvolutionEngine {
class GPUUpsampler;
class FourChannelFIR;
}  // namespace ConvolutionEngine

namespace daemon_control {

struct CrossfeedControls {
    ConvolutionEngine::FourChannelFIR** processor = nullptr;
    std::atomic<bool>* enabledFlag = nullptr;
    std::mutex* mutex = nullptr;
    std::function<void()> resetStreamingState;
};

struct ControlPlaneDependencies {
    AppConfig* config = nullptr;
    std::atomic<bool>* runningFlag = nullptr;
    std::atomic<bool>* reloadRequested = nullptr;
    std::atomic<bool>* zmqBindFailed = nullptr;
    std::atomic<int>* currentOutputRate = nullptr;
    SoftMute::Controller** softMute = nullptr;
    PhaseType* activePhaseType = nullptr;
    int* inputSampleRate = nullptr;
    const char* defaultAlsaDevice = nullptr;
    daemon_core::api::EventDispatcher* dispatcher = nullptr;

    std::function<void()> quitMainLoop;
    std::function<runtime_stats::Dependencies()> buildRuntimeStats;
    std::function<size_t()> bufferCapacityFrames;
    std::function<void(std::function<bool()>)> applySoftMuteForFilterSwitch;
    // Clear playback + streaming caches without touching soft mute (caller wraps with soft mute).
    // Used for glitch-free transitions that must not mix old/new audio blocks (Issue #888).
    std::function<bool()> resetStreamingCachesForSwitch;
    std::function<void(const std::string&)> refreshHeadroom;
    std::function<bool()> reinitializeStreamingForLegacyMode;
    std::function<void(AppConfig&, const std::string&)> setPreferredOutputDevice;

    dac::DacManager* dacManager = nullptr;
    ConvolutionEngine::GPUUpsampler** upsampler = nullptr;
    CrossfeedControls crossfeed;

    std::string statsFilePath;
};

class ControlPlane {
   public:
    explicit ControlPlane(ControlPlaneDependencies deps);
    ~ControlPlane();

    ControlPlane(const ControlPlane&) = delete;
    ControlPlane& operator=(const ControlPlane&) = delete;

    bool start();
    void stop();

    std::function<void(const nlohmann::json&)> eventPublisher();

   private:
    void registerHandlers();
    void startStatsThread();
    void stopStatsThread();
    void publish(const nlohmann::json& payload);

    std::string handlePing(const daemon_ipc::ZmqRequest& request);
    std::string handleReload(const daemon_ipc::ZmqRequest& request);
    std::string handleStats(const daemon_ipc::ZmqRequest& request);
    std::string handleCrossfeedEnable(const daemon_ipc::ZmqRequest& request);
    std::string handleCrossfeedDisable(const daemon_ipc::ZmqRequest& request);
    std::string handleCrossfeedStatus(const daemon_ipc::ZmqRequest& request, bool includeHeadSize);
    std::string handleCrossfeedGetStatus(const daemon_ipc::ZmqRequest& request);
    std::string handleCrossfeedSetCombined(const daemon_ipc::ZmqRequest& request);
    std::string handleCrossfeedGenerate(const daemon_ipc::ZmqRequest& request);
    std::string handleCrossfeedSetSize(const daemon_ipc::ZmqRequest& request);
    std::string handleDacList(const daemon_ipc::ZmqRequest& request);
    std::string handleDacStatus(const daemon_ipc::ZmqRequest& request);
    std::string handleDacSelect(const daemon_ipc::ZmqRequest& request);
    std::string handleDacRescan(const daemon_ipc::ZmqRequest& request);
    std::string handleOutputModeGet(const daemon_ipc::ZmqRequest& request);
    std::string handleOutputModeSet(const daemon_ipc::ZmqRequest& request);
    std::string handlePhaseTypeGet(const daemon_ipc::ZmqRequest& request);
    std::string handlePhaseTypeSet(const daemon_ipc::ZmqRequest& request);

    ControlPlaneDependencies deps_;
    std::unique_ptr<daemon_ipc::ZmqCommandServer> zmqServer_;
    std::atomic<bool> statsThreadRunning_{false};
    std::thread statsThread_;
};

}  // namespace daemon_control
