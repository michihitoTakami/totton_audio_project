#pragma once

#include "config_loader.h"
#include "rtp_session_manager.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace rtp_engine {

class RtpEngineCoordinator {
   public:
    struct Dependencies {
        AppConfig* config = nullptr;
        std::atomic<bool>* runningFlag = nullptr;
        std::atomic<int>* currentInputRate = nullptr;
        std::atomic<int>* currentOutputRate = nullptr;
        std::atomic<bool>* alsaReconfigureNeeded = nullptr;
        std::function<bool(int)> handleRateChange;
        std::function<bool()> isUpsamplerReady;
        std::function<bool()> isMultiRateEnabled;
        std::function<int()> getUpsampleRatio;
        std::function<int()> getInputSampleRate;
        std::function<void(const float*, size_t, uint32_t)> processInterleaved;
        std::function<Network::PtpSyncState()> ptpProvider;
        std::function<void(const Network::SessionMetrics&)> telemetry;
    };

    explicit RtpEngineCoordinator(Dependencies deps);

    void startFromConfig();
    bool handleZeroMqCommand(const std::string& cmdType, const nlohmann::json& message,
                             std::string& responseOut);
    void shutdown();

   private:
    Network::SessionConfig buildSessionConfig(const AppConfig::RtpInputConfig& cfg) const;
    void maybeSwitchRateForRtp(uint32_t sessionRate, const std::string& sessionId);
    void ensureManagerInitialized();
    nlohmann::json runDiscoveryScan();
    nlohmann::json getOrRunDiscovery();
    uint32_t clampDiscoveryDuration(uint32_t value) const;
    uint32_t clampDiscoveryCooldown(uint32_t value) const;
    size_t clampDiscoveryStreamLimit(size_t value) const;
    std::vector<uint16_t> buildDiscoveryPorts(const AppConfig::RtpInputConfig& cfg) const;

    Dependencies deps_;
    std::unique_ptr<Network::RtpSessionManager> manager_;

    std::mutex discoveryMutex_;
    nlohmann::json discoveryCache_;
    std::chrono::steady_clock::time_point lastDiscovery_{
        std::chrono::steady_clock::time_point::min()};
};

}  // namespace rtp_engine
