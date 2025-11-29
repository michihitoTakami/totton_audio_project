#ifndef RTP_SESSION_MANAGER_H
#define RTP_SESSION_MANAGER_H

#include "error_codes.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace Network {

struct PtpSyncState {
    bool locked = false;
    double offsetNs = 0.0;
    double meanPathDelayNs = 0.0;
    uint64_t lastSyncTimestampNs = 0;
};

struct SessionConfig {
    std::string sessionId = "default";
    std::string bindAddress = "0.0.0.0";
    uint16_t port = 6000;
    std::string sourceHost;  // Optional sender IP for filtering (empty = accept all)
    bool autoStart = false;
    bool multicast = false;
    std::string multicastGroup;
    std::string interfaceName;
    uint8_t ttl = 32;
    int dscp = -1;
    uint32_t sampleRate = 48000;
    uint8_t channels = 2;
    uint8_t bitsPerSample = 24;
    bool bigEndian = true;
    bool signedSamples = true;
    uint8_t payloadType = 97;
    size_t socketBufferBytes = 1 << 20;  // 1MB
    size_t mtuBytes = 1500;
    uint32_t targetLatencyMs = 5;
    uint32_t watchdogTimeoutMs = 500;
    uint32_t telemetryIntervalMs = 1000;
    bool enableRtcp = true;
    uint16_t rtcpPort = 0;  // 0 => port + 1
    bool enablePtp = false;
    std::string ptpInterface;
    int ptpDomain = 0;
    std::string sdp;
};

struct SessionMetrics {
    std::string sessionId;
    std::string bindAddress;
    uint16_t port = 0;
    std::string sourceHost;
    bool multicast = false;
    std::string multicastGroup;
    std::string interfaceName;
    uint8_t payloadType = 0;
    uint8_t channels = 0;
    uint8_t bitsPerSample = 0;
    bool bigEndian = true;
    bool signedSamples = true;
    bool enableRtcp = false;
    uint16_t rtcpPort = 0;
    bool enablePtp = false;
    uint32_t targetLatencyMs = 0;
    uint32_t watchdogTimeoutMs = 0;
    uint32_t telemetryIntervalMs = 0;
    bool autoStart = false;
    uint32_t ssrc = 0;
    bool ssrcLocked = false;
    uint16_t lastSequence = 0;
    bool hasSequence = false;
    uint64_t packetsReceived = 0;
    uint64_t packetsDropped = 0;
    uint64_t sequenceResets = 0;
    uint64_t bytesReceived = 0;
    uint64_t rtcpPackets = 0;
    uint64_t latePackets = 0;
    double avgTransitUsec = 0.0;
    double networkJitterUsec = 0.0;
    double ptpOffsetNs = 0.0;
    double ptpMeanPathNs = 0.0;
    bool ptpLocked = false;
    uint32_t sampleRate = 0;
    uint32_t lastRtpTimestamp = 0;
    std::chrono::system_clock::time_point lastPacketWallclock{};
    std::chrono::steady_clock::time_point lastPacketMonotonic{};
};

nlohmann::json sessionConfigToJson(const SessionConfig& config);
bool sessionConfigFromJson(const nlohmann::json& input, SessionConfig& config, std::string& error);
nlohmann::json sessionMetricsToJson(const SessionMetrics& metrics);
bool validateSessionConfig(SessionConfig& config, std::string& error);

class RtpSessionManager {
   public:
    using FrameCallback =
        std::function<void(const float* interleaved, size_t frames, uint32_t sampleRate)>;
    using TelemetryCallback = std::function<void(const SessionMetrics&)>;
    using PtpStateProvider = std::function<PtpSyncState()>;

    explicit RtpSessionManager(FrameCallback frameCallback, PtpStateProvider ptpProvider = {},
                               TelemetryCallback telemetryCallback = {});
    ~RtpSessionManager();

    bool startSession(const SessionConfig& config, std::string& errorMessage);
    bool stopSession(const std::string& sessionId, std::string& errorMessage);
    void stopAll();
    std::vector<SessionMetrics> listSessions() const;
    std::optional<SessionMetrics> getMetrics(const std::string& sessionId) const;
    bool hasSession(const std::string& sessionId) const;

   private:
    struct Session;

    static bool validateConfig(const SessionConfig& config, std::string& errorMessage);
    void receiverLoop(std::shared_ptr<Session> session);
    void rtcpLoop(std::shared_ptr<Session> session);
    void updatePtpState(Session& session);
    void recordWatchdog(Session& session);
    void emitTelemetry(Session& session);

    FrameCallback frameCallback_;
    PtpStateProvider ptpProvider_;
    TelemetryCallback telemetryCallback_;

    mutable std::mutex sessionsMutex_;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
};

}  // namespace Network

#endif  // RTP_SESSION_MANAGER_H
