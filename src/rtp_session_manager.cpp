#include "rtp_session_manager.h"

#include "logging/logger.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sstream>
#include <string_view>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace Network {
namespace {

constexpr int kDefaultPollTimeoutMs = 50;

struct SdpFormatHint {
    uint8_t payloadType = 0;
    uint32_t sampleRate = 0;
    uint8_t channels = 0;
    uint8_t bitsPerSample = 0;
};

std::string errnoMessage(const std::string& prefix) {
    return prefix + ": " + std::strerror(errno);
}

bool parseIpv4(const std::string& text, in_addr& out) {
    if (text.empty()) {
        out.s_addr = INADDR_ANY;
        return true;
    }
    return ::inet_pton(AF_INET, text.c_str(), &out) == 1;
}

int createAndBindSocket(const std::string& bindAddress, uint16_t port, size_t bufferBytes,
                        std::string& error) {
    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        error = errnoMessage("socket");
        return -1;
    }

    int reuse = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    if (bufferBytes > 0) {
        int buf = static_cast<int>(bufferBytes);
        ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (!parseIpv4(bindAddress, addr.sin_addr)) {
        error = "Invalid bind address: " + bindAddress;
        ::close(fd);
        return -1;
    }

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        error = errnoMessage("bind");
        ::close(fd);
        return -1;
    }

    int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags >= 0) {
        ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

    return fd;
}

std::string trim(const std::string& text) {
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
        start++;
    }
    size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        end--;
    }
    return text.substr(start, end - start);
}

bool parsePositiveInt(const std::string& text, int& out) {
    try {
        size_t idx = 0;
        int value = std::stoi(text, &idx);
        if (idx != text.size() || value <= 0) {
            return false;
        }
        out = value;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseRtpmapLine(const std::string& line, SdpFormatHint& out) {
    // Example: a=rtpmap:96 L24/48000/2
    static constexpr std::string_view prefix = "a=rtpmap:";
    if (!line.starts_with(prefix)) {
        return false;
    }
    std::string rest = trim(line.substr(prefix.size()));
    size_t spacePos = rest.find_first_of(" \t");
    if (spacePos == std::string::npos) {
        return false;
    }
    std::string payloadText = trim(rest.substr(0, spacePos));
    std::string encoding = trim(rest.substr(spacePos + 1));

    int payload = 0;
    if (!parsePositiveInt(payloadText, payload) || payload > 127) {
        return false;
    }

    std::vector<std::string> tokens;
    std::stringstream ss(encoding);
    std::string token;
    while (std::getline(ss, token, '/')) {
        tokens.push_back(token);
    }
    if (tokens.size() < 2) {
        return false;
    }

    std::string encodingName = tokens[0];
    for (auto& ch : encodingName) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    if (encodingName.size() < 2 || encodingName[0] != 'L') {
        return false;
    }

    int bits = 0;
    if (!parsePositiveInt(encodingName.substr(1), bits)) {
        return false;
    }
    if (bits != 16 && bits != 24 && bits != 32) {
        return false;
    }

    int sampleRate = 0;
    if (!parsePositiveInt(tokens[1], sampleRate)) {
        return false;
    }

    int channels = 0;
    if (tokens.size() >= 3) {
        parsePositiveInt(tokens[2], channels);
    }

    out.payloadType = static_cast<uint8_t>(payload);
    out.bitsPerSample = static_cast<uint8_t>(bits);
    out.sampleRate = static_cast<uint32_t>(sampleRate);
    out.channels = static_cast<uint8_t>(channels);
    return true;
}

bool applySdpOverrides(SessionConfig& config) {
    if (config.sdp.empty()) {
        return false;
    }

    SdpFormatHint chosen{};
    bool hasCandidate = false;
    bool matchedPayload = false;

    std::stringstream ss(config.sdp);
    std::string line;
    while (std::getline(ss, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        SdpFormatHint fmt;
        if (!parseRtpmapLine(line, fmt)) {
            continue;
        }

        bool isPreferred = fmt.payloadType == config.payloadType;
        if (!hasCandidate || (isPreferred && !matchedPayload)) {
            chosen = fmt;
            hasCandidate = true;
            matchedPayload = isPreferred;
            if (matchedPayload) {
                break;  // Prefer exact payload type match
            }
        }
    }

    if (!hasCandidate) {
        LOG_WARN("RTP session {}: SDP provided but no parsable rtpmap line found",
                 config.sessionId);
        return false;
    }

    config.payloadType = chosen.payloadType;
    config.sampleRate = chosen.sampleRate > 0 ? chosen.sampleRate : config.sampleRate;
    if (chosen.channels > 0) {
        config.channels = chosen.channels;
    }
    if (chosen.bitsPerSample > 0) {
        config.bitsPerSample = chosen.bitsPerSample;
    }

    LOG_INFO("RTP session {}: applied SDP rtpmap PT{} => {} Hz, {}ch, {}-bit", config.sessionId,
             chosen.payloadType, config.sampleRate, config.channels, config.bitsPerSample);
    return true;
}

void configureQos(const SessionConfig& config, int fd) {
    if (config.dscp >= 0 && config.dscp <= 63) {
        int tos = (config.dscp & 0x3F) << 2;
        ::setsockopt(fd, IPPROTO_IP, IP_TOS, &tos, sizeof(tos));
    }

    if (config.ttl > 0) {
        int ttl = config.ttl;
        ::setsockopt(fd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));
    }
}

void configureMulticast(const SessionConfig& config, int fd) {
    if (!config.multicast || config.multicastGroup.empty()) {
        return;
    }

    ip_mreq mreq{};
    if (::inet_pton(AF_INET, config.multicastGroup.c_str(), &mreq.imr_multiaddr) != 1) {
        LOG_WARN("RTP session {}: invalid multicast group {}", config.sessionId,
                 config.multicastGroup);
        return;
    }

    if (!config.interfaceName.empty()) {
        in_addr ifaceAddr{};
        if (::inet_pton(AF_INET, config.interfaceName.c_str(), &ifaceAddr) == 1) {
            mreq.imr_interface = ifaceAddr;
        } else {
            unsigned int index = ::if_nametoindex(config.interfaceName.c_str());
            if (index > 0) {
#ifdef IP_MULTICAST_IF
                ip_mreqn reqn{};
                reqn.imr_multiaddr = mreq.imr_multiaddr;
                reqn.imr_address.s_addr = INADDR_ANY;
                reqn.imr_ifindex = static_cast<int>(index);
                if (::setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &reqn, sizeof(reqn)) == 0) {
                    LOG_INFO("RTP session {} joined multicast {} on ifindex {}", config.sessionId,
                             config.multicastGroup, index);
                    return;
                }
#endif
            }
        }
    }

    if (::setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0) {
        LOG_WARN("RTP session {}: failed to join multicast {} ({})", config.sessionId,
                 config.multicastGroup, std::strerror(errno));
    }
}

float decodePcmSample(const uint8_t* data, uint8_t bitsPerSample, bool bigEndian,
                      bool signedSample) {
    switch (bitsPerSample) {
    case 16: {
        int16_t value;
        if (bigEndian) {
            value = static_cast<int16_t>((data[0] << 8) | data[1]);
        } else {
            value = static_cast<int16_t>((data[1] << 8) | data[0]);
        }
        if (!signedSample) {
            value = static_cast<int16_t>(value - 32768);
        }
        return static_cast<float>(value) / 32768.0f;
    }
    case 24: {
        int32_t value = 0;
        if (bigEndian) {
            value = (data[0] << 16) | (data[1] << 8) | data[2];
        } else {
            value = (data[2] << 16) | (data[1] << 8) | data[0];
        }
        if (value & 0x800000) {
            value |= ~0xFFFFFF;
        }
        if (!signedSample) {
            value -= 0x800000;
        }
        return static_cast<float>(value) / 8388608.0f;
    }
    case 32: {
        int32_t value;
        if (bigEndian) {
            value = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
        } else {
            value = (data[3] << 24) | (data[2] << 16) | (data[1] << 8) | data[0];
        }
        if (!signedSample) {
            value = static_cast<int32_t>(value - 0x80000000);
        }
        return static_cast<float>(value) / 2147483648.0f;
    }
    default:
        return 0.0f;
    }
}

struct RtpHeader {
    uint8_t version = 0;
    bool padding = false;
    bool extension = false;
    uint8_t csrcCount = 0;
    bool marker = false;
    uint8_t payloadType = 0;
    uint16_t sequenceNumber = 0;
    uint32_t timestamp = 0;
    uint32_t ssrc = 0;
    size_t headerBytes = 0;
};

bool parseRtpHeader(const uint8_t* data, size_t length, RtpHeader& header) {
    if (length < 12) {
        return false;
    }
    header.version = (data[0] & 0b11000000) >> 6;
    header.padding = (data[0] & 0b00100000) != 0;
    header.extension = (data[0] & 0b00010000) != 0;
    header.csrcCount = data[0] & 0b00001111;
    header.marker = (data[1] & 0b10000000) != 0;
    header.payloadType = data[1] & 0b01111111;
    header.sequenceNumber = static_cast<uint16_t>((data[2] << 8) | data[3]);
    header.timestamp = (static_cast<uint32_t>(data[4]) << 24) |
                       (static_cast<uint32_t>(data[5]) << 16) |
                       (static_cast<uint32_t>(data[6]) << 8) | static_cast<uint32_t>(data[7]);
    header.ssrc = (static_cast<uint32_t>(data[8]) << 24) | (static_cast<uint32_t>(data[9]) << 16) |
                  (static_cast<uint32_t>(data[10]) << 8) | static_cast<uint32_t>(data[11]);
    header.headerBytes = 12 + header.csrcCount * 4;
    if (header.version != 2 || header.headerBytes > length) {
        return false;
    }
    if (header.extension) {
        if (length < header.headerBytes + 4) {
            return false;
        }
        uint16_t extensionLength =
            static_cast<uint16_t>((data[header.headerBytes + 2] << 8) |
                                  data[header.headerBytes + 3]);  // length in 32-bit words
        header.headerBytes += 4 + (extensionLength * 4);
        if (header.headerBytes > length) {
            return false;
        }
    }
    return true;
}

}  // namespace

struct RtpSessionManager::Session {
    SessionConfig config;
    SessionMetrics metrics;
    mutable std::mutex metricsMutex;
    std::atomic<bool> running{false};
    int rtpSocket = -1;
    int rtcpSocket = -1;
    std::thread rtpThread;
    std::thread rtcpThread;
    std::chrono::steady_clock::time_point lastTelemetry;
    std::chrono::steady_clock::time_point lastWatchdog;
};

nlohmann::json sessionConfigToJson(const SessionConfig& config) {
    nlohmann::json j;
    j["session_id"] = config.sessionId;
    j["bind_address"] = config.bindAddress;
    j["port"] = config.port;
    j["source_host"] = config.sourceHost;
    j["multicast"] = config.multicast;
    j["multicast_group"] = config.multicastGroup;
    j["interface"] = config.interfaceName;
    j["ttl"] = config.ttl;
    j["dscp"] = config.dscp;
    j["sample_rate"] = config.sampleRate;
    j["channels"] = config.channels;
    j["bits_per_sample"] = config.bitsPerSample;
    j["big_endian"] = config.bigEndian;
    j["signed"] = config.signedSamples;
    j["payload_type"] = config.payloadType;
    j["socket_buffer_bytes"] = config.socketBufferBytes;
    j["mtu_bytes"] = config.mtuBytes;
    j["target_latency_ms"] = config.targetLatencyMs;
    j["watchdog_timeout_ms"] = config.watchdogTimeoutMs;
    j["telemetry_interval_ms"] = config.telemetryIntervalMs;
    j["enable_rtcp"] = config.enableRtcp;
    j["rtcp_port"] = config.rtcpPort;
    j["enable_ptp"] = config.enablePtp;
    j["ptp_interface"] = config.ptpInterface;
    j["ptp_domain"] = config.ptpDomain;
    j["sdp"] = config.sdp;
    j["auto_start"] = config.autoStart;
    return j;
}

bool validateSessionConfig(SessionConfig& config, std::string& error) {
    if (config.sessionId.empty()) {
        error = "session_id must not be empty";
        return false;
    }
    applySdpOverrides(config);
    if (config.sampleRate == 0) {
        error = "sample_rate must be greater than 0";
        return false;
    }
    if (config.channels == 0 || config.channels > 8) {
        error = "channels must be between 1 and 8";
        return false;
    }
    if (config.bitsPerSample != 16 && config.bitsPerSample != 24 && config.bitsPerSample != 32) {
        error = "bits_per_sample must be 16, 24, or 32";
        return false;
    }
    config.socketBufferBytes = std::max<size_t>(65536, config.socketBufferBytes);
    config.mtuBytes = std::max<size_t>(256, config.mtuBytes);
    config.targetLatencyMs = std::max<uint32_t>(1, config.targetLatencyMs);
    config.watchdogTimeoutMs = std::max<uint32_t>(100, config.watchdogTimeoutMs);
    config.telemetryIntervalMs = std::max<uint32_t>(100, config.telemetryIntervalMs);
    if (config.rtcpPort == 0) {
        config.rtcpPort = static_cast<uint16_t>(config.port + 1);
    }
    if (!config.sourceHost.empty()) {
        in_addr addr{};
        if (::inet_pton(AF_INET, config.sourceHost.c_str(), &addr) != 1) {
            error = "source_host must be IPv4 literal";
            return false;
        }
    }
    return true;
}

bool sessionConfigFromJson(const nlohmann::json& input, SessionConfig& config, std::string& error) {
    SessionConfig parsed;
    try {
        if (input.contains("session_id")) {
            parsed.sessionId = input.at("session_id").get<std::string>();
        }
        if (input.contains("bind_address")) {
            parsed.bindAddress = input.at("bind_address").get<std::string>();
        }
        if (input.contains("port")) {
            parsed.port = static_cast<uint16_t>(input.at("port").get<int>());
        }
        if (input.contains("source_host")) {
            parsed.sourceHost = input.at("source_host").get<std::string>();
        }
        if (input.contains("multicast")) {
            parsed.multicast = input.at("multicast").get<bool>();
        }
        if (input.contains("auto_start")) {
            parsed.autoStart = input.at("auto_start").get<bool>();
        }
        if (input.contains("multicast_group")) {
            parsed.multicastGroup = input.at("multicast_group").get<std::string>();
        }
        if (input.contains("interface")) {
            parsed.interfaceName = input.at("interface").get<std::string>();
        }
        if (input.contains("ttl")) {
            parsed.ttl = static_cast<uint8_t>(input.at("ttl").get<int>());
        }
        if (input.contains("dscp")) {
            parsed.dscp = input.at("dscp").get<int>();
        }
        if (input.contains("sample_rate")) {
            parsed.sampleRate = input.at("sample_rate").get<uint32_t>();
        }
        if (input.contains("channels")) {
            parsed.channels = static_cast<uint8_t>(input.at("channels").get<int>());
        }
        if (input.contains("bits_per_sample")) {
            parsed.bitsPerSample = static_cast<uint8_t>(input.at("bits_per_sample").get<int>());
        }
        if (input.contains("big_endian")) {
            parsed.bigEndian = input.at("big_endian").get<bool>();
        }
        if (input.contains("signed")) {
            parsed.signedSamples = input.at("signed").get<bool>();
        }
        if (input.contains("payload_type")) {
            parsed.payloadType = static_cast<uint8_t>(input.at("payload_type").get<int>());
        }
        if (input.contains("socket_buffer_bytes")) {
            parsed.socketBufferBytes = input.at("socket_buffer_bytes").get<size_t>();
        }
        if (input.contains("mtu_bytes")) {
            parsed.mtuBytes = input.at("mtu_bytes").get<size_t>();
        }
        if (input.contains("target_latency_ms")) {
            parsed.targetLatencyMs = input.at("target_latency_ms").get<uint32_t>();
        }
        if (input.contains("watchdog_timeout_ms")) {
            parsed.watchdogTimeoutMs = input.at("watchdog_timeout_ms").get<uint32_t>();
        }
        if (input.contains("telemetry_interval_ms")) {
            parsed.telemetryIntervalMs = input.at("telemetry_interval_ms").get<uint32_t>();
        }
        if (input.contains("enable_rtcp")) {
            parsed.enableRtcp = input.at("enable_rtcp").get<bool>();
        }
        if (input.contains("rtcp_port")) {
            parsed.rtcpPort = static_cast<uint16_t>(input.at("rtcp_port").get<int>());
        }
        if (input.contains("enable_ptp")) {
            parsed.enablePtp = input.at("enable_ptp").get<bool>();
        }
        if (input.contains("ptp_interface")) {
            parsed.ptpInterface = input.at("ptp_interface").get<std::string>();
        }
        if (input.contains("ptp_domain")) {
            parsed.ptpDomain = input.at("ptp_domain").get<int>();
        }
        if (input.contains("sdp")) {
            parsed.sdp = input.at("sdp").get<std::string>();
        }
    } catch (const std::exception& ex) {
        error = std::string("Invalid RTP session parameters: ") + ex.what();
        return false;
    }

    if (!validateSessionConfig(parsed, error)) {
        return false;
    }
    config = parsed;
    return true;
}

nlohmann::json sessionMetricsToJson(const SessionMetrics& metrics) {
    nlohmann::json j;
    j["session_id"] = metrics.sessionId;
    j["bind_address"] = metrics.bindAddress;
    j["port"] = metrics.port;
    if (!metrics.sourceHost.empty()) {
        j["source_host"] = metrics.sourceHost;
    }
    j["multicast"] = metrics.multicast;
    if (!metrics.multicastGroup.empty()) {
        j["multicast_group"] = metrics.multicastGroup;
    }
    if (!metrics.interfaceName.empty()) {
        j["interface"] = metrics.interfaceName;
    }
    j["payload_type"] = metrics.payloadType;
    j["channels"] = metrics.channels;
    j["bits_per_sample"] = metrics.bitsPerSample;
    j["big_endian"] = metrics.bigEndian;
    j["signed"] = metrics.signedSamples;
    j["enable_rtcp"] = metrics.enableRtcp;
    j["rtcp_port"] = metrics.rtcpPort;
    j["enable_ptp"] = metrics.enablePtp;
    j["target_latency_ms"] = metrics.targetLatencyMs;
    j["watchdog_timeout_ms"] = metrics.watchdogTimeoutMs;
    j["telemetry_interval_ms"] = metrics.telemetryIntervalMs;
    j["auto_start"] = metrics.autoStart;
    j["ssrc"] = metrics.ssrc;
    j["ssrc_locked"] = metrics.ssrcLocked;
    j["packets_received"] = metrics.packetsReceived;
    j["packets_dropped"] = metrics.packetsDropped;
    j["sequence_resets"] = metrics.sequenceResets;
    j["bytes_received"] = metrics.bytesReceived;
    j["rtcp_packets"] = metrics.rtcpPackets;
    j["late_packets"] = metrics.latePackets;
    j["avg_transit_usec"] = metrics.avgTransitUsec;
    j["network_jitter_usec"] = metrics.networkJitterUsec;
    j["ptp_offset_ns"] = metrics.ptpOffsetNs;
    j["ptp_mean_path_ns"] = metrics.ptpMeanPathNs;
    j["ptp_locked"] = metrics.ptpLocked;
    j["sample_rate"] = metrics.sampleRate;
    j["last_rtp_timestamp"] = metrics.lastRtpTimestamp;
    if (metrics.lastPacketWallclock.time_since_epoch().count() > 0) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            metrics.lastPacketWallclock.time_since_epoch());
        j["last_packet_unix_ms"] = ms.count();
    }
    return j;
}

RtpSessionManager::RtpSessionManager(FrameCallback frameCallback, PtpStateProvider ptpProvider,
                                     TelemetryCallback telemetryCallback)
    : frameCallback_(std::move(frameCallback)),
      ptpProvider_(std::move(ptpProvider)),
      telemetryCallback_(std::move(telemetryCallback)) {}

RtpSessionManager::~RtpSessionManager() {
    stopAll();
}

bool RtpSessionManager::validateConfig(const SessionConfig& config, std::string& errorMessage) {
    SessionConfig copy = config;
    return validateSessionConfig(copy, errorMessage);
}

bool RtpSessionManager::startSession(const SessionConfig& config, std::string& errorMessage) {
    SessionConfig normalized = config;
    if (!validateSessionConfig(normalized, errorMessage)) {
        return false;
    }

    auto session = std::make_shared<Session>();
    session->config = normalized;
    session->metrics.sessionId = normalized.sessionId;
    session->metrics.bindAddress = normalized.bindAddress;
    session->metrics.port = normalized.port;
    session->metrics.sourceHost = normalized.sourceHost;
    session->metrics.multicast = normalized.multicast;
    session->metrics.multicastGroup = normalized.multicastGroup;
    session->metrics.interfaceName = normalized.interfaceName;
    session->metrics.payloadType = normalized.payloadType;
    session->metrics.channels = normalized.channels;
    session->metrics.bitsPerSample = normalized.bitsPerSample;
    session->metrics.bigEndian = normalized.bigEndian;
    session->metrics.signedSamples = normalized.signedSamples;
    session->metrics.enableRtcp = normalized.enableRtcp;
    session->metrics.rtcpPort = normalized.rtcpPort;
    session->metrics.enablePtp = normalized.enablePtp;
    session->metrics.targetLatencyMs = normalized.targetLatencyMs;
    session->metrics.watchdogTimeoutMs = normalized.watchdogTimeoutMs;
    session->metrics.telemetryIntervalMs = normalized.telemetryIntervalMs;
    session->metrics.autoStart = normalized.autoStart;
    session->metrics.sampleRate = normalized.sampleRate;
    session->lastTelemetry = std::chrono::steady_clock::now();
    session->lastWatchdog = session->lastTelemetry;

    session->rtpSocket = createAndBindSocket(normalized.bindAddress, normalized.port,
                                             normalized.socketBufferBytes, errorMessage);
    if (session->rtpSocket < 0) {
        return false;
    }
    configureQos(normalized, session->rtpSocket);
    configureMulticast(normalized, session->rtpSocket);

    if (normalized.enableRtcp) {
        session->rtcpSocket = createAndBindSocket(normalized.bindAddress, normalized.rtcpPort,
                                                  normalized.socketBufferBytes / 2, errorMessage);
        if (session->rtcpSocket < 0) {
            ::close(session->rtpSocket);
            session->rtpSocket = -1;
            return false;
        }
    }

    session->running.store(true);
    session->rtpThread = std::thread(&RtpSessionManager::receiverLoop, this, session);
    if (normalized.enableRtcp) {
        session->rtcpThread = std::thread(&RtpSessionManager::rtcpLoop, this, session);
    }

    {
        std::lock_guard<std::mutex> lock(sessionsMutex_);
        if (sessions_.count(normalized.sessionId) != 0) {
            errorMessage = "Session '" + normalized.sessionId + "' already exists";
            session->running.store(false);
            if (session->rtpThread.joinable()) {
                session->rtpThread.join();
            }
            if (session->rtcpThread.joinable()) {
                session->rtcpThread.join();
            }
            if (session->rtpSocket >= 0) {
                ::close(session->rtpSocket);
            }
            if (session->rtcpSocket >= 0) {
                ::close(session->rtcpSocket);
            }
            return false;
        }
        sessions_[normalized.sessionId] = session;
    }

    LOG_INFO("RTP session {} listening on {}:{} (payload PT {}, {} ch @ {} Hz)",
             normalized.sessionId, normalized.bindAddress, normalized.port, normalized.payloadType,
             normalized.channels, normalized.sampleRate);
    return true;
}

bool RtpSessionManager::stopSession(const std::string& sessionId, std::string& errorMessage) {
    std::shared_ptr<Session> session;
    {
        std::lock_guard<std::mutex> lock(sessionsMutex_);
        auto it = sessions_.find(sessionId);
        if (it == sessions_.end()) {
            errorMessage = "Session '" + sessionId + "' not found";
            return false;
        }
        session = it->second;
        sessions_.erase(it);
    }

    session->running.store(false);
    if (session->rtpSocket >= 0) {
        ::shutdown(session->rtpSocket, SHUT_RDWR);
    }
    if (session->rtpThread.joinable()) {
        session->rtpThread.join();
    }
    if (session->rtcpSocket >= 0) {
        ::shutdown(session->rtcpSocket, SHUT_RDWR);
    }
    if (session->rtcpThread.joinable()) {
        session->rtcpThread.join();
    }
    if (session->rtpSocket >= 0) {
        ::close(session->rtpSocket);
    }
    if (session->rtcpSocket >= 0) {
        ::close(session->rtcpSocket);
    }
    LOG_INFO("RTP session {} stopped", sessionId);
    return true;
}

void RtpSessionManager::stopAll() {
    std::vector<std::string> ids;
    {
        std::lock_guard<std::mutex> lock(sessionsMutex_);
        for (const auto& entry : sessions_) {
            ids.emplace_back(entry.first);
        }
    }
    for (const auto& id : ids) {
        std::string ignored;
        stopSession(id, ignored);
    }
}

std::vector<SessionMetrics> RtpSessionManager::listSessions() const {
    std::vector<SessionMetrics> metrics;
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    metrics.reserve(sessions_.size());
    for (const auto& entry : sessions_) {
        std::lock_guard<std::mutex> metricsLock(entry.second->metricsMutex);
        metrics.push_back(entry.second->metrics);
    }
    return metrics;
}

std::optional<SessionMetrics> RtpSessionManager::getMetrics(const std::string& sessionId) const {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    auto it = sessions_.find(sessionId);
    if (it == sessions_.end()) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> metricsLock(it->second->metricsMutex);
    return it->second->metrics;
}

bool RtpSessionManager::hasSession(const std::string& sessionId) const {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    return sessions_.count(sessionId) != 0;
}

void RtpSessionManager::receiverLoop(std::shared_ptr<Session> session) {
    std::vector<uint8_t> packet(session->config.mtuBytes);
    while (session->running.load()) {
        pollfd pfd{session->rtpSocket, POLLIN, 0};
        int pollResult = ::poll(&pfd, 1, kDefaultPollTimeoutMs);
        if (!session->running.load()) {
            break;
        }
        if (pollResult < 0) {
            if (errno == EINTR) {
                continue;
            }
            LOG_WARN("RTP session {} poll error: {}", session->config.sessionId,
                     std::strerror(errno));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (pollResult == 0) {
            recordWatchdog(*session);
            emitTelemetry(*session);
            continue;
        }

        sockaddr_in remote{};
        socklen_t remoteLen = sizeof(remote);
        ssize_t bytes = ::recvfrom(session->rtpSocket, packet.data(), packet.size(), 0,
                                   reinterpret_cast<sockaddr*>(&remote), &remoteLen);
        if (bytes <= 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            LOG_DEBUG("RTP session {} recv error: {}", session->config.sessionId,
                      std::strerror(errno));
            continue;
        }

        if (!session->config.sourceHost.empty()) {
            char addrBuffer[INET_ADDRSTRLEN]{};
            ::inet_ntop(AF_INET, &remote.sin_addr, addrBuffer, sizeof(addrBuffer));
            if (session->config.sourceHost != addrBuffer) {
                session->metrics.packetsDropped++;
                continue;
            }
        }

        RtpHeader header;
        if (!parseRtpHeader(packet.data(), static_cast<size_t>(bytes), header)) {
            session->metrics.packetsDropped++;
            continue;
        }

        if (header.payloadType != session->config.payloadType) {
            session->metrics.packetsDropped++;
            continue;
        }

        {
            std::lock_guard<std::mutex> metricsLock(session->metricsMutex);
            if (!session->metrics.ssrcLocked) {
                session->metrics.ssrc = header.ssrc;
                session->metrics.ssrcLocked = true;
                session->metrics.sequenceResets = 0;
                session->metrics.hasSequence = false;
            } else if (session->metrics.ssrc != header.ssrc) {
                session->metrics.ssrc = header.ssrc;
                session->metrics.sequenceResets++;
                session->metrics.hasSequence = false;
            }

            if (session->metrics.hasSequence) {
                uint16_t expected = static_cast<uint16_t>(session->metrics.lastSequence + 1);
                if (header.sequenceNumber != expected) {
                    uint16_t delta = static_cast<uint16_t>(header.sequenceNumber - expected);
                    session->metrics.packetsDropped += delta;
                }
            }
            session->metrics.lastSequence = header.sequenceNumber;
            session->metrics.hasSequence = true;
        }

        const uint8_t* payload = packet.data() + header.headerBytes;
        size_t payloadBytes = static_cast<size_t>(bytes) - header.headerBytes;
        size_t frameBytes = (session->config.bitsPerSample / 8) * session->config.channels;
        if (frameBytes == 0 || payloadBytes % frameBytes != 0) {
            session->metrics.packetsDropped++;
            continue;
        }

        size_t frames = payloadBytes / frameBytes;
        std::vector<float> interleaved(frames * session->config.channels);
        const uint8_t* cursor = payload;
        for (size_t frame = 0; frame < frames; ++frame) {
            for (uint8_t ch = 0; ch < session->config.channels; ++ch) {
                interleaved[frame * session->config.channels + ch] =
                    decodePcmSample(cursor, session->config.bitsPerSample,
                                    session->config.bigEndian, session->config.signedSamples);
                cursor += session->config.bitsPerSample / 8;
            }
        }

        {
            std::lock_guard<std::mutex> metricsLock(session->metricsMutex);
            session->metrics.packetsReceived++;
            session->metrics.bytesReceived += payloadBytes;
            session->metrics.lastPacketWallclock = std::chrono::system_clock::now();
            session->metrics.lastPacketMonotonic = std::chrono::steady_clock::now();
            session->metrics.lastRtpTimestamp = header.timestamp;
        }

        if (frameCallback_) {
            frameCallback_(interleaved.data(), frames, session->config.sampleRate);
        }

        updatePtpState(*session);
        emitTelemetry(*session);
    }
}

void RtpSessionManager::rtcpLoop(std::shared_ptr<Session> session) {
    std::vector<uint8_t> buffer(1024);
    while (session->running.load()) {
        pollfd pfd{session->rtcpSocket, POLLIN, 0};
        int pollResult = ::poll(&pfd, 1, kDefaultPollTimeoutMs);
        if (!session->running.load()) {
            break;
        }
        if (pollResult <= 0) {
            continue;
        }
        ssize_t bytes = ::recv(session->rtcpSocket, buffer.data(), buffer.size(), 0);
        if (bytes > 0) {
            std::lock_guard<std::mutex> metricsLock(session->metricsMutex);
            session->metrics.rtcpPackets++;
        }
    }
}

void RtpSessionManager::updatePtpState(Session& session) {
    if (!session.config.enablePtp || !ptpProvider_) {
        return;
    }
    PtpSyncState state = ptpProvider_();
    std::lock_guard<std::mutex> metricsLock(session.metricsMutex);
    session.metrics.ptpLocked = state.locked;
    session.metrics.ptpOffsetNs = state.offsetNs;
    session.metrics.ptpMeanPathNs = state.meanPathDelayNs;
}

void RtpSessionManager::recordWatchdog(Session& session) {
    std::lock_guard<std::mutex> metricsLock(session.metricsMutex);
    if (session.metrics.lastPacketMonotonic.time_since_epoch().count() == 0) {
        return;
    }
    auto now = std::chrono::steady_clock::now();
    if (now - session.metrics.lastPacketMonotonic >=
        std::chrono::milliseconds(session.config.watchdogTimeoutMs)) {
        LOG_WARN("RTP session {} has not received packets for {} ms", session.config.sessionId,
                 session.config.watchdogTimeoutMs);
        session.metrics.packetsDropped++;
        session.metrics.lastPacketMonotonic = now;
    }
}

void RtpSessionManager::emitTelemetry(Session& session) {
    if (!telemetryCallback_) {
        return;
    }
    auto now = std::chrono::steady_clock::now();
    if (now - session.lastTelemetry <
        std::chrono::milliseconds(session.config.telemetryIntervalMs)) {
        return;
    }
    SessionMetrics snapshot;
    {
        std::lock_guard<std::mutex> metricsLock(session.metricsMutex);
        snapshot = session.metrics;
    }
    session.lastTelemetry = now;
    telemetryCallback_(snapshot);
}

}  // namespace Network
