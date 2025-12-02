#include "rtp_engine_coordinator.h"

#include "logging/logger.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <optional>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace rtp_engine {

namespace {

static constexpr size_t RTP_DISCOVERY_MAX_PORTS = 16;
static constexpr uint32_t RTP_DISCOVERY_MIN_DURATION_MS = 50;
static constexpr uint32_t RTP_DISCOVERY_MAX_DURATION_MS = 5000;
static constexpr uint32_t RTP_DISCOVERY_MIN_COOLDOWN_MS = 250;
static constexpr uint32_t RTP_DISCOVERY_MAX_COOLDOWN_MS = 30000;
static constexpr size_t RTP_DISCOVERY_MAX_STREAM_LIMIT = 64;

struct RtpDiscoveryCandidate {
    std::string host;
    uint16_t port = 0;
    uint8_t payloadType = 0;
    bool multicast = false;
    uint32_t packets = 0;
    uint64_t firstSeenMs = 0;
    uint64_t lastSeenMs = 0;
};

uint64_t unix_time_millis() {
    auto now = std::chrono::system_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
}

bool is_ipv4_multicast(const in_addr& addr) {
    uint32_t ip = ntohl(addr.s_addr);
    return ip >= 0xE0000000u && ip <= 0xEFFFFFFFu;
}

std::string slugify_discovery_session_id(const std::string& host, uint16_t port) {
    std::string slug;
    slug.reserve(host.size() + 6);
    for (char ch : host) {
        unsigned char c = static_cast<unsigned char>(ch);
        if (std::isalnum(c)) {
            slug.push_back(static_cast<char>(std::tolower(c)));
        } else {
            slug.push_back('-');
        }
    }
    while (!slug.empty() && slug.front() == '-') {
        slug.erase(slug.begin());
    }
    while (!slug.empty() && slug.back() == '-') {
        slug.pop_back();
    }
    if (slug.empty()) {
        slug = "rtp";
    }
    slug.push_back('-');
    slug.append(std::to_string(port));
    if (slug.size() > 63) {
        slug.resize(63);
    }
    return slug;
}

std::string build_discovery_display_name(const std::string& host, uint16_t port,
                                         uint8_t payloadType) {
    std::ostringstream oss;
    oss << host << ":" << port << " (PT" << static_cast<int>(payloadType) << ")";
    return oss.str();
}

void join_discovery_multicast_group(int fd, const std::string& group,
                                    const std::string& interfaceName) {
    if (group.empty()) {
        return;
    }

    ip_mreq mreq{};
    if (::inet_pton(AF_INET, group.c_str(), &mreq.imr_multiaddr) != 1) {
        LOG_WARN("RTP discovery: invalid multicast group {}", group);
        return;
    }

    if (!interfaceName.empty()) {
        in_addr ifaceAddr{};
        if (::inet_pton(AF_INET, interfaceName.c_str(), &ifaceAddr) == 1) {
            mreq.imr_interface = ifaceAddr;
        } else {
            LOG_WARN("RTP discovery: interface '{}' is not IPv4 literal, using INADDR_ANY",
                     interfaceName);
            mreq.imr_interface.s_addr = htonl(INADDR_ANY);
        }
    } else {
        mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    }

    if (::setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0) {
        LOG_WARN("RTP discovery: failed to join multicast group {} ({})", group,
                 std::strerror(errno));
    }
}

bool extract_rtp_payload_type(const uint8_t* data, size_t length, uint8_t& payloadType) {
    if (length < 12) {
        return false;
    }
    if ((data[0] & 0xC0) != 0x80) {
        return false;
    }
    uint8_t csrcCount = data[0] & 0x0F;
    size_t headerBytes = 12 + static_cast<size_t>(csrcCount) * 4;
    if (headerBytes > length) {
        return false;
    }
    bool extension = (data[0] & 0x10) != 0;
    if (extension) {
        if (length < headerBytes + 4) {
            return false;
        }
        uint16_t extensionLength =
            static_cast<uint16_t>((data[headerBytes + 2] << 8) | data[headerBytes + 3]);
        headerBytes += 4 + static_cast<size_t>(extensionLength) * 4;
        if (headerBytes > length) {
            return false;
        }
    }
    payloadType = data[1] & 0x7F;
    return true;
}

std::optional<std::vector<RtpDiscoveryCandidate>> collect_rtp_discovery_candidates(
    const std::vector<uint16_t>& ports, uint32_t durationMs, bool allowMulticast,
    bool allowUnicast) {
    struct DiscoverySocket {
        int fd = -1;
        uint16_t port = 0;
    };

    std::vector<DiscoverySocket> sockets;
    sockets.reserve(ports.size());
    for (uint16_t port : ports) {
        int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd < 0) {
            LOG_WARN("RTP discovery: failed to create socket on port {} ({})", port,
                     std::strerror(errno));
            continue;
        }
        int reuse = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#ifdef SO_REUSEPORT
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse));
#endif
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port);
        if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            LOG_WARN("RTP discovery: bind failed on port {} ({})", port, std::strerror(errno));
            ::close(fd);
            continue;
        }
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }
        sockets.push_back({fd, port});
    }

    if (sockets.empty()) {
        return std::nullopt;
    }

    for (const auto& sock : sockets) {
        if (!allowMulticast) {
            continue;
        }
        join_discovery_multicast_group(sock.fd, "239.255.0.1", "");
    }

    std::vector<pollfd> pollFds;
    pollFds.reserve(sockets.size());
    for (const auto& sock : sockets) {
        pollfd pfd{};
        pfd.fd = sock.fd;
        pfd.events = POLLIN;
        pollFds.push_back(pfd);
    }

    auto start = std::chrono::steady_clock::now();
    auto deadline = start + std::chrono::milliseconds(durationMs);

    std::vector<RtpDiscoveryCandidate> candidates;

    while (std::chrono::steady_clock::now() < deadline) {
        int timeoutMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                             deadline - std::chrono::steady_clock::now())
                                             .count());
        if (timeoutMs < 0) {
            break;
        }
        int ret = ::poll(pollFds.data(), pollFds.size(), timeoutMs);
        if (ret < 0) {
            LOG_WARN("RTP discovery: poll failed ({})", std::strerror(errno));
            break;
        }
        if (ret == 0) {
            continue;
        }

        for (size_t i = 0; i < pollFds.size(); ++i) {
            if (!(pollFds[i].revents & POLLIN)) {
                continue;
            }
            uint8_t buffer[2048];
            sockaddr_in srcAddr{};
            socklen_t addrLen = sizeof(srcAddr);
            ssize_t n = ::recvfrom(pollFds[i].fd, buffer, sizeof(buffer), 0,
                                   reinterpret_cast<sockaddr*>(&srcAddr), &addrLen);
            if (n <= 0) {
                continue;
            }

            uint8_t payloadType = 0;
            if (!extract_rtp_payload_type(buffer, static_cast<size_t>(n), payloadType)) {
                continue;
            }

            char host[INET_ADDRSTRLEN] = {0};
            ::inet_ntop(AF_INET, &srcAddr.sin_addr, host, sizeof(host));
            // Use the daemon's listening port rather than the sender's ephemeral source port.
            uint16_t listenPort = sockets[i].port;

            bool multicast = is_ipv4_multicast(srcAddr.sin_addr);
            if (multicast && !allowMulticast) {
                continue;
            }
            if (!multicast && !allowUnicast) {
                continue;
            }

            auto nowMs = unix_time_millis();
            auto it = std::find_if(
                candidates.begin(), candidates.end(), [&](const RtpDiscoveryCandidate& c) {
                    return c.host == host && c.port == listenPort;
                });
            if (it == candidates.end()) {
                RtpDiscoveryCandidate cand;
                cand.host = host;
                cand.port = listenPort;
                cand.payloadType = payloadType;
                cand.multicast = multicast;
                cand.packets = 1;
                cand.firstSeenMs = nowMs;
                cand.lastSeenMs = nowMs;
                candidates.push_back(cand);
            } else {
                it->packets++;
                it->lastSeenMs = nowMs;
            }
        }
    }

    for (auto& sock : sockets) {
        ::close(sock.fd);
    }

    return candidates;
}

nlohmann::json build_discovery_response(const std::vector<RtpDiscoveryCandidate>& candidates,
                                        uint64_t scannedAtMs, uint32_t durationMs,
                                        size_t maxStreams,
                                        const Network::RtpSessionManager* manager,
                                        const AppConfig::RtpInputConfig& cfg) {
    nlohmann::json resp;
    resp["status"] = "ok";
    resp["data"]["scanned_at_unix_ms"] = scannedAtMs;
    resp["data"]["duration_ms"] = durationMs;
    resp["data"]["streams"] = nlohmann::json::array();

    size_t limit = std::min(maxStreams, candidates.size());
    for (size_t i = 0; i < limit; ++i) {
        const auto& candidate = candidates[i];
        std::string sessionId = slugify_discovery_session_id(candidate.host, candidate.port);
        bool existing = manager && manager->hasSession(sessionId);

        nlohmann::json stream;
        stream["session_id"] = sessionId;
        stream["display_name"] =
            build_discovery_display_name(candidate.host, candidate.port, candidate.payloadType);
        stream["source_host"] = candidate.host;
        stream["port"] = candidate.port;
        stream["status"] = candidate.packets >= 4   ? "active"
                           : candidate.packets >= 2 ? "detected"
                                                    : "probing";
        stream["existing_session"] = existing;
        stream["sample_rate"] = cfg.sampleRate;
        stream["channels"] = cfg.channels;
        stream["payload_type"] = candidate.payloadType;
        stream["multicast"] = candidate.multicast;
        if (candidate.multicast) {
            stream["multicast_group"] = candidate.host;
        } else {
            stream["multicast_group"] = nullptr;
        }
        stream["bind_address"] = cfg.bindAddress;
        stream["last_seen_unix_ms"] = candidate.lastSeenMs;
        stream["packet_count"] = candidate.packets;

        resp["data"]["streams"].push_back(stream);
    }

    return resp;
}

}  // namespace

RtpEngineCoordinator::RtpEngineCoordinator(Dependencies deps) : deps_(std::move(deps)) {}

Network::SessionConfig RtpEngineCoordinator::buildSessionConfig(
    const AppConfig::RtpInputConfig& cfg) const {
    Network::SessionConfig session{};
    session.sessionId = cfg.sessionId.empty() ? std::string("rtp_default") : cfg.sessionId;
    session.bindAddress = cfg.bindAddress;
    session.port = cfg.port;
    session.payloadType = cfg.payloadType;
    session.sampleRate = cfg.sampleRate;
    session.channels = cfg.channels;
    session.payloadType = cfg.payloadType;
    session.ptpInterface = cfg.ptpInterface;
    session.ptpDomain = cfg.ptpDomain;
    session.sdp = cfg.sdp;
    return session;
}

void RtpEngineCoordinator::maybeSwitchRateForRtp(uint32_t sessionRate,
                                                 const std::string& sessionId) {
    if (!deps_.isUpsamplerReady || !deps_.isUpsamplerReady() || sessionRate == 0) {
        return;
    }

    int targetRate = static_cast<int>(sessionRate);
    int currentRate =
        deps_.currentInputRate ? deps_.currentInputRate->load(std::memory_order_acquire) : 0;
    if (targetRate == currentRate) {
        return;
    }

    if (!deps_.isMultiRateEnabled || !deps_.isMultiRateEnabled()) {
        LOG_WARN("[RTP] Session {} requests {} Hz but multi-rate is disabled (engine at {} Hz)",
                 sessionId, targetRate, currentRate);
        return;
    }

    LOG_INFO("[RTP] Session {} -> switching engine input rate {} -> {} Hz", sessionId, currentRate,
             targetRate);
    if (!deps_.handleRateChange || !deps_.handleRateChange(targetRate)) {
        LOG_ERROR("[RTP] Failed to switch engine rate to {} Hz for session {}", targetRate,
                  sessionId);
    }
}

void RtpEngineCoordinator::ensureManagerInitialized() {
    if (manager_) {
        return;
    }

    auto frameCallback = [this](const float* interleaved, size_t frames, uint32_t sampleRate) {
        if (!deps_.runningFlag || !deps_.runningFlag->load()) {
            return;
        }
        int engineInput = deps_.getInputSampleRate ? deps_.getInputSampleRate() : 0;
        if (engineInput > 0 && sampleRate != static_cast<uint32_t>(engineInput)) {
            LOG_EVERY_N(WARN, 200,
                        "RTP session sample rate {} differs from engine input {} Hz. "
                        "Reconfigure engine to avoid drift.",
                        sampleRate, engineInput);
        }
        if (deps_.processInterleaved) {
            deps_.processInterleaved(interleaved, frames, sampleRate);
        }
    };

    auto ptpProvider =
        deps_.ptpProvider ? deps_.ptpProvider : []() -> Network::PtpSyncState { return {}; };

    auto telemetryCallback =
        deps_.telemetry ? deps_.telemetry : [](const Network::SessionMetrics&) {};

    manager_ =
        std::make_unique<Network::RtpSessionManager>(frameCallback, ptpProvider, telemetryCallback);
}

uint32_t RtpEngineCoordinator::clampDiscoveryDuration(uint32_t value) const {
    return std::clamp(value, RTP_DISCOVERY_MIN_DURATION_MS, RTP_DISCOVERY_MAX_DURATION_MS);
}

uint32_t RtpEngineCoordinator::clampDiscoveryCooldown(uint32_t value) const {
    return std::clamp(value, RTP_DISCOVERY_MIN_COOLDOWN_MS, RTP_DISCOVERY_MAX_COOLDOWN_MS);
}

size_t RtpEngineCoordinator::clampDiscoveryStreamLimit(size_t value) const {
    return std::clamp(value, static_cast<size_t>(1), RTP_DISCOVERY_MAX_STREAM_LIMIT);
}

std::vector<uint16_t> RtpEngineCoordinator::buildDiscoveryPorts(
    const AppConfig::RtpInputConfig& cfg) const {
    std::vector<uint16_t> ports = cfg.discoveryPorts;
    auto appendPort = [&](uint16_t candidate) {
        if (candidate < 1024 || candidate > 65535) {
            return;
        }
        if (std::find(ports.begin(), ports.end(), candidate) == ports.end()) {
            ports.push_back(candidate);
        }
    };

    appendPort(cfg.port);
    appendPort(5004);

    if (ports.empty()) {
        ports.push_back(cfg.port);
    }

    std::sort(ports.begin(), ports.end());
    ports.erase(std::unique(ports.begin(), ports.end()), ports.end());
    if (ports.size() > RTP_DISCOVERY_MAX_PORTS) {
        ports.resize(RTP_DISCOVERY_MAX_PORTS);
    }
    return ports;
}

nlohmann::json RtpEngineCoordinator::runDiscoveryScan() {
    if (!deps_.config) {
        nlohmann::json resp;
        resp["status"] = "error";
        resp["error_code"] = "AUDIO_RTP_SOCKET_ERROR";
        resp["message"] = "Missing configuration";
        return resp;
    }

    auto ports = buildDiscoveryPorts(deps_.config->rtp);
    if (ports.empty()) {
        nlohmann::json resp;
        resp["status"] = "error";
        resp["error_code"] = "AUDIO_RTP_SOCKET_ERROR";
        resp["message"] = "No discovery ports configured";
        return resp;
    }

    uint32_t durationMs = clampDiscoveryDuration(deps_.config->rtp.discoveryScanDurationMs);
    bool allowMulticast = deps_.config->rtp.discoveryEnableMulticast;
    bool allowUnicast = deps_.config->rtp.discoveryEnableUnicast;

    auto start = std::chrono::steady_clock::now();
    auto candidates =
        collect_rtp_discovery_candidates(ports, durationMs, allowMulticast, allowUnicast);
    auto elapsedMs = static_cast<uint32_t>(std::min<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                              start)
            .count(),
        std::numeric_limits<uint32_t>::max()));

    if (!candidates.has_value()) {
        nlohmann::json resp;
        resp["status"] = "error";
        resp["error_code"] = "AUDIO_RTP_SOCKET_ERROR";
        resp["message"] = "Failed to bind discovery sockets";
        return resp;
    }

    return build_discovery_response(
        candidates.value(), unix_time_millis(), elapsedMs,
        clampDiscoveryStreamLimit(deps_.config->rtp.discoveryMaxStreams), manager_.get(),
        deps_.config->rtp);
}

nlohmann::json RtpEngineCoordinator::getOrRunDiscovery() {
    uint32_t cooldownMs = clampDiscoveryCooldown(
        deps_.config ? deps_.config->rtp.discoveryCooldownMs : RTP_DISCOVERY_MIN_COOLDOWN_MS);
    auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(discoveryMutex_);
        if (!discoveryCache_.is_null() &&
            std::chrono::duration_cast<std::chrono::milliseconds>(now - lastDiscovery_).count() <
                cooldownMs) {
            return discoveryCache_;
        }
    }

    nlohmann::json result = runDiscoveryScan();
    if (result.contains("status") && result["status"] == "ok") {
        std::lock_guard<std::mutex> lock(discoveryMutex_);
        discoveryCache_ = result;
        lastDiscovery_ = std::chrono::steady_clock::now();
    }
    return result;
}

void RtpEngineCoordinator::startFromConfig() {
    if (!deps_.config || !deps_.config->rtp.enabled || !deps_.config->rtp.autoStart) {
        return;
    }

    ensureManagerInitialized();
    Network::SessionConfig sessionCfg = buildSessionConfig(deps_.config->rtp);
    std::string error;
    if (!manager_->startSession(sessionCfg, error)) {
        LOG_ERROR("Failed to auto-start RTP session {}: {}", sessionCfg.sessionId, error);
    } else {
        LOG_INFO("RTP session '{}' auto-started on {}:{}", sessionCfg.sessionId,
                 sessionCfg.bindAddress, sessionCfg.port);
        maybeSwitchRateForRtp(sessionCfg.sampleRate, sessionCfg.sessionId);
    }
}

bool RtpEngineCoordinator::handleZeroMqCommand(const std::string& cmdType,
                                               const nlohmann::json& message,
                                               std::string& responseOut) {
    if (cmdType == "RTP_START_SESSION" || cmdType == "StartSession") {
        if (!message.contains("params") || !message["params"].is_object()) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "IPC_INVALID_PARAMS";
            resp["message"] = "Missing params object";
            responseOut = resp.dump();
            return true;
        }
        ensureManagerInitialized();
        Network::SessionConfig sessionCfg;
        std::string parseError;
        if (!Network::sessionConfigFromJson(message["params"], sessionCfg, parseError)) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "VALIDATION_INVALID_CONFIG";
            resp["message"] = parseError;
            responseOut = resp.dump();
            return true;
        }
        if (!manager_) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "AUDIO_RTP_SOCKET_ERROR";
            resp["message"] = "RTP manager unavailable";
            responseOut = resp.dump();
            return true;
        }
        std::string startError;
        if (!manager_->startSession(sessionCfg, startError)) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "AUDIO_RTP_SOCKET_ERROR";
            resp["message"] = startError;
            responseOut = resp.dump();
            return true;
        }
        nlohmann::json resp;
        resp["status"] = "ok";
        resp["message"] = "RTP session started";
        resp["data"] = Network::sessionConfigToJson(sessionCfg);
        responseOut = resp.dump();
        maybeSwitchRateForRtp(sessionCfg.sampleRate, sessionCfg.sessionId);
        return true;
    } else if (cmdType == "RTP_STOP_SESSION" || cmdType == "StopSession") {
        if (!message.contains("params") || !message["params"].is_object() ||
            !message["params"].contains("session_id")) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "IPC_INVALID_PARAMS";
            resp["message"] = "Missing params.session_id";
            responseOut = resp.dump();
            return true;
        }
        if (!manager_) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "AUDIO_RTP_SESSION_NOT_FOUND";
            resp["message"] = "RTP manager not initialized";
            responseOut = resp.dump();
            return true;
        }
        std::string sessionId = message["params"]["session_id"].get<std::string>();
        std::string stopError;
        if (!manager_->stopSession(sessionId, stopError)) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "AUDIO_RTP_SESSION_NOT_FOUND";
            resp["message"] = stopError;
            responseOut = resp.dump();
            return true;
        }
        nlohmann::json resp;
        resp["status"] = "ok";
        resp["message"] = "RTP session stopped";
        resp["data"]["session_id"] = sessionId;
        responseOut = resp.dump();
        return true;
    } else if (cmdType == "RTP_LIST_SESSIONS" || cmdType == "ListSessions") {
        ensureManagerInitialized();
        nlohmann::json resp;
        resp["status"] = "ok";
        nlohmann::json sessionsJson = nlohmann::json::array();
        if (manager_) {
            auto sessions = manager_->listSessions();
            for (const auto& metrics : sessions) {
                sessionsJson.push_back(Network::sessionMetricsToJson(metrics));
            }
        }
        resp["data"]["sessions"] = sessionsJson;
        responseOut = resp.dump();
        return true;
    } else if (cmdType == "RTP_GET_SESSION" || cmdType == "GetSession") {
        if (!message.contains("params") || !message["params"].is_object() ||
            !message["params"].contains("session_id")) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "IPC_INVALID_PARAMS";
            resp["message"] = "Missing params.session_id";
            responseOut = resp.dump();
            return true;
        }
        if (!manager_) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "AUDIO_RTP_SESSION_NOT_FOUND";
            resp["message"] = "RTP manager not initialized";
            responseOut = resp.dump();
            return true;
        }
        std::string sessionId = message["params"]["session_id"].get<std::string>();
        auto metrics = manager_->getMetrics(sessionId);
        if (!metrics.has_value()) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "AUDIO_RTP_SESSION_NOT_FOUND";
            resp["message"] = "Session not found: " + sessionId;
            responseOut = resp.dump();
            return true;
        }
        nlohmann::json resp;
        resp["status"] = "ok";
        resp["data"] = Network::sessionMetricsToJson(metrics.value());
        responseOut = resp.dump();
        return true;
    } else if (cmdType == "RTP_DISCOVER_STREAMS" || cmdType == "DiscoverStreams") {
        nlohmann::json resp = getOrRunDiscovery();
        responseOut = resp.dump();
        return true;
    } else if (cmdType == "SWITCH_RATE") {
        if (!deps_.isUpsamplerReady || !deps_.isUpsamplerReady()) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "ENGINE_NOT_INITIALIZED";
            resp["message"] = "Upsampler not initialized";
            responseOut = resp.dump();
            return true;
        }
        if (!deps_.isMultiRateEnabled || !deps_.isMultiRateEnabled()) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "MULTI_RATE_NOT_ENABLED";
            resp["message"] = "Multi-rate mode not enabled";
            responseOut = resp.dump();
            return true;
        }
        if (!message.contains("params") || !message["params"].contains("input_rate")) {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "IPC_INVALID_PARAMS";
            resp["message"] = "Missing params.input_rate field";
            responseOut = resp.dump();
            return true;
        }
        int targetRate = message["params"]["input_rate"].get<int>();
        int currentRate =
            deps_.currentInputRate ? deps_.currentInputRate->load(std::memory_order_acquire) : 0;

        if (targetRate == currentRate) {
            nlohmann::json resp;
            resp["status"] = "ok";
            resp["data"]["input_rate"] = currentRate;
            resp["data"]["output_rate"] =
                deps_.currentOutputRate ? deps_.currentOutputRate->load(std::memory_order_acquire)
                                        : 0;
            resp["data"]["upsample_ratio"] = deps_.getUpsampleRatio ? deps_.getUpsampleRatio() : 0;
            resp["data"]["alsa_reconfigure_scheduled"] =
                deps_.alsaReconfigureNeeded
                    ? deps_.alsaReconfigureNeeded->load(std::memory_order_acquire)
                    : false;
            resp["message"] = "Already at requested rate";
            responseOut = resp.dump();
            return true;
        }

        bool switch_success = deps_.handleRateChange ? deps_.handleRateChange(targetRate) : false;
        if (switch_success) {
            nlohmann::json resp;
            resp["status"] = "ok";
            resp["data"]["input_rate"] =
                deps_.currentInputRate ? deps_.currentInputRate->load(std::memory_order_acquire)
                                       : targetRate;
            resp["data"]["output_rate"] =
                deps_.currentOutputRate ? deps_.currentOutputRate->load(std::memory_order_acquire)
                                        : 0;
            resp["data"]["upsample_ratio"] = deps_.getUpsampleRatio ? deps_.getUpsampleRatio() : 0;
            resp["data"]["alsa_reconfigure_scheduled"] =
                deps_.alsaReconfigureNeeded
                    ? deps_.alsaReconfigureNeeded->load(std::memory_order_acquire)
                    : false;
            responseOut = resp.dump();
        } else {
            nlohmann::json resp;
            resp["status"] = "error";
            resp["error_code"] = "RATE_SWITCH_FAILED";
            resp["message"] = "Failed to switch to rate " + std::to_string(targetRate) + " Hz";
            responseOut = resp.dump();
        }
        return true;
    }
    return false;
}

void RtpEngineCoordinator::shutdown() {
    if (!manager_) {
        return;
    }
    manager_->stopAll();
    manager_.reset();
}

}  // namespace rtp_engine
