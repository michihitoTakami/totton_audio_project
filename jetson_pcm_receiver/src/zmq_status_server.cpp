#include "zmq_status_server.h"

#include "connection_mode.h"
#include "logging.h"

#include <chrono>
#include <iostream>
#include <optional>

namespace {

constexpr const char* kDefaultEndpoint = "ipc:///tmp/jetson_pcm_receiver.sock";

std::string okResponse(const daemon_ipc::ZmqRequest& request, const nlohmann::json& data) {
    if (request.isJson) {
        nlohmann::json resp;
        resp["status"] = "ok";
        if (!data.is_null() && !data.empty()) {
            resp["data"] = data;
        }
        return resp.dump();
    }
    if (data.is_null() || data.empty()) {
        return "OK";
    }
    return "OK:" + data.dump();
}

std::string extractToken(const daemon_ipc::ZmqRequest& request) {
    if (!request.json) {
        return {};
    }
    if (request.json->contains("token") && (*request.json)["token"].is_string()) {
        return (*request.json)["token"].get<std::string>();
    }
    if (request.json->contains("params") && (*request.json)["params"].is_object()) {
        const auto& params = (*request.json)["params"];
        if (params.contains("token") && params["token"].is_string()) {
            return params["token"].get<std::string>();
        }
    }
    return {};
}

}  // namespace

ZmqStatusServer::ZmqStatusServer(StatusTracker& status, PcmStreamConfig& config,
                                 std::mutex& configMutex, std::atomic_bool& stopFlag)
    : status_(status), config_(config), configMutex_(configMutex), stopFlag_(stopFlag) {}

ZmqStatusServer::~ZmqStatusServer() {
    stop();
}

bool ZmqStatusServer::start(const Options& options) {
    options_ = options;
    if (!options_.enabled) {
        return true;
    }

    const std::string endpoint =
        options_.endpoint.empty() ? std::string(kDefaultEndpoint) : options_.endpoint;

    server_ = std::make_unique<daemon_ipc::ZmqCommandServer>(endpoint);
    server_->registerCommand("PING", [this](const auto& req) { return handlePing(req); });
    server_->registerCommand("STATUS", [this](const auto& req) { return handleStatus(req); });
    server_->registerCommand("SET_CACHE", [this](const auto& req) { return handleSetCache(req); });
    server_->registerCommand("SET_RING_BUFFER",
                             [this](const auto& req) { return handleSetCache(req); });
    server_->registerCommand("RESTART", [this](const auto& req) { return handleRestart(req); });
    server_->registerCommand("SHUTDOWN", [this](const auto& req) { return handleRestart(req); });

    if (!server_->start()) {
        server_.reset();
        return false;
    }

    startPublisher();
    std::cout << "[ZMQ] REP=" << server_->endpoint() << " PUB=" << server_->pubEndpoint()
              << std::endl;
    return true;
}

void ZmqStatusServer::stop() {
    stopPublisher();
    if (server_) {
        server_->stop();
        server_.reset();
    }
}

std::string ZmqStatusServer::endpoint() const {
    static std::string empty;
    if (!server_) {
        return empty;
    }
    return server_->endpoint();
}

std::string ZmqStatusServer::pubEndpoint() const {
    static std::string empty;
    if (!server_) {
        return empty;
    }
    return server_->pubEndpoint();
}

bool ZmqStatusServer::checkToken(const daemon_ipc::ZmqRequest& request, std::string& error) const {
    if (options_.token.empty()) {
        return true;
    }
    const std::string provided = extractToken(request);
    if (provided.empty()) {
        error = "Missing token";
        return false;
    }
    if (provided != options_.token) {
        error = "Invalid token";
        return false;
    }
    return true;
}

std::string ZmqStatusServer::buildError(const daemon_ipc::ZmqRequest& request,
                                        const std::string& code, const std::string& message) const {
    if (request.isJson) {
        nlohmann::json resp;
        resp["status"] = "error";
        resp["error_code"] = code;
        resp["message"] = message;
        return resp.dump();
    }
    return "ERR:" + message;
}

std::string ZmqStatusServer::handlePing(const daemon_ipc::ZmqRequest& request) {
    std::string error;
    if (!checkToken(request, error)) {
        return buildError(request, "IPC_UNAUTHORIZED", error);
    }
    return okResponse(request, {});
}

std::string ZmqStatusServer::handleStatus(const daemon_ipc::ZmqRequest& request) {
    std::string error;
    if (!checkToken(request, error)) {
        return buildError(request, "IPC_UNAUTHORIZED", error);
    }

    nlohmann::json data = buildStatusJson();
    if (server_) {
        data["rep_endpoint"] = server_->endpoint();
        data["pub_endpoint"] = server_->pubEndpoint();
    }
    return okResponse(request, data);
}

std::string ZmqStatusServer::handleSetCache(const daemon_ipc::ZmqRequest& request) {
    std::string error;
    if (!checkToken(request, error)) {
        return buildError(request, "IPC_UNAUTHORIZED", error);
    }
    if (!request.json || !request.json->contains("params")) {
        return buildError(request, "IPC_INVALID_PARAMS", "params object is required");
    }
    const auto& params = (*request.json)["params"];
    if (!params.is_object()) {
        return buildError(request, "IPC_INVALID_PARAMS", "params must be an object");
    }

    std::size_t newRing = config_.ringBufferFrames;
    std::size_t newWatermark = config_.watermarkFrames;
    if (params.contains("ring_buffer_frames")) {
        if (!params["ring_buffer_frames"].is_number_unsigned()) {
            return buildError(request, "IPC_INVALID_PARAMS",
                              "ring_buffer_frames must be an unsigned integer");
        }
        newRing = params["ring_buffer_frames"].get<std::size_t>();
    }
    if (params.contains("watermark_frames")) {
        if (!params["watermark_frames"].is_number_unsigned()) {
            return buildError(request, "IPC_INVALID_PARAMS",
                              "watermark_frames must be an unsigned integer");
        }
        newWatermark = params["watermark_frames"].get<std::size_t>();
    }

    {
        std::lock_guard<std::mutex> lock(configMutex_);
        config_.ringBufferFrames = newRing;
        config_.watermarkFrames = newWatermark;
    }
    status_.updateRingConfig(newRing, newWatermark);

    nlohmann::json data;
    data["ring_buffer_frames"] = newRing;
    data["watermark_frames"] = newWatermark;
    return okResponse(request, data);
}

std::string ZmqStatusServer::handleRestart(const daemon_ipc::ZmqRequest& request) {
    std::string error;
    if (!checkToken(request, error)) {
        return buildError(request, "IPC_UNAUTHORIZED", error);
    }
    restartRequested_.store(true);
    stopFlag_.store(true, std::memory_order_relaxed);
    nlohmann::json data;
    data["message"] = "Restart requested";
    return okResponse(request, data);
}

nlohmann::json ZmqStatusServer::buildStatusJson() {
    auto snap = status_.snapshot();
    nlohmann::json data;
    data["listening"] = snap.listening;
    data["bound_port"] = snap.boundPort;
    data["client_connected"] = snap.clientConnected;
    data["streaming"] = snap.streaming;
    data["xrun_count"] = snap.xrunCount;
    data["ring_buffer_frames"] = snap.ring.configuredFrames;
    data["watermark_frames"] = snap.ring.watermarkFrames;
    data["buffered_frames"] = snap.ring.bufferedFrames;
    data["max_buffered_frames"] = snap.ring.maxBufferedFrames;
    data["dropped_frames"] = snap.ring.droppedFrames;
    {
        std::lock_guard<std::mutex> lock(configMutex_);
        data["connection_mode"] = toString(config_.connectionMode);
        data["priority_clients"] = config_.priorityClients;
    }
    if (snap.header.present) {
        nlohmann::json hdr;
        hdr["sample_rate"] = snap.header.header.sample_rate;
        hdr["channels"] = snap.header.header.channels;
        hdr["format"] = snap.header.header.format;
        hdr["version"] = snap.header.header.version;
        data["last_header"] = hdr;
    } else {
        data["last_header"] = nullptr;
    }
    return data;
}

void ZmqStatusServer::startPublisher() {
    if (!server_ || options_.publishIntervalMs <= 0) {
        return;
    }
    if (publisherRunning_.exchange(true)) {
        return;
    }
    publisherThread_ = std::thread([this]() { publisherLoop(); });
}

void ZmqStatusServer::stopPublisher() {
    if (!publisherRunning_.exchange(false)) {
        return;
    }
    if (publisherThread_.joinable()) {
        publisherThread_.join();
    }
}

void ZmqStatusServer::publisherLoop() {
    using namespace std::chrono_literals;
    while (publisherRunning_.load()) {
        if (server_ && server_->isRunning()) {
            try {
                nlohmann::json payload = buildStatusJson();
                payload["event"] = "status";
                server_->publish(payload.dump());
            } catch (const std::exception& e) {
                logWarn(std::string("[ZMQ] publish failed: ") + e.what());
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(options_.publishIntervalMs));
    }
}

bool ZmqStatusServer::publishHeaderChange(const PcmHeader& header,
                                          const std::optional<PcmHeader>& previousHeader) {
    if (!options_.enabled || !options_.publishHeaderEvents) {
        return false;
    }
    if (!server_ || !server_->isRunning()) {
        return false;
    }

    nlohmann::json payload;
    payload["event"] = "pcm_header_changed";
    payload["header"]["sample_rate"] = header.sample_rate;
    payload["header"]["channels"] = header.channels;
    payload["header"]["format"] = header.format;
    payload["header"]["version"] = header.version;
    if (previousHeader.has_value()) {
        payload["previous_header"]["sample_rate"] = previousHeader->sample_rate;
        payload["previous_header"]["channels"] = previousHeader->channels;
        payload["previous_header"]["format"] = previousHeader->format;
        payload["previous_header"]["version"] = previousHeader->version;
    }
    payload["timestamp_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();

    try {
        return server_->publish(payload.dump());
    } catch (const std::exception& e) {
        logWarn(std::string("[ZMQ] header publish failed: ") + e.what());
        return false;
    }
}
