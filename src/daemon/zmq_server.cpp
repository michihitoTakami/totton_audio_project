#include "daemon/zmq_server.h"

#include <cstdio>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <zmq.hpp>

namespace daemon_ipc {
namespace {

constexpr const char* kJsonErrorStatus = "error";
constexpr const char* kJsonOkStatus = "ok";

bool startsWith(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

}  // namespace

ZmqCommandServer::ZmqCommandServer(std::string endpoint, int recvTimeoutMs)
    : endpoint_(std::move(endpoint)),
      pubEndpoint_(derivePubEndpoint(endpoint_)),
      recvTimeoutMs_(recvTimeoutMs) {}

ZmqCommandServer::~ZmqCommandServer() {
    stop();
}

void ZmqCommandServer::registerCommand(const std::string& command, Handler handler) {
    handlers_[command] = std::move(handler);
}

bool ZmqCommandServer::start() {
    if (running_.load()) {
        return true;
    }

    try {
        context_ = std::make_unique<zmq::context_t>(1);
        repSocket_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::rep);
        repSocket_->set(zmq::sockopt::rcvtimeo, recvTimeoutMs_);
        repSocket_->set(zmq::sockopt::linger, 0);

        cleanupIpcPath(endpoint_);
        repSocket_->bind(endpoint_);

        pubSocket_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::pub);
        cleanupIpcPath(pubEndpoint_);
        pubSocket_->bind(pubEndpoint_);

        running_.store(true);
        bindFailed_.store(false);
        serverThread_ = std::thread(&ZmqCommandServer::serverLoop, this);

        std::cout << "ZeroMQ: Listening on " << endpoint_ << std::endl;
        std::cout << "ZeroMQ: PUB socket on " << pubEndpoint_ << std::endl;
        return true;
    } catch (const zmq::error_t& e) {
        std::cerr << "ZeroMQ: Fatal error - " << e.what() << std::endl;
        bindFailed_.store(true);
        running_.store(false);
        cleanupSockets();
        return false;
    }
}

void ZmqCommandServer::stop() {
    if (!running_.exchange(false)) {
        return;
    }

    try {
        zmq::context_t tempCtx{1};
        zmq::socket_t tempSocket{tempCtx, zmq::socket_type::req};
        tempSocket.connect(endpoint_);
        tempSocket.send(zmq::buffer("SHUTDOWN"), zmq::send_flags::dontwait);
    } catch (...) {}

    if (serverThread_.joinable()) {
        serverThread_.join();
    }

    cleanupSockets();
    cleanupIpcPath(endpoint_);
    cleanupIpcPath(pubEndpoint_);
}

bool ZmqCommandServer::publish(const std::string& message) {
    std::lock_guard<std::mutex> lock(pubMutex_);
    if (!pubSocket_) {
        return false;
    }

    try {
        pubSocket_->send(zmq::buffer(message), zmq::send_flags::dontwait);
        return true;
    } catch (const zmq::error_t& e) {
        std::cerr << "ZeroMQ: PUB send failed: " << e.what() << std::endl;
        return false;
    }
}

ZmqRequest ZmqCommandServer::buildRequest(const std::string& raw) const {
    ZmqRequest request;
    request.raw = raw;

    if (raw.empty()) {
        return request;
    }

    if (raw.front() == '{') {
        request.isJson = true;
        try {
            request.json = nlohmann::json::parse(raw);
            if (request.json->contains("cmd") && (*request.json)["cmd"].is_string()) {
                request.command = (*request.json)["cmd"].get<std::string>();
            }
        } catch (const nlohmann::json::exception& e) {
            request.parseError = e.what();
        }
        return request;
    }

    auto colonPos = raw.find(':');
    if (colonPos != std::string::npos) {
        request.command = raw.substr(0, colonPos);
        request.payload = raw.substr(colonPos + 1);
    } else {
        request.command = raw;
    }

    auto trimNull = [](std::string& value) {
        auto pos = value.find('\0');
        if (pos != std::string::npos) {
            value.erase(pos);
        }
    };
    trimNull(request.command);
    trimNull(request.payload);

    return request;
}

std::string ZmqCommandServer::dispatchRequest(const ZmqRequest& request) {
    if (!request.parseError.empty()) {
        return buildErrorResponse(request, "IPC_PROTOCOL_ERROR",
                                  "JSON parse error: " + request.parseError);
    }

    auto it = handlers_.find(request.command);
    if (it == handlers_.end()) {
        std::string message = request.command.empty() ? "Unknown command" : request.command;
        return buildErrorResponse(request, "IPC_INVALID_COMMAND", "Unknown command: " + message);
    }

    try {
        return it->second(request);
    } catch (const std::exception& e) {
        return buildErrorResponse(request, "IPC_PROTOCOL_ERROR",
                                  std::string("Handler exception: ") + e.what());
    }
}

std::string ZmqCommandServer::buildErrorResponse(const ZmqRequest& request, const std::string& code,
                                                 const std::string& message) const {
    if (request.isJson) {
        nlohmann::json resp;
        resp["status"] = kJsonErrorStatus;
        resp["error_code"] = code;
        resp["message"] = message;
        return resp.dump();
    }
    return "ERR:" + message;
}

void ZmqCommandServer::serverLoop() {
    while (running_.load()) {
        try {
            zmq::message_t request;
            auto recvResult = repSocket_->recv(request, zmq::recv_flags::none);

            if (!recvResult) {
                continue;
            }

            std::string raw(static_cast<char*>(request.data()), request.size());
            if (raw == "SHUTDOWN") {
                repSocket_->send(zmq::buffer("OK"), zmq::send_flags::dontwait);
                continue;
            }

            std::string response = dispatchRequest(buildRequest(raw));
            repSocket_->send(zmq::buffer(response), zmq::send_flags::none);
        } catch (const zmq::error_t& e) {
            if (running_.load()) {
                std::cerr << "ZeroMQ: Listener error - " << e.what() << std::endl;
            }
        }
    }

    cleanupSockets();
}

void ZmqCommandServer::cleanupSockets() {
    if (repSocket_) {
        try {
            repSocket_->close();
        } catch (...) {}
    }
    if (pubSocket_) {
        try {
            pubSocket_->close();
        } catch (...) {}
    }
    repSocket_.reset();
    pubSocket_.reset();
    context_.reset();
}

void ZmqCommandServer::cleanupIpcPath(const std::string& endpoint) const {
    if (!startsWith(endpoint, "ipc://")) {
        return;
    }
    std::string path = endpoint.substr(6);
    if (path.empty()) {
        return;
    }
    std::remove(path.c_str());
}

std::string ZmqCommandServer::derivePubEndpoint(const std::string& endpoint) {
    if (startsWith(endpoint, "ipc://")) {
        return endpoint + DaemonConstants::ZEROMQ_PUB_SUFFIX;
    }

    if (startsWith(endpoint, "tcp://")) {
        auto colonPos = endpoint.rfind(':');
        if (colonPos != std::string::npos) {
            try {
                int port = std::stoi(endpoint.substr(colonPos + 1));
                return endpoint.substr(0, colonPos + 1) + std::to_string(port + 1);
            } catch (...) {}
        }
    }

    return endpoint + DaemonConstants::ZEROMQ_PUB_SUFFIX;
}

}  // namespace daemon_ipc
