#include "zeromq_interface.h"

#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <zmq.hpp>

using json = nlohmann::json;

namespace ZMQComm {

// ============================================================
// String conversion utilities
// ============================================================

const char* commandTypeToString(CommandType type) {
    switch (type) {
    case CommandType::LOAD_IR:
        return "LOAD_IR";
    case CommandType::SET_GAIN:
        return "SET_GAIN";
    case CommandType::SOFT_RESET:
        return "SOFT_RESET";
    case CommandType::GET_STATUS:
        return "GET_STATUS";
    case CommandType::SWITCH_RATE:
        return "SWITCH_RATE";
    case CommandType::APPLY_EQ:
        return "APPLY_EQ";
    case CommandType::RESTORE_EQ:
        return "RESTORE_EQ";
    case CommandType::SHUTDOWN:
        return "SHUTDOWN";
    default:
        return "UNKNOWN";
    }
}

CommandType stringToCommandType(const std::string& str) {
    static const std::map<std::string, CommandType> lookup = {
        {"LOAD_IR", CommandType::LOAD_IR},         {"SET_GAIN", CommandType::SET_GAIN},
        {"SOFT_RESET", CommandType::SOFT_RESET},   {"GET_STATUS", CommandType::GET_STATUS},
        {"SWITCH_RATE", CommandType::SWITCH_RATE}, {"APPLY_EQ", CommandType::APPLY_EQ},
        {"RESTORE_EQ", CommandType::RESTORE_EQ},   {"SHUTDOWN", CommandType::SHUTDOWN}};

    auto it = lookup.find(str);
    if (it != lookup.end()) {
        return it->second;
    }
    throw std::invalid_argument("Unknown command type: " + str);
}

const char* responseStatusToString(ResponseStatus status) {
    switch (status) {
    case ResponseStatus::OK:
        return "ok";
    case ResponseStatus::ERROR:
        return "error";
    case ResponseStatus::INVALID_COMMAND:
        return "invalid_command";
    case ResponseStatus::INVALID_PARAMS:
        return "invalid_params";
    default:
        return "unknown";
    }
}

// ============================================================
// JSON utilities
// ============================================================

namespace JSON {

std::string buildCommand(CommandType type, const std::string& params) {
    json j;
    j["cmd"] = commandTypeToString(type);
    if (!params.empty()) {
        try {
            j["params"] = json::parse(params);
        } catch (...) {
            // If params is not valid JSON, store as string
            j["params"] = params;
        }
    }
    return j.dump();
}

bool parseCommand(const std::string& jsonStr, CommandType& type, std::string& params) {
    try {
        auto j = json::parse(jsonStr);
        if (!j.contains("cmd")) {
            return false;
        }
        type = stringToCommandType(j["cmd"].get<std::string>());
        if (j.contains("params")) {
            if (j["params"].is_string()) {
                params = j["params"].get<std::string>();
            } else {
                params = j["params"].dump();
            }
        } else {
            params = "";
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ZMQ JSON parse error: " << e.what() << std::endl;
        return false;
    }
}

std::string buildResponse(ResponseStatus status, const std::string& message,
                          const std::string& data) {
    json j;
    j["status"] = responseStatusToString(status);
    if (!message.empty()) {
        j["message"] = message;
    }
    if (!data.empty()) {
        try {
            j["data"] = json::parse(data);
        } catch (...) {
            j["data"] = data;
        }
    }
    return j.dump();
}

std::string buildErrorResponse(AudioEngine::ErrorCode code, const std::string& message,
                               const AudioEngine::InnerError& innerError) {
    json j;
    j["status"] = "error";
    j["error_code"] = AudioEngine::errorCodeToString(code);
    j["message"] = message;

    // Build inner_error object
    json inner;
    if (!innerError.cpp_code.empty()) {
        inner["cpp_code"] = innerError.cpp_code;
    } else {
        inner["cpp_code"] = AudioEngine::errorCodeToHex(code);
    }
    if (!innerError.cpp_message.empty()) {
        inner["cpp_message"] = innerError.cpp_message;
    }
    if (innerError.alsa_errno.has_value()) {
        inner["alsa_errno"] = innerError.alsa_errno.value();
    } else {
        inner["alsa_errno"] = nullptr;
    }
    if (innerError.alsa_func.has_value()) {
        inner["alsa_func"] = innerError.alsa_func.value();
    } else {
        inner["alsa_func"] = nullptr;
    }
    if (innerError.cuda_error.has_value()) {
        inner["cuda_error"] = innerError.cuda_error.value();
    } else {
        inner["cuda_error"] = nullptr;
    }
    j["inner_error"] = inner;

    return j.dump();
}

std::string buildOkResponse(const std::string& message, const std::string& data) {
    json j;
    j["status"] = "ok";
    if (!message.empty()) {
        j["message"] = message;
    }
    if (!data.empty()) {
        try {
            j["data"] = json::parse(data);
        } catch (...) {
            j["data"] = data;
        }
    }
    return j.dump();
}

bool parseResponse(const std::string& jsonStr, ResponseStatus& status, std::string& message,
                   std::string& data) {
    try {
        auto j = json::parse(jsonStr);
        if (!j.contains("status")) {
            return false;
        }

        std::string statusStr = j["status"].get<std::string>();
        if (statusStr == "ok")
            status = ResponseStatus::OK;
        else if (statusStr == "error")
            status = ResponseStatus::ERROR;
        else if (statusStr == "invalid_command")
            status = ResponseStatus::INVALID_COMMAND;
        else if (statusStr == "invalid_params")
            status = ResponseStatus::INVALID_PARAMS;
        else
            status = ResponseStatus::ERROR;

        message = j.value("message", "");
        if (j.contains("data")) {
            if (j["data"].is_string()) {
                data = j["data"].get<std::string>();
            } else {
                data = j["data"].dump();
            }
        } else {
            data = "";
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ZMQ JSON parse error: " << e.what() << std::endl;
        return false;
    }
}

bool parseErrorResponse(const std::string& jsonStr, std::string& errorCode, std::string& message,
                        std::string& innerErrorJson) {
    try {
        auto j = json::parse(jsonStr);
        if (!j.contains("status") || j["status"].get<std::string>() != "error") {
            return false;
        }

        errorCode = j.value("error_code", "UNKNOWN_ERROR");
        message = j.value("message", "");

        if (j.contains("inner_error") && j["inner_error"].is_object()) {
            innerErrorJson = j["inner_error"].dump();
        } else {
            innerErrorJson = "";
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ZMQ JSON parse error: " << e.what() << std::endl;
        return false;
    }
}

std::string buildStatus(const EngineStatus& status) {
    json j;
    j["type"] = "status";
    j["data"]["input_rate"] = status.inputSampleRate;
    j["data"]["output_rate"] = status.outputSampleRate;
    j["data"]["gpu_load"] = status.gpuLoad;
    j["data"]["buffer_level"] = status.bufferLevel;
    j["data"]["eq_applied"] = status.eqApplied;
    j["data"]["rate_family"] = status.currentRateFamily;
    j["data"]["frames_processed"] = status.framesProcessed;
    return j.dump();
}

bool parseStatus(const std::string& jsonStr, EngineStatus& status) {
    try {
        auto j = json::parse(jsonStr);
        if (!j.contains("data")) {
            return false;
        }
        auto& d = j["data"];
        status.inputSampleRate = d.value("input_rate", 0);
        status.outputSampleRate = d.value("output_rate", 0);
        status.gpuLoad = d.value("gpu_load", 0.0);
        status.bufferLevel = d.value("buffer_level", 0.0);
        status.eqApplied = d.value("eq_applied", false);
        status.currentRateFamily = d.value("rate_family", "");
        status.framesProcessed = d.value("frames_processed", 0);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ZMQ JSON parse error: " << e.what() << std::endl;
        return false;
    }
}

}  // namespace JSON

// ============================================================
// ZMQServer implementation
// ============================================================

struct ZMQServer::Impl {
    zmq::context_t context{1};
    std::unique_ptr<zmq::socket_t> repSocket;
    std::unique_ptr<zmq::socket_t> pubSocket;
    std::string endpoint;
    std::string pubEndpoint;
};

ZMQServer::ZMQServer() : impl_(std::make_unique<Impl>()) {}

ZMQServer::~ZMQServer() {
    stop();
}

bool ZMQServer::initialize(const std::string& endpoint) {
    try {
        impl_->endpoint = endpoint;
        impl_->repSocket = std::make_unique<zmq::socket_t>(impl_->context, zmq::socket_type::rep);
        impl_->repSocket->bind(endpoint);

        // Setup PUB socket for status updates
        // Derive PUB endpoint from REP endpoint
        if (endpoint.find("ipc://") == 0) {
            impl_->pubEndpoint = endpoint + ".pub";
        } else if (endpoint.find("tcp://") == 0) {
            // Add 1 to port number for PUB socket
            size_t colonPos = endpoint.rfind(':');
            if (colonPos != std::string::npos) {
                int port = std::stoi(endpoint.substr(colonPos + 1));
                impl_->pubEndpoint = endpoint.substr(0, colonPos + 1) + std::to_string(port + 1);
            }
        }

        impl_->pubSocket = std::make_unique<zmq::socket_t>(impl_->context, zmq::socket_type::pub);
        impl_->pubSocket->bind(impl_->pubEndpoint);

        std::cout << "ZMQ Server initialized on " << endpoint << std::endl;
        std::cout << "ZMQ PUB socket on " << impl_->pubEndpoint << std::endl;
        return true;
    } catch (const zmq::error_t& e) {
        std::cerr << "ZMQ Server init error: " << e.what() << std::endl;
        return false;
    }
}

void ZMQServer::registerHandler(CommandType type, CommandHandler handler) {
    handlers_[type] = std::move(handler);
}

bool ZMQServer::start(bool blocking) {
    if (running_.load()) {
        return false;
    }

    running_.store(true);

    if (blocking) {
        serverLoop();
    } else {
        serverThread_ = std::thread(&ZMQServer::serverLoop, this);
    }
    return true;
}

void ZMQServer::stop() {
    if (running_.load()) {
        running_.store(false);

        // Send a dummy message to unblock recv
        try {
            zmq::context_t tempCtx{1};
            zmq::socket_t tempSocket{tempCtx, zmq::socket_type::req};
            tempSocket.connect(impl_->endpoint);
            tempSocket.send(zmq::buffer(JSON::buildCommand(CommandType::SHUTDOWN, "")),
                            zmq::send_flags::dontwait);
        } catch (...) {}

        if (serverThread_.joinable()) {
            serverThread_.join();
        }
    }
}

void ZMQServer::serverLoop() {
    std::cout << "ZMQ Server listening..." << std::endl;

    while (running_.load()) {
        try {
            zmq::message_t request;
            auto result = impl_->repSocket->recv(request, zmq::recv_flags::none);

            if (!result) {
                continue;
            }

            std::string requestStr(static_cast<char*>(request.data()), request.size());
            std::string response = processMessage(requestStr);

            impl_->repSocket->send(zmq::buffer(response), zmq::send_flags::none);

        } catch (const zmq::error_t& e) {
            if (running_.load()) {
                std::cerr << "ZMQ Server error: " << e.what() << std::endl;
            }
        }
    }

    std::cout << "ZMQ Server stopped" << std::endl;
}

std::string ZMQServer::processMessage(const std::string& message) {
    CommandType type;
    std::string params;

    if (!JSON::parseCommand(message, type, params)) {
        return JSON::buildResponse(ResponseStatus::INVALID_COMMAND, "Failed to parse command");
    }

    // Handle SHUTDOWN specially
    if (type == CommandType::SHUTDOWN) {
        running_.store(false);
        return JSON::buildResponse(ResponseStatus::OK, "Shutting down");
    }

    // Find and execute handler
    auto it = handlers_.find(type);
    if (it == handlers_.end()) {
        return JSON::buildResponse(
            ResponseStatus::INVALID_COMMAND,
            std::string("No handler for command: ") + commandTypeToString(type));
    }

    try {
        CommandResult result = it->second(params);
        return JSON::buildResponse(result.status, result.message, result.data);
    } catch (const std::exception& e) {
        return JSON::buildResponse(ResponseStatus::ERROR,
                                   std::string("Handler exception: ") + e.what());
    }
}

bool ZMQServer::publishStatus(const EngineStatus& status) {
    if (!impl_->pubSocket) {
        return false;
    }

    try {
        std::string statusJson = JSON::buildStatus(status);
        impl_->pubSocket->send(zmq::buffer(statusJson), zmq::send_flags::dontwait);
        return true;
    } catch (const zmq::error_t& e) {
        std::cerr << "ZMQ PUB error: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================
// ZMQClient implementation
// ============================================================

struct ZMQClient::Impl {
    zmq::context_t context{1};
    std::unique_ptr<zmq::socket_t> reqSocket;
    std::unique_ptr<zmq::socket_t> subSocket;
    std::string endpoint;
};

ZMQClient::ZMQClient() : impl_(std::make_unique<Impl>()) {}

ZMQClient::~ZMQClient() {
    disconnect();
    unsubscribeStatus();
}

bool ZMQClient::connect(const std::string& endpoint) {
    try {
        impl_->endpoint = endpoint;
        impl_->reqSocket = std::make_unique<zmq::socket_t>(impl_->context, zmq::socket_type::req);
        impl_->reqSocket->connect(endpoint);
        connected_.store(true);
        std::cout << "ZMQ Client connected to " << endpoint << std::endl;
        return true;
    } catch (const zmq::error_t& e) {
        std::cerr << "ZMQ Client connect error: " << e.what() << std::endl;
        return false;
    }
}

void ZMQClient::disconnect() {
    if (connected_.load()) {
        connected_.store(false);
        impl_->reqSocket.reset();
    }
}

CommandResult ZMQClient::sendCommand(CommandType type, const std::string& params, int timeout_ms) {
    CommandResult result;

    if (!connected_.load()) {
        result.status = ResponseStatus::ERROR;
        result.message = "Not connected";
        return result;
    }

    try {
        // Set receive timeout
        if (timeout_ms > 0) {
            impl_->reqSocket->set(zmq::sockopt::rcvtimeo, timeout_ms);
        }

        // Send command
        std::string cmdJson = JSON::buildCommand(type, params);
        impl_->reqSocket->send(zmq::buffer(cmdJson), zmq::send_flags::none);

        // Receive response
        zmq::message_t response;
        auto recvResult = impl_->reqSocket->recv(response, zmq::recv_flags::none);

        if (!recvResult) {
            result.status = ResponseStatus::ERROR;
            result.message = "Timeout waiting for response";
            return result;
        }

        std::string responseStr(static_cast<char*>(response.data()), response.size());

        ResponseStatus status;
        std::string message, data;
        if (JSON::parseResponse(responseStr, status, message, data)) {
            result.status = status;
            result.message = message;
            result.data = data;
        } else {
            result.status = ResponseStatus::ERROR;
            result.message = "Failed to parse response";
        }

    } catch (const zmq::error_t& e) {
        result.status = ResponseStatus::ERROR;
        result.message = std::string("ZMQ error: ") + e.what();
    }

    return result;
}

// Convenience methods
CommandResult ZMQClient::loadIR(const std::string& path, const std::string& rateFamily) {
    json params;
    params["path"] = path;
    params["rate_family"] = rateFamily;
    return sendCommand(CommandType::LOAD_IR, params.dump());
}

CommandResult ZMQClient::setGain(double gainDb) {
    json params;
    params["gain_db"] = gainDb;
    return sendCommand(CommandType::SET_GAIN, params.dump());
}

CommandResult ZMQClient::softReset() {
    return sendCommand(CommandType::SOFT_RESET);
}

CommandResult ZMQClient::getStatus() {
    return sendCommand(CommandType::GET_STATUS);
}

CommandResult ZMQClient::switchRate(const std::string& rateFamily) {
    json params;
    params["rate_family"] = rateFamily;
    return sendCommand(CommandType::SWITCH_RATE, params.dump());
}

CommandResult ZMQClient::applyEQ(const std::string& eqMagnitudeJson) {
    return sendCommand(CommandType::APPLY_EQ, eqMagnitudeJson);
}

CommandResult ZMQClient::restoreEQ() {
    return sendCommand(CommandType::RESTORE_EQ);
}

CommandResult ZMQClient::shutdown() {
    return sendCommand(CommandType::SHUTDOWN);
}

bool ZMQClient::subscribeStatus(const std::string& pubEndpoint, StatusCallback callback) {
    if (subRunning_.load()) {
        return false;
    }

    try {
        impl_->subSocket = std::make_unique<zmq::socket_t>(impl_->context, zmq::socket_type::sub);
        impl_->subSocket->connect(pubEndpoint);
        impl_->subSocket->set(zmq::sockopt::subscribe, "");  // Subscribe to all messages
        // Set receive timeout to allow periodic check of subRunning_ flag
        impl_->subSocket->set(zmq::sockopt::rcvtimeo, 100);  // 100ms timeout

        subRunning_.store(true);
        subThread_ = std::thread([this, callback]() {
            while (subRunning_.load()) {
                try {
                    zmq::message_t message;
                    auto result = impl_->subSocket->recv(message, zmq::recv_flags::none);

                    if (result) {
                        std::string msgStr(static_cast<char*>(message.data()), message.size());
                        EngineStatus status;
                        if (JSON::parseStatus(msgStr, status)) {
                            callback(status);
                        }
                    }
                    // Timeout (no message) is normal - just loop and check subRunning_
                } catch (const zmq::error_t& e) {
                    // EAGAIN is expected on timeout, don't log it
                    if (subRunning_.load() && e.num() != EAGAIN) {
                        std::cerr << "ZMQ SUB error: " << e.what() << std::endl;
                    }
                }
            }
        });

        return true;
    } catch (const zmq::error_t& e) {
        std::cerr << "ZMQ subscribe error: " << e.what() << std::endl;
        return false;
    }
}

void ZMQClient::unsubscribeStatus() {
    if (subRunning_.load()) {
        // Signal thread to stop first
        subRunning_.store(false);
        // Wait for thread to exit (it will timeout on recv and check subRunning_)
        if (subThread_.joinable()) {
            subThread_.join();
        }
        // Now safe to reset socket after thread has exited
        impl_->subSocket.reset();
    }
}

}  // namespace ZMQComm
