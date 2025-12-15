#pragma once

#include "core/daemon_constants.h"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>

namespace zmq {
class context_t;
class socket_t;
}  // namespace zmq

namespace daemon_ipc {

struct ZmqRequest {
    std::string raw;
    std::optional<nlohmann::json> json;
    std::string command;
    std::string payload;
    bool isJson = false;
    std::string parseError;
};

class ZmqCommandServer {
   public:
    using Handler = std::function<std::string(const ZmqRequest&)>;

    explicit ZmqCommandServer(std::string endpoint = DaemonConstants::ZEROMQ_IPC_PATH,
                              int recvTimeoutMs = 1000);
    ~ZmqCommandServer();

    void registerCommand(const std::string& command, Handler handler);

    bool start();
    void stop();
    bool isRunning() const {
        return running_.load();
    }
    bool hasBindError() const {
        return bindFailed_.load();
    }

    bool publish(const std::string& message);
    const std::string& endpoint() const {
        return endpoint_;
    }
    const std::string& pubEndpoint() const {
        return pubEndpoint_;
    }

   private:
    ZmqRequest buildRequest(const std::string& raw) const;
    std::string dispatchRequest(const ZmqRequest& request);
    std::string buildErrorResponse(const ZmqRequest& request, const std::string& code,
                                   const std::string& message) const;
    void serverLoop();
    void cleanupSockets();
    void cleanupIpcPath(const std::string& endpoint) const;
    static std::string derivePubEndpoint(const std::string& endpoint);

    std::string endpoint_;
    std::string pubEndpoint_;
    int recvTimeoutMs_;
    std::unique_ptr<zmq::context_t> context_;
    std::unique_ptr<zmq::socket_t> repSocket_;
    std::unique_ptr<zmq::socket_t> pubSocket_;
    std::thread serverThread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> bindFailed_{false};
    std::map<std::string, Handler> handlers_;
    mutable std::mutex pubMutex_;
};

}  // namespace daemon_ipc
