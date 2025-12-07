#pragma once

#include "pcm_stream_handler.h"
#include "status_tracker.h"

#include <atomic>
#include <daemon/control/zmq_server.h>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>

class ZmqStatusServer {
   public:
    struct Options {
        bool enabled{true};
        std::string endpoint;
        std::string token;
        int publishIntervalMs{1000};
    };

    ZmqStatusServer(StatusTracker& status, PcmStreamConfig& config, std::mutex& configMutex,
                    std::atomic_bool& stopFlag);
    ~ZmqStatusServer();

    bool start(const Options& options);
    void stop();

    std::string endpoint() const;
    std::string pubEndpoint() const;
    bool running() const {
        return server_ && server_->isRunning();
    }
    bool restartRequested() const {
        return restartRequested_.load();
    }

   private:
    bool checkToken(const daemon_ipc::ZmqRequest& request, std::string& error) const;
    std::string buildError(const daemon_ipc::ZmqRequest& request, const std::string& code,
                           const std::string& message) const;
    std::string handleStatus(const daemon_ipc::ZmqRequest& request);
    std::string handleSetCache(const daemon_ipc::ZmqRequest& request);
    std::string handleRestart(const daemon_ipc::ZmqRequest& request);
    std::string handlePing(const daemon_ipc::ZmqRequest& request);
    nlohmann::json buildStatusJson() const;
    void startPublisher();
    void stopPublisher();
    void publisherLoop();

    StatusTracker& status_;
    PcmStreamConfig& config_;
    std::mutex& configMutex_;
    std::atomic_bool& stopFlag_;
    std::unique_ptr<daemon_ipc::ZmqCommandServer> server_;
    Options options_;
    std::thread publisherThread_;
    std::atomic<bool> publisherRunning_{false};
    std::atomic<bool> restartRequested_{false};
};
