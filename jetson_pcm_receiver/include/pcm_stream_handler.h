#pragma once

#include "connection_mode.h"
#include "pcm_header.h"
#include "status_tracker.h"

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

class AlsaPlayback;
class TcpServer;
class ZmqStatusServer;

struct PcmStreamConfig {
    std::size_t ringBufferFrames{0};  // 0で無効
    std::size_t watermarkFrames{0};   // 0なら自動設定
    int recvTimeoutMs{250};
    int recvTimeoutSleepMs{50};
    int acceptCooldownMs{250};
    int maxConsecutiveTimeouts{3};
    ConnectionMode connectionMode{ConnectionMode::Single};
    std::vector<std::string> priorityClients;
};

// PCM ストリームの受信と再生を橋渡しする雛形。
class PcmStreamHandler {
   public:
    PcmStreamHandler(AlsaPlayback &playback, TcpServer &server, std::atomic_bool &stopFlag,
                     PcmStreamConfig &config, std::mutex *configMutex = nullptr,
                     StatusTracker *status = nullptr, ZmqStatusServer *zmqServer = nullptr);

    void run();
    bool handleClientForTest(int fd);

   private:
    bool receiveHeader(int fd, PcmHeader &header) const;
    bool handleClient(int fd);
    void publishHeaderEvent(const PcmHeader &header);

    AlsaPlayback &playback_;
    TcpServer &server_;
    std::atomic_bool &stopFlag_;
    PcmStreamConfig &config_;
    std::mutex *configMutex_{nullptr};
    StatusTracker *status_{nullptr};
    ZmqStatusServer *zmqServer_{nullptr};
    bool hasLastHeader_{false};
    PcmHeader lastHeader_{};
};
