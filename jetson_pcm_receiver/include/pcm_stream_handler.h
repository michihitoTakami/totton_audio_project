#pragma once

#include "pcm_header.h"
#include "status_server.h"

#include <atomic>
#include <mutex>
#include <string>

class AlsaPlayback;
class TcpServer;
class StatusServer;

struct PcmStreamConfig {
    std::size_t ringBufferFrames{0};  // 0で無効
    std::size_t watermarkFrames{0};   // 0なら自動設定
};

// PCM ストリームの受信と再生を橋渡しする雛形。
class PcmStreamHandler {
   public:
    PcmStreamHandler(AlsaPlayback &playback, TcpServer &server, std::atomic_bool &stopFlag,
                     PcmStreamConfig config);

    void run();
    bool handleClientForTest(int fd);
    PcmStatusSnapshot snapshot() const;
    void setStatusServer(StatusServer *server);

   private:
    bool receiveHeader(int fd, PcmHeader &header) const;
    bool handleClient(int fd);

    AlsaPlayback &playback_;
    TcpServer &server_;
    std::atomic_bool &stopFlag_;
    PcmStreamConfig config_;
    StatusServer *statusServer_{nullptr};
    std::atomic_bool connected_{false};
    std::string lastHeaderSummary_;
    mutable std::mutex mutex_;
    std::size_t bufferedFrames_{0};
    std::size_t maxBufferedFrames_{0};
    std::size_t droppedFrames_{0};
};
