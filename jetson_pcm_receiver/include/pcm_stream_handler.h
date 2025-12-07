#pragma once

#include "pcm_header.h"

#include <atomic>

class AlsaPlayback;
class TcpServer;

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
    bool handleClientForTest(int fd) const;

   private:
    bool receiveHeader(int fd, PcmHeader &header) const;
    bool handleClient(int fd) const;

    AlsaPlayback &playback_;
    TcpServer &server_;
    std::atomic_bool &stopFlag_;
    PcmStreamConfig config_;
};
