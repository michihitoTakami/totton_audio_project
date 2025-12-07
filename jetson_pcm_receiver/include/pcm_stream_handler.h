#pragma once

#include "pcm_header.h"

class AlsaPlayback;
class TcpServer;

// PCM ストリームの受信と再生を橋渡しする雛形。
class PcmStreamHandler {
public:
    PcmStreamHandler(AlsaPlayback &playback, TcpServer &server);

    void run();

private:
    bool receiveHeader(int fd, PcmHeader &header) const;
    bool handleClient(int fd) const;

    AlsaPlayback &playback_;
    TcpServer &server_;
};

