#pragma once

class AlsaPlayback;
class TcpServer;

// PCM ストリームの受信と再生を橋渡しする雛形。
class PcmStreamHandler {
public:
    PcmStreamHandler(AlsaPlayback &playback, TcpServer &server);

    void run();

private:
    AlsaPlayback &playback_;
    TcpServer &server_;
};

