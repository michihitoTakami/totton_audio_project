#include "pcm_stream_handler.h"

#include "alsa_playback.h"
#include "tcp_server.h"

#include <iostream>

PcmStreamHandler::PcmStreamHandler(AlsaPlayback &playback, TcpServer &server)
    : playback_(playback), server_(server) {}

void PcmStreamHandler::run() {
    std::cout << "[PcmStreamHandler] 未実装: PCM 受信ループ" << std::endl;
    std::cout << "  - ALSA device: " << playback_.device() << std::endl;
    std::cout << "  - TCP port:    " << server_.port() << std::endl;
}

