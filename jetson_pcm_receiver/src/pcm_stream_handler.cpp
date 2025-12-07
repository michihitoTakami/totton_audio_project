#include "pcm_stream_handler.h"

#include "alsa_playback.h"
#include "tcp_server.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

PcmStreamHandler::PcmStreamHandler(AlsaPlayback &playback, TcpServer &server)
    : playback_(playback), server_(server) {}

bool PcmStreamHandler::receiveHeader(int fd, PcmHeader &header) const {
    std::size_t bytesNeeded = sizeof(PcmHeader);
    std::size_t offset = 0;
    auto *raw = reinterpret_cast<std::uint8_t *>(&header);

    while (offset < bytesNeeded) {
        ssize_t n = ::recv(fd, raw + offset, bytesNeeded - offset, 0);
        if (n == 0) {
            std::cout << "[PcmStreamHandler] client closed before header complete" << std::endl;
            return false;
        }
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::cout << "[PcmStreamHandler] header recv timeout" << std::endl;
            } else {
                std::perror("recv");
            }
            return false;
        }
        offset += static_cast<std::size_t>(n);
    }
    return true;
}

bool PcmStreamHandler::handleClient(int fd) const {
    PcmHeader header{};
    if (!receiveHeader(fd, header)) {
        std::cout << "[PcmStreamHandler] disconnecting client (header recv failure)" << std::endl;
        return false;
    }

    auto validation = validateHeader(header);
    if (!validation.ok) {
        std::cout << "[PcmStreamHandler] header invalid: " << validation.reason << std::endl;
        return false;
    }

    std::cout << "[PcmStreamHandler] header ok" << std::endl;
    std::cout << "  - rate:     " << header.sample_rate << std::endl;
    std::cout << "  - channels: " << header.channels << std::endl;
    std::cout << "  - format:   " << header.format << std::endl;
    std::cout << "  - device:   " << playback_.device() << " (ALSA)" << std::endl;
    std::cout << "[PcmStreamHandler] PCMデータ処理は未実装のため接続を閉じます" << std::endl;
    return true;
}

void PcmStreamHandler::run() {
    if (!server_.listening() && !server_.start()) {
        std::cerr << "[PcmStreamHandler] failed to start TcpServer" << std::endl;
        return;
    }

    std::cout << "[PcmStreamHandler] waiting for clients on port " << server_.port() << std::endl;
    while (true) {
        int fd = server_.acceptClient();
        if (fd < 0) {
            if (fd == -2) {
                // 既にクライアントを保持している場合の拒否
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }

        handleClient(fd);
        server_.closeClient();
    }
}

