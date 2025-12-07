#include "pcm_stream_handler.h"

#include "alsa_playback.h"
#include "tcp_server.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

PcmStreamHandler::PcmStreamHandler(AlsaPlayback &playback, TcpServer &server)
    : playback_(playback), server_(server) {}

namespace {

std::size_t bytesPerSample(uint16_t format) {
    switch (format) {
    case 1:
        return 2;  // S16_LE
    case 2:
        return 3;  // S24_3LE
    case 3:
        return 4;  // S24_LE (packed in 32bit)
    case 4:
        return 4;  // S32_LE
    default:
        return 0;
    }
}

}  // namespace

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

    const std::size_t sampleBytes = bytesPerSample(header.format);
    if (sampleBytes == 0) {
        std::cerr << "[PcmStreamHandler] unsupported format code: " << header.format << std::endl;
        return false;
    }

    if (!playback_.open(header.sample_rate, header.channels, header.format)) {
        std::cerr << "[PcmStreamHandler] failed to open ALSA playback for received header"
                  << std::endl;
        return false;
    }

    const std::size_t bytesPerFrame = sampleBytes * header.channels;
    constexpr std::size_t RECV_CHUNK_BYTES = 4096;

    std::vector<std::uint8_t> recvBuf(RECV_CHUNK_BYTES);
    std::vector<std::uint8_t> staging;
    staging.reserve(RECV_CHUNK_BYTES * 2);

    bool ok = true;
    while (true) {
        ssize_t n = ::recv(fd, recvBuf.data(), recvBuf.size(), 0);
        if (n == 0) {
            std::cout << "[PcmStreamHandler] client closed connection" << std::endl;
            break;
        }
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::cerr << "[PcmStreamHandler] recv timeout; keep waiting" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            } else {
                std::perror("recv");
            }
            ok = false;
            break;
        }

        staging.insert(staging.end(), recvBuf.begin(), recvBuf.begin() + n);
        const std::size_t framesAvailable = staging.size() / bytesPerFrame;
        const std::size_t bytesAvailable = framesAvailable * bytesPerFrame;

        if (framesAvailable > 0) {
            if (!playback_.write(staging.data(), framesAvailable)) {
                std::cerr << "[PcmStreamHandler] ALSA write failed" << std::endl;
                ok = false;
                break;
            }
            staging.erase(staging.begin(),
                          staging.begin() + static_cast<std::ptrdiff_t>(bytesAvailable));
        }
    }

    playback_.close();
    return ok;
}

bool PcmStreamHandler::handleClientForTest(int fd) const {
    return handleClient(fd);
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
