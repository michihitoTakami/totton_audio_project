#include "pcm_stream_handler.h"

#include "alsa_playback.h"
#include "logging.h"
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

PcmStreamHandler::PcmStreamHandler(AlsaPlayback &playback, TcpServer &server,
                                   std::atomic_bool &stopFlag)
    : playback_(playback), server_(server), stopFlag_(stopFlag) {}

namespace {

std::size_t bytesPerSample(uint16_t format) {
    switch (format) {
    case 1:
        return 2;  // S16_LE
    case 2:
        return 3;  // S24_3LE
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
            logWarn("[PcmStreamHandler] client closed before header complete");
            return false;
        }
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                logWarn("[PcmStreamHandler] header recv timeout");
            } else {
                logError(std::string("recv: ") + std::strerror(errno));
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
        logInfo("[PcmStreamHandler] disconnecting client (header recv failure)");
        return false;
    }

    auto validation = validateHeader(header);
    if (!validation.ok) {
        logWarn(std::string("[PcmStreamHandler] header invalid: ") + validation.reason);
        return false;
    }

    logInfo("[PcmStreamHandler] header ok");
    logInfo("  - rate:     " + std::to_string(header.sample_rate));
    logInfo("  - channels: " + std::to_string(header.channels));
    logInfo("  - format:   " + std::to_string(header.format));
    logInfo("  - device:   " + playback_.device() + " (ALSA)");

    const std::size_t sampleBytes = bytesPerSample(header.format);
    if (sampleBytes == 0) {
        logError("[PcmStreamHandler] unsupported format code: " + std::to_string(header.format));
        return false;
    }

    if (!playback_.open(header.sample_rate, header.channels, header.format)) {
        logError("[PcmStreamHandler] failed to open ALSA playback for received header");
        return false;
    }

    const std::size_t bytesPerFrame = sampleBytes * header.channels;
    constexpr std::size_t RECV_CHUNK_BYTES = 4096;

    std::vector<std::uint8_t> recvBuf(RECV_CHUNK_BYTES);
    std::vector<std::uint8_t> staging;
    staging.reserve(RECV_CHUNK_BYTES * 2);

    bool ok = true;
    while (!stopFlag_.load(std::memory_order_relaxed)) {
        ssize_t n = ::recv(fd, recvBuf.data(), recvBuf.size(), 0);
        if (n == 0) {
            logInfo("[PcmStreamHandler] client closed connection");
            break;
        }
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                logWarn("[PcmStreamHandler] recv timeout; keep waiting");
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
                logError("[PcmStreamHandler] ALSA write failed");
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
        logError("[PcmStreamHandler] failed to start TcpServer");
        return;
    }

    logInfo("[PcmStreamHandler] waiting for clients on port " + std::to_string(server_.port()));
    while (!stopFlag_.load(std::memory_order_relaxed)) {
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
    logInfo("[PcmStreamHandler] stop requested; exit accept loop");
}
