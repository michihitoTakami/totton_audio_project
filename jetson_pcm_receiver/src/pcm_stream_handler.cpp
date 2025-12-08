#include "pcm_stream_handler.h"

#include "alsa_playback.h"
#include "logging.h"
#include "tcp_server.h"
#include "zmq_status_server.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <vector>

PcmStreamHandler::PcmStreamHandler(AlsaPlayback &playback, TcpServer &server,
                                   std::atomic_bool &stopFlag, PcmStreamConfig &config,
                                   std::mutex *configMutex, StatusTracker *status,
                                   ZmqStatusServer *zmqServer)
    : playback_(playback),
      server_(server),
      stopFlag_(stopFlag),
      config_(config),
      configMutex_(configMutex),
      status_(status),
      zmqServer_(zmqServer) {}

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

    while (offset < bytesNeeded && !stopFlag_.load(std::memory_order_relaxed)) {
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
    if (stopFlag_.load(std::memory_order_relaxed)) {
        logWarn("[PcmStreamHandler] stop requested during header receive");
        return false;
    }
    return true;
}

bool PcmStreamHandler::handleClient(int fd) {
    PcmStreamConfig configSnapshot;
    if (configMutex_) {
        std::lock_guard<std::mutex> lock(*configMutex_);
        configSnapshot = config_;
    } else {
        configSnapshot = config_;
    }

    const int recvTimeoutMs = configSnapshot.recvTimeoutMs > 0 ? configSnapshot.recvTimeoutMs : 250;
    const int recvTimeoutSleepMs =
        configSnapshot.recvTimeoutSleepMs > 0 ? configSnapshot.recvTimeoutSleepMs : 50;
    const int acceptCooldownMs =
        configSnapshot.acceptCooldownMs > 0 ? configSnapshot.acceptCooldownMs : 250;
    const int maxConsecutiveTimeouts =
        configSnapshot.maxConsecutiveTimeouts > 0 ? configSnapshot.maxConsecutiveTimeouts : 3;
    const ConnectionMode connectionMode = configSnapshot.connectionMode;

    if (recvTimeoutMs > 0) {
        struct timeval tv {};
        tv.tv_sec = recvTimeoutMs / 1000;
        tv.tv_usec = (recvTimeoutMs % 1000) * 1000;
        if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
            logWarn(std::string("[PcmStreamHandler] setsockopt(SO_RCVTIMEO) failed: ") +
                    std::strerror(errno));
        }
    }

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

    if (status_) {
        status_->clearDisconnectReason();
        status_->setHeader(header);
    }
    publishHeaderEvent(header);

    const std::size_t bytesPerFrame = sampleBytes * header.channels;
    constexpr std::size_t RECV_CHUNK_BYTES = 4096;
    constexpr std::size_t HEADER_BYTES = sizeof(PcmHeader);

    std::vector<std::uint8_t> recvBuf(RECV_CHUNK_BYTES);
    std::vector<std::uint8_t> detectionBuf;
    detectionBuf.reserve(RECV_CHUNK_BYTES + HEADER_BYTES);

    const bool useRing = configSnapshot.ringBufferFrames > 0;
    const std::size_t capacityBytes = configSnapshot.ringBufferFrames * bytesPerFrame;
    std::size_t watermarkFrames = configSnapshot.watermarkFrames > 0
                                      ? configSnapshot.watermarkFrames
                                      : configSnapshot.ringBufferFrames * 3 / 4;
    if (!useRing) {
        watermarkFrames = 0;
    }
    std::vector<std::uint8_t> ringBuf;
    if (useRing) {
        ringBuf.reserve(capacityBytes);
        logInfo("[PcmStreamHandler] ring buffer enabled: frames=" +
                std::to_string(configSnapshot.ringBufferFrames) +
                " watermark=" + std::to_string(watermarkFrames));
    }

    std::vector<std::uint8_t> staging;
    staging.reserve(RECV_CHUNK_BYTES * 2);

    std::size_t maxBufferedFrames = 0;
    std::size_t droppedFrames = 0;
    bool watermarkLogged = false;

    if (status_) {
        status_->updateRingConfig(configSnapshot.ringBufferFrames, watermarkFrames);
        status_->updateRingBuffer(0, maxBufferedFrames, droppedFrames);
        status_->setStreaming(true);
    }

    auto enqueueData = [&](const std::uint8_t *data, std::size_t bytes) {
        if (!useRing) {
            staging.insert(staging.end(), data, data + bytes);
            if (status_) {
                status_->updateRingBuffer(staging.size() / bytesPerFrame, maxBufferedFrames,
                                          droppedFrames);
            }
            return;
        }
        ringBuf.insert(ringBuf.end(), data, data + bytes);
        if (ringBuf.size() > capacityBytes) {
            const std::size_t overBytes = ringBuf.size() - capacityBytes;
            const std::size_t overFrames = overBytes / bytesPerFrame;
            droppedFrames += overFrames;
            ringBuf.erase(ringBuf.begin(), ringBuf.begin() + static_cast<long>(overBytes));
            logWarn("[PcmStreamHandler] ring buffer overflow; dropped frames=" +
                    std::to_string(overFrames));
        }
        const std::size_t bufferedFrames = ringBuf.size() / bytesPerFrame;
        if (bufferedFrames > maxBufferedFrames) {
            maxBufferedFrames = bufferedFrames;
        }
        if (!watermarkLogged && bufferedFrames >= watermarkFrames) {
            logWarn("[PcmStreamHandler] ring buffer watermark reached: " +
                    std::to_string(bufferedFrames) + " frames");
            watermarkLogged = true;
        }
        if (status_) {
            status_->updateRingBuffer(ringBuf.size() / bytesPerFrame, maxBufferedFrames,
                                      droppedFrames);
        }
    };

    auto tryWrite = [&](std::vector<std::uint8_t> &buf) -> bool {
        const std::size_t framesAvailable = buf.size() / bytesPerFrame;
        if (framesAvailable == 0) {
            return true;
        }
        if (!playback_.write(buf.data(), framesAvailable)) {
            const bool xrunStorm = playback_.wasXrunStorm();
            logError(xrunStorm ? "[PcmStreamHandler] XRUN storm detected; disconnecting stream"
                               : "[PcmStreamHandler] ALSA write failed");
            if (status_) {
                status_->setDisconnectReason(xrunStorm ? "xrun_storm" : "playback_error");
            }
            return false;
        }
        const std::size_t bytesUsed = framesAvailable * bytesPerFrame;
        buf.erase(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(bytesUsed));
        if (status_) {
            status_->updateRingBuffer(buf.size() / bytesPerFrame, maxBufferedFrames, droppedFrames);
        }
        return true;
    };

    bool ok = true;
    bool takeoverPending = false;
    int consecutiveTimeouts = 0;
    bool formatChangeDetected = false;
    constexpr std::size_t KEEP_BYTES_FOR_DETECTION = HEADER_BYTES - 1;
    while (!stopFlag_.load(std::memory_order_relaxed)) {
        if (connectionMode != ConnectionMode::Single) {
            auto res = server_.acceptClient();
            if (res.takeoverPending || server_.hasPendingClient()) {
                logWarn("[PcmStreamHandler] takeover requested; switching client");
                ok = false;
                takeoverPending = true;
                break;
            }
            if (res.rejected) {
                logWarn("[PcmStreamHandler] additional client rejected while streaming");
            }
        }

        ssize_t n = ::recv(fd, recvBuf.data(), recvBuf.size(), 0);
        if (n == 0) {
            logInfo("[PcmStreamHandler] client closed connection");
            break;
        }
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                ++consecutiveTimeouts;
                if (consecutiveTimeouts >= maxConsecutiveTimeouts) {
                    logWarn(
                        "[PcmStreamHandler] recv timeout repeated; disconnecting with cooldown");
                    ok = false;
                    break;
                }
                logWarn("[PcmStreamHandler] recv timeout; keep waiting");
                std::this_thread::sleep_for(std::chrono::milliseconds(recvTimeoutSleepMs));
                continue;
            } else {
                std::perror("recv");
            }
            ok = false;
            break;
        }

        consecutiveTimeouts = 0;

        const std::size_t bytesReceived = static_cast<std::size_t>(n);

        detectionBuf.insert(detectionBuf.end(), recvBuf.begin(), recvBuf.begin() + bytesReceived);

        std::size_t searchPos = 0;
        while (searchPos + HEADER_BYTES <= detectionBuf.size()) {
            if (std::memcmp(detectionBuf.data() + searchPos, "PCMA", 4) != 0) {
                ++searchPos;
                continue;
            }
            PcmHeader candidate{};
            std::memcpy(&candidate, detectionBuf.data() + searchPos, HEADER_BYTES);
            auto candidateValidation = validateHeader(candidate);
            if (!candidateValidation.ok) {
                ++searchPos;
                continue;
            }

            const bool differs = candidate.sample_rate != header.sample_rate ||
                                 candidate.channels != header.channels ||
                                 candidate.format != header.format;
            if (differs) {
                logWarn(
                    "[PcmStreamHandler] detected new header mid-stream; disconnecting to "
                    "renegotiate format");
                logWarn("  - previous: rate=" + std::to_string(header.sample_rate) +
                        " format=" + std::to_string(header.format));
                logWarn("  - incoming: rate=" + std::to_string(candidate.sample_rate) +
                        " format=" + std::to_string(candidate.format));
                if (status_) {
                    status_->setDisconnectReason("format_changed");
                }
                formatChangeDetected = true;
                break;
            }
            searchPos += HEADER_BYTES;
        }

        if (formatChangeDetected) {
            ok = false;
            break;
        }

        if (!formatChangeDetected) {
            if (detectionBuf.size() > KEEP_BYTES_FOR_DETECTION) {
                const std::size_t toRemove = detectionBuf.size() - KEEP_BYTES_FOR_DETECTION;
                detectionBuf.erase(detectionBuf.begin(),
                                   detectionBuf.begin() + static_cast<std::ptrdiff_t>(toRemove));
            }
            enqueueData(recvBuf.data(), bytesReceived);
        }

        if (useRing) {
            if (!tryWrite(ringBuf)) {
                if (status_) {
                    status_->setDisconnectReason("playback_error");
                }
                ok = false;
                break;
            }
        } else {
            if (!tryWrite(staging)) {
                if (status_) {
                    status_->setDisconnectReason("playback_error");
                }
                ok = false;
                break;
            }
        }
    }

    playback_.close();
    if (useRing) {
        logInfo("[PcmStreamHandler] ring stats: max_buffered_frames=" +
                std::to_string(maxBufferedFrames) +
                " dropped_frames=" + std::to_string(droppedFrames));
    }
    if (!ok && status_ && playback_.wasXrunStorm()) {
        status_->setDisconnectReason("xrun_storm");
    }
    if (status_) {
        status_->updateRingBuffer(0, maxBufferedFrames, droppedFrames);
        status_->setStreaming(false);
    }
    if (!stopFlag_.load(std::memory_order_relaxed) && !takeoverPending) {
        std::this_thread::sleep_for(std::chrono::milliseconds(acceptCooldownMs));
    }
    return ok;
}

bool PcmStreamHandler::handleClientForTest(int fd) {
    return handleClient(fd);
}

void PcmStreamHandler::publishHeaderEvent(const PcmHeader &header) {
    if (!zmqServer_) {
        return;
    }
    std::optional<PcmHeader> previous{};
    if (hasLastHeader_) {
        previous = lastHeader_;
    }
    if (zmqServer_->publishHeaderChange(header, previous)) {
        lastHeader_ = header;
        hasLastHeader_ = true;
    }
}

void PcmStreamHandler::run() {
    if (!server_.listening() && !server_.start()) {
        logError("[PcmStreamHandler] failed to start TcpServer");
        return;
    }
    if (status_) {
        status_->setListening(server_.port());
    }

    logInfo("[PcmStreamHandler] waiting for clients on port " + std::to_string(server_.port()));
    while (!stopFlag_.load(std::memory_order_relaxed)) {
        if (!server_.hasActiveClient()) {
            auto res = server_.acceptClient();
            if (!res.accepted) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
        }

        if (status_) {
            status_->setClientConnected(true);
        }
        handleClient(server_.clientFd());
        server_.closeClient();
        if (status_) {
            status_->setClientConnected(false);
        }
        if (server_.hasPendingClient()) {
            server_.promotePendingClient();
            if (status_) {
                status_->setClientConnected(true);
            }
            continue;
        }
    }
    logInfo("[PcmStreamHandler] stop requested; exit accept loop");
    if (status_) {
        status_->clearListening();
        status_->setStreaming(false);
        status_->setClientConnected(false);
    }
}
