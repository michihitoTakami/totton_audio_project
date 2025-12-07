#include "TcpClient.h"

#include "logging.h"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {

constexpr int kDefaultTimeoutMs = 2000;
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

}  // namespace

TcpClient::TcpClient() = default;

TcpClient::~TcpClient() {
    disconnect();
}

bool TcpClient::configure(const std::string &host, std::uint16_t port, const PcmHeader &header) {
    host_ = host;
    port_ = port;
    headerBytes_ = packPcmHeader(header);
    headerReady_ = true;
    return ensureConnected();
}

bool TcpClient::ensureConnected() {
    if (sock_ >= 0) {
        return true;
    }

    while (true) {
        if (connectOnce()) {
            backoff_ = backoffMin_;
            return true;
        }
        std::this_thread::sleep_for(backoff_);
        backoff_ = std::min(backoff_ * 2, backoffMax_);
    }
}

bool TcpClient::connectOnce() {
    if (!headerReady_) {
        return false;
    }

    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        logError(std::string("[TcpClient] socket() failed: ") + std::strerror(errno));
        return false;
    }

    applySocketOptions(fd);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
        logError("[TcpClient] inet_pton failed for host " + host_);
        ::close(fd);
        return false;
    }

    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        logError(std::string("[TcpClient] connect failed: ") + std::strerror(errno));
        ::close(fd);
        return false;
    }

    sock_ = fd;

    if (!sendAll(headerBytes_.data(), headerBytes_.size())) {
        logError("[TcpClient] failed to send header after connect");
        closeSocket();
        return false;
    }

    logInfo("[TcpClient] connected to " + host_ + ":" + std::to_string(port_) + " and sent header");
    return true;
}

bool TcpClient::sendPcmChunk(const std::vector<std::uint8_t> &payload) {
    if (payload.empty()) {
        return true;
    }

    // Detect remote close proactively.
    if (sock_ >= 0) {
        char probe;
        ssize_t rc = ::recv(sock_, &probe, 1, MSG_PEEK | MSG_DONTWAIT);
        if (rc == 0) {
            closeSocket();
        }
    }

    if (!ensureConnected()) {
        return false;
    }

    if (sendAll(payload.data(), payload.size())) {
        return true;
    }

    logWarn("[TcpClient] send failed, reconnecting...");
    closeSocket();

    std::this_thread::sleep_for(backoff_);
    if (!ensureConnected()) {
        return false;
    }

    return sendAll(payload.data(), payload.size());
}

bool TcpClient::sendAll(const std::uint8_t *data, std::size_t size) {
    if (sock_ < 0) {
        return false;
    }

    const std::uint8_t *ptr = data;
    std::size_t remaining = size;
    while (remaining > 0) {
        ssize_t sent = ::send(sock_, ptr, remaining, MSG_NOSIGNAL);
        if (sent > 0) {
            ptr += sent;
            remaining -= static_cast<std::size_t>(sent);
            continue;
        }
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        logError(std::string("[TcpClient] send error: ") + std::strerror(errno));
        return false;
    }
    return true;
}

void TcpClient::disconnect() {
    closeSocket();
}

void TcpClient::closeSocket() {
    if (sock_ >= 0) {
        ::shutdown(sock_, SHUT_RDWR);
        ::close(sock_);
        sock_ = -1;
        logInfo("[TcpClient] socket closed");
    }
}

void TcpClient::applySocketOptions(int fd) {
    int enable = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable));

#ifdef TCP_KEEPIDLE
    int idle = 30;
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
#endif
#ifdef TCP_KEEPINTVL
    int interval = 10;
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof(interval));
#endif
#ifdef TCP_KEEPCNT
    int count = 5;
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &count, sizeof(count));
#endif

    timeval tv{};
    tv.tv_sec = kDefaultTimeoutMs / 1000;
    tv.tv_usec = static_cast<__suseconds_t>((kDefaultTimeoutMs % 1000) * 1000);
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}
