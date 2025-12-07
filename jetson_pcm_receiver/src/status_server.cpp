#include "status_server.h"

#include "logging.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace {
std::string buildResponse(const PcmStatusSnapshot &snap) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"connected\":" << (snap.connected ? "true" : "false") << ",";
    oss << "\"last_header\":\"" << snap.lastHeaderSummary << "\",";
    oss << "\"buffered_frames\":" << snap.bufferedFrames << ",";
    oss << "\"max_buffered_frames\":" << snap.maxBufferedFrames << ",";
    oss << "\"dropped_frames\":" << snap.droppedFrames << ",";
    oss << "\"xrun_count\":" << snap.xrunCount;
    oss << "}";
    return oss.str();
}
}  // namespace

StatusServer::StatusServer(int port, std::atomic_bool &stopFlag)
    : port_(port), stopFlag_(stopFlag) {}

StatusServer::~StatusServer() {
    stop();
}

void StatusServer::setSnapshot(const PcmStatusSnapshot &snapshot) {
    snapshot_ = snapshot;
}

void StatusServer::start() {
    if (running_.load(std::memory_order_relaxed) || port_ <= 0) {
        return;
    }
    running_.store(true, std::memory_order_relaxed);
    worker_ = std::thread([this]() { serveLoop(); });
}

void StatusServer::stop() {
    running_.store(false, std::memory_order_relaxed);
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    if (worker_.joinable()) {
        worker_.join();
    }
}

void StatusServer::serveLoop() {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        logWarn(std::string("[StatusServer] socket failed: ") + std::strerror(errno));
        return;
    }
    int opt = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // ローカルバインド
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::bind(fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        logWarn(std::string("[StatusServer] bind failed: ") + std::strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return;
    }
    if (::listen(fd_, 4) < 0) {
        logWarn(std::string("[StatusServer] listen failed: ") + std::strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return;
    }

    logInfo("[StatusServer] listening on 127.0.0.1:" + std::to_string(port_));

    while (running_.load(std::memory_order_relaxed) && !stopFlag_.load(std::memory_order_relaxed)) {
        int cfd = ::accept(fd_, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (!running_.load(std::memory_order_relaxed)) {
                break;
            }
            logWarn(std::string("[StatusServer] accept failed: ") + std::strerror(errno));
            continue;
        }

        const std::string body = buildResponse(snapshot_);
        std::ostringstream oss;
        oss << "HTTP/1.1 200 OK\r\n";
        oss << "Content-Type: application/json\r\n";
        oss << "Content-Length: " << body.size() << "\r\n";
        oss << "Connection: close\r\n\r\n";
        oss << body;
        const auto resp = oss.str();
        ::send(cfd, resp.data(), resp.size(), 0);
        ::close(cfd);
    }
}
