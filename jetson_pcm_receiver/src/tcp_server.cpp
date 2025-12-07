#include "tcp_server.h"

#include "logging.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

TcpServer::TcpServer(int port, TcpServerOptions options)
    : port_(port), options_(std::move(options)) {}

TcpServer::~TcpServer() {
    stop();
    if (pendingFd_ >= 0) {
        ::close(pendingFd_);
        pendingFd_ = -1;
    }
}

bool TcpServer::setCommonSocketOptions(int fd) {
    int enable = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
        std::perror("setsockopt(SO_REUSEADDR)");
        return false;
    }
    if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable)) < 0) {
        std::perror("setsockopt(SO_KEEPALIVE)");
        // keepaliveが無効でも致命的ではないので続行
    }
    return true;
}

bool TcpServer::isPriorityAddress(const struct sockaddr_storage &addr, socklen_t len) const {
    if (options_.priorityClients.empty()) {
        return false;
    }

    char host[NI_MAXHOST] = {0};
    char service[NI_MAXSERV] = {0};
    if (getnameinfo(reinterpret_cast<const struct sockaddr *>(&addr), len, host, sizeof(host),
                    service, sizeof(service), NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
        return false;
    }
    const std::string addressOnly(host);
    for (const auto &priority : options_.priorityClients) {
        if (priority == addressOnly) {
            return true;
        }
    }
    return false;
}

std::string TcpServer::addressString(const struct sockaddr_storage &addr, socklen_t len) const {
    char host[NI_MAXHOST] = {0};
    char service[NI_MAXSERV] = {0};
    if (getnameinfo(reinterpret_cast<const struct sockaddr *>(&addr), len, host, sizeof(host),
                    service, sizeof(service), NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
        return "<unknown>";
    }
    std::string hostStr(host);
    std::string serviceStr(service);
    return hostStr + ":" + serviceStr;
}

bool TcpServer::start() {
    if (listening_) {
        return true;
    }

    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo *res = nullptr;
    std::string portStr = std::to_string(port_);
    int gai = getaddrinfo(nullptr, portStr.c_str(), &hints, &res);
    if (gai != 0) {
        logError(std::string("[TcpServer] getaddrinfo failed: ") + gai_strerror(gai));
        return false;
    }

    int fd = -1;
    for (auto *p = res; p != nullptr; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (!setCommonSocketOptions(fd)) {
            ::close(fd);
            fd = -1;
            continue;
        }
        if (::bind(fd, p->ai_addr, p->ai_addrlen) == 0) {
            listenFd_ = fd;
            struct sockaddr_storage local {};
            socklen_t len = sizeof(local);
            if (::getsockname(fd, reinterpret_cast<sockaddr *>(&local), &len) == 0) {
                if (local.ss_family == AF_INET) {
                    boundPort_ = ntohs(reinterpret_cast<sockaddr_in *>(&local)->sin_port);
                } else if (local.ss_family == AF_INET6) {
                    boundPort_ = ntohs(reinterpret_cast<sockaddr_in6 *>(&local)->sin6_port);
                }
            }
            break;
        }
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(res);

    if (listenFd_ < 0) {
        logError("[TcpServer] bind failed for port " + std::to_string(port_));
        return false;
    }

    int flags = fcntl(listenFd_, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(listenFd_, F_SETFL, flags | O_NONBLOCK);
    }

    const int backlog = options_.backlog > 0 ? options_.backlog : 4;
    if (::listen(listenFd_, backlog) < 0) {
        logError(std::string("listen: ") + std::strerror(errno));
        ::close(listenFd_);
        listenFd_ = -1;
        return false;
    }

    listening_ = true;
    logInfo("[TcpServer] listening on port " + std::to_string(port()) + " (mode=" +
            toString(options_.connectionMode) + ", backlog=" + std::to_string(backlog) + ")");
    return true;
}

AcceptResult TcpServer::acceptClient() {
    AcceptResult result;
    if (!listening_ || listenFd_ < 0) {
        return result;
    }

    struct sockaddr_storage addr {};
    socklen_t addrlen = sizeof(addr);
    int fd = ::accept(listenFd_, reinterpret_cast<struct sockaddr *>(&addr), &addrlen);
    if (fd < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return result;
        }
        logError(std::string("accept: ") + std::strerror(errno));
        return result;
    }

    setCommonSocketOptions(fd);
    struct timeval tv {};
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        logWarn(std::string("setsockopt(SO_RCVTIMEO): ") + std::strerror(errno));
    }

    const bool isPriority = isPriorityAddress(addr, addrlen);
    const std::string addrStr = addressString(addr, addrlen);
    result.address = addrStr;
    result.isPriority = isPriority;

    if (clientFd_ < 0) {
        clientFd_ = fd;
        clientAddress_ = addrStr;
        clientIsPriority_ = isPriority;
        result.fd = fd;
        result.accepted = true;
        logInfo("[TcpServer] client accepted from " + addrStr + (isPriority ? " (priority)" : ""));
        return result;
    }

    switch (options_.connectionMode) {
    case ConnectionMode::Single:
        logWarn("[TcpServer] rejecting extra connection from " + addrStr +
                " (mode=single, active=" + clientAddress_ + ")");
        ::close(fd);
        result.rejected = true;
        return result;
    case ConnectionMode::Takeover:
        logWarn("[TcpServer] takeover requested by " + addrStr + "; scheduling handover");
        if (pendingFd_ >= 0) {
            ::close(pendingFd_);
        }
        pendingFd_ = fd;
        pendingAddress_ = addrStr;
        pendingIsPriority_ = isPriority;
        result.takeoverPending = true;
        return result;
    case ConnectionMode::Priority:
        if (isPriority && !clientIsPriority_) {
            logInfo("[TcpServer] priority client " + addrStr +
                    " will take over non-priority client " + clientAddress_);
            if (pendingFd_ >= 0) {
                ::close(pendingFd_);
            }
            pendingFd_ = fd;
            pendingAddress_ = addrStr;
            pendingIsPriority_ = isPriority;
            result.takeoverPending = true;
            return result;
        }
        if (!isPriority && clientIsPriority_) {
            logWarn("[TcpServer] rejecting non-priority client " + addrStr +
                    " while priority session active");
            ::close(fd);
            result.rejected = true;
            return result;
        }
        if (isPriority && clientIsPriority_) {
            logWarn("[TcpServer] priority client " + addrStr +
                    " requested takeover; scheduling handover");
            if (pendingFd_ >= 0) {
                ::close(pendingFd_);
            }
            pendingFd_ = fd;
            pendingAddress_ = addrStr;
            pendingIsPriority_ = isPriority;
            result.takeoverPending = true;
            return result;
        }
        logWarn("[TcpServer] rejecting extra non-priority client " + addrStr +
                " (active=" + clientAddress_ + ")");
        ::close(fd);
        result.rejected = true;
        return result;
    }

    ::close(fd);
    return result;
}

void TcpServer::closeClient() {
    if (clientFd_ >= 0) {
        ::close(clientFd_);
        clientFd_ = -1;
        clientAddress_.clear();
        clientIsPriority_ = false;
        logInfo("[TcpServer] client connection closed");
    }
}

int TcpServer::promotePendingClient() {
    if (pendingFd_ < 0) {
        return -1;
    }
    closeClient();
    clientFd_ = pendingFd_;
    clientAddress_ = pendingAddress_;
    clientIsPriority_ = pendingIsPriority_;
    pendingFd_ = -1;
    pendingAddress_.clear();
    pendingIsPriority_ = false;
    logInfo("[TcpServer] switched to pending client " + clientAddress_ +
            (clientIsPriority_ ? " (priority)" : ""));
    return clientFd_;
}

void TcpServer::stop() {
    closeClient();
    if (pendingFd_ >= 0) {
        ::close(pendingFd_);
        pendingFd_ = -1;
        pendingAddress_.clear();
        pendingIsPriority_ = false;
        logInfo("[TcpServer] pending client connection closed");
    }
    if (listenFd_ >= 0) {
        ::close(listenFd_);
        listenFd_ = -1;
        listening_ = false;
        boundPort_ = 0;
        logInfo("[TcpServer] stop listening");
    }
}
