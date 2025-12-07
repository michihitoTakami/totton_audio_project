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

TcpServer::TcpServer(int port) : port_(port) {}

TcpServer::~TcpServer() {
    stop();
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

    if (::listen(listenFd_, 1) < 0) {
        logError(std::string("listen: ") + std::strerror(errno));
        ::close(listenFd_);
        listenFd_ = -1;
        return false;
    }

    listening_ = true;
    logInfo("[TcpServer] listening on port " + std::to_string(port()));
    return true;
}

int TcpServer::acceptClient() {
    if (!listening_ || listenFd_ < 0) {
        return -1;
    }

    struct sockaddr_storage addr {};
    socklen_t addrlen = sizeof(addr);
    int fd = ::accept(listenFd_, reinterpret_cast<struct sockaddr *>(&addr), &addrlen);
    if (fd < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return -1;
        }
        logError(std::string("accept: ") + std::strerror(errno));
        return -1;
    }

    if (clientFd_ >= 0) {
        logWarn("[TcpServer] rejecting extra connection; already handling a client");
        ::close(fd);
        return -2;
    }

    setCommonSocketOptions(fd);
    struct timeval tv {};
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        logWarn(std::string("setsockopt(SO_RCVTIMEO): ") + std::strerror(errno));
    }

    clientFd_ = fd;
    logInfo("[TcpServer] client accepted");
    return fd;
}

void TcpServer::closeClient() {
    if (clientFd_ >= 0) {
        ::close(clientFd_);
        clientFd_ = -1;
        logInfo("[TcpServer] client connection closed");
    }
}

void TcpServer::stop() {
    closeClient();
    if (listenFd_ >= 0) {
        ::close(listenFd_);
        listenFd_ = -1;
        listening_ = false;
        boundPort_ = 0;
        logInfo("[TcpServer] stop listening");
    }
}
