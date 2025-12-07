#pragma once

#include "connection_mode.h"

#include <cstdint>
#include <netdb.h>
#include <string>
#include <sys/socket.h>
#include <vector>

struct TcpServerOptions {
    ConnectionMode connectionMode{ConnectionMode::Single};
    int backlog{4};
    std::vector<std::string> priorityClients;
};

struct AcceptResult {
    int fd{-1};
    bool accepted{false};
    bool rejected{false};
    bool takeoverPending{false};
    bool isPriority{false};
    std::string address;
};

// TCP サーバ待受。複数クライアントの優先度制御に対応。
class TcpServer {
   public:
    explicit TcpServer(int port, TcpServerOptions options = {});
    ~TcpServer();

    bool start();
    void stop();
    AcceptResult acceptClient();
    void closeClient();
    bool hasActiveClient() const {
        return clientFd_ >= 0;
    }
    bool hasPendingClient() const {
        return pendingFd_ >= 0;
    }
    int promotePendingClient();

    int port() const {
        return boundPort_ > 0 ? boundPort_ : port_;
    }
    uint16_t boundPort() const {
        return boundPort_;
    }
    bool listening() const {
        return listening_;
    }
    int clientFd() const {
        return clientFd_;
    }
    std::string clientAddress() const {
        return clientAddress_;
    }
    bool clientIsPriority() const {
        return clientIsPriority_;
    }

   private:
    int port_;
    TcpServerOptions options_;
    int listenFd_{-1};
    int clientFd_{-1};
    int pendingFd_{-1};
    bool listening_{false};
    uint16_t boundPort_{0};
    std::string clientAddress_;
    std::string pendingAddress_;
    bool clientIsPriority_{false};
    bool pendingIsPriority_{false};

    bool isPriorityAddress(const struct sockaddr_storage &addr, socklen_t len) const;
    std::string addressString(const struct sockaddr_storage &addr, socklen_t len) const;
    bool setCommonSocketOptions(int fd);
};
