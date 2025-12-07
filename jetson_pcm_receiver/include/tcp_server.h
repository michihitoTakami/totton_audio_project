#pragma once

#include <cstdint>
#include <string>

// TCP サーバ待受。単一クライアントのみを受け付ける。
class TcpServer {
   public:
    explicit TcpServer(int port);
    ~TcpServer();

    bool start();
    void stop();
    int acceptClient();
    void closeClient();

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

   private:
    int port_;
    int listenFd_{-1};
    int clientFd_{-1};
    bool listening_{false};
    uint16_t boundPort_{0};

    bool setCommonSocketOptions(int fd);
};
