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

    int port() const { return port_; }
    bool listening() const { return listening_; }
    int clientFd() const { return clientFd_; }

private:
    int port_;
    int listenFd_{-1};
    int clientFd_{-1};
    bool listening_{false};

    bool setCommonSocketOptions(int fd);
};

