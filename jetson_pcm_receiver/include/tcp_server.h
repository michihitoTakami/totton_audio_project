#pragma once

#include <cstdint>

// TCP サーバ待受の雛形クラス。実装は後続タスクで追加する。
class TcpServer {
public:
    explicit TcpServer(int port);
    ~TcpServer();

    bool start();
    void stop();

    int port() const { return port_; }

private:
    int port_;
};

