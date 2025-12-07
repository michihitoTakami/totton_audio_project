#pragma once

#include <cstdint>
#include <string>
#include <vector>

class TcpClient {
public:
    TcpClient();
    ~TcpClient();

    bool connect(const std::string &host, std::uint16_t port);
    bool send(const std::vector<std::uint8_t> &payload);
    void disconnect();

private:
    std::string host_;
    std::uint16_t port_{0};
};

