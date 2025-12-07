#pragma once

#include "PcmHeader.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

class TcpClient {
   public:
    TcpClient();
    ~TcpClient();

    bool configure(const std::string &host, std::uint16_t port, const PcmHeader &header);
    bool sendPcmChunk(const std::vector<std::uint8_t> &payload);
    void disconnect();

   private:
    bool ensureConnected();
    bool connectOnce();
    bool sendAll(const std::uint8_t *data, std::size_t size);
    void closeSocket();
    void applySocketOptions(int fd);

    std::string host_;
    std::uint16_t port_{0};
    int sock_{-1};
    std::array<std::uint8_t, kPcmHeaderSize> headerBytes_{};
    bool headerReady_{false};
    std::chrono::milliseconds backoff_{1000};
    const std::chrono::milliseconds backoffMin_{1000};
    const std::chrono::milliseconds backoffMax_{8000};
};
