#include "TcpClient.h"

#include <iostream>

TcpClient::TcpClient() = default;

TcpClient::~TcpClient() = default;

bool TcpClient::connect(const std::string &host, std::uint16_t port)
{
    host_ = host;
    port_ = port;
    std::clog << "[TcpClient] connect requested to " << host_ << ":" << port_
              << " (not implemented yet)" << std::endl;
    return false;
}

bool TcpClient::send(const std::vector<std::uint8_t> &payload)
{
    std::clog << "[TcpClient] send requested (" << payload.size()
              << " bytes, not implemented yet)" << std::endl;
    return false;
}

void TcpClient::disconnect()
{
    std::clog << "[TcpClient] disconnect requested (not implemented yet)"
              << std::endl;
}

