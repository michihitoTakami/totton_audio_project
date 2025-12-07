#include "tcp_server.h"

#include <iostream>

TcpServer::TcpServer(int port) : port_(port) {}

TcpServer::~TcpServer() = default;

bool TcpServer::start() {
    std::cout << "[TcpServer] 未実装: listen on port " << port_ << std::endl;
    return false;
}

void TcpServer::stop() {
    std::cout << "[TcpServer] 未実装: stop server" << std::endl;
}

