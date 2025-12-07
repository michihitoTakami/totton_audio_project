#include "tcp_server.h"

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

int connectLoopback(uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

}  // namespace

TEST(TcpServer, BindsAndAcceptsSingleClient) {
    TcpServer server(0);  // ask OS for an available port
    ASSERT_TRUE(server.start());
    ASSERT_GT(server.boundPort(), 0);

    int clientFd = connectLoopback(server.boundPort());
    ASSERT_GE(clientFd, 0);

    EXPECT_GE(server.acceptClient(), 0);

    int secondClientFd = connectLoopback(server.boundPort());
    ASSERT_GE(secondClientFd, 0);
    int rejected = server.acceptClient();
    EXPECT_EQ(rejected, -2);

    ::close(secondClientFd);
    server.closeClient();
    server.stop();
    ::close(clientFd);
}
