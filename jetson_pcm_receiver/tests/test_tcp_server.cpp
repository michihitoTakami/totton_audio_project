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

    auto res1 = server.acceptClient();
    EXPECT_TRUE(res1.accepted);
    EXPECT_FALSE(server.hasPendingClient());

    int secondClientFd = connectLoopback(server.boundPort());
    ASSERT_GE(secondClientFd, 0);
    auto rejected = server.acceptClient();
    EXPECT_TRUE(rejected.rejected);
    EXPECT_FALSE(server.hasPendingClient());

    ::close(secondClientFd);
    server.closeClient();
    server.stop();
    ::close(clientFd);
}

TEST(TcpServer, TakeoverQueuesPendingClient) {
    TcpServerOptions opts;
    opts.connectionMode = ConnectionMode::Takeover;
    opts.backlog = 4;
    TcpServer server(0, opts);
    ASSERT_TRUE(server.start());
    ASSERT_GT(server.boundPort(), 0);

    int firstClient = connectLoopback(server.boundPort());
    ASSERT_GE(firstClient, 0);
    auto res1 = server.acceptClient();
    ASSERT_TRUE(res1.accepted);
    EXPECT_FALSE(server.hasPendingClient());

    int secondClient = connectLoopback(server.boundPort());
    ASSERT_GE(secondClient, 0);
    auto res2 = server.acceptClient();
    EXPECT_FALSE(res2.accepted);
    EXPECT_TRUE(res2.takeoverPending);
    EXPECT_TRUE(server.hasPendingClient());

    server.closeClient();
    int promoted = server.promotePendingClient();
    EXPECT_GE(promoted, 0);
    EXPECT_FALSE(server.hasPendingClient());

    server.closeClient();
    server.stop();
    ::close(firstClient);
    ::close(secondClient);
}

TEST(TcpServer, PriorityModeRejectsNonPriorityWhilePriorityActive) {
    TcpServerOptions opts;
    opts.connectionMode = ConnectionMode::Priority;
    // No loopback address in allowlist -> all clients are non-priority.
    opts.priorityClients = {"192.0.2.10"};
    TcpServer server(0, opts);
    ASSERT_TRUE(server.start());

    int firstClient = connectLoopback(server.boundPort());
    ASSERT_GE(firstClient, 0);
    auto res1 = server.acceptClient();
    ASSERT_TRUE(res1.accepted);

    int secondClient = connectLoopback(server.boundPort());
    ASSERT_GE(secondClient, 0);
    auto res2 = server.acceptClient();
    EXPECT_TRUE(res2.rejected);
    EXPECT_FALSE(res2.takeoverPending);
    EXPECT_FALSE(server.hasPendingClient());

    server.closeClient();
    server.stop();
    ::close(firstClient);
    ::close(secondClient);
}
