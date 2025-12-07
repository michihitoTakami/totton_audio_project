#include "PcmHeader.h"
#include "TcpClient.h"

#include <arpa/inet.h>
#include <chrono>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <string>
#include <sys/poll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

bool recvExact(int fd, std::vector<std::uint8_t> &out, std::size_t size, int timeoutMs = 2000) {
    out.resize(size);
    std::size_t offset = 0;
    while (offset < size) {
        pollfd pfd{};
        pfd.fd = fd;
        pfd.events = POLLIN;
        int rv = ::poll(&pfd, 1, timeoutMs);
        if (rv <= 0) {
            return false;
        }
        ssize_t n = ::recv(fd, out.data() + offset, size - offset, 0);
        if (n <= 0) {
            return false;
        }
        offset += static_cast<std::size_t>(n);
    }
    return true;
}

class DummyTcpServer {
   public:
    DummyTcpServer(std::vector<std::size_t> payloadSizes, bool closeAfterFirst)
        : payloadSizes_(std::move(payloadSizes)), closeAfterFirst_(closeAfterFirst) {
        listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        int rc = ::bind(listenFd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
        if (rc != 0) {
            throw std::runtime_error("bind failed");
        }
        ::listen(listenFd_, 4);

        sockaddr_in bound{};
        socklen_t len = sizeof(bound);
        ::getsockname(listenFd_, reinterpret_cast<sockaddr *>(&bound), &len);
        port_ = ntohs(bound.sin_port);

        serverThread_ = std::thread([this]() { this->serve(); });
    }

    ~DummyTcpServer() {
        if (listenFd_ >= 0) {
            ::shutdown(listenFd_, SHUT_RDWR);
            ::close(listenFd_);
        }
        if (serverThread_.joinable()) {
            serverThread_.join();
        }
    }

    std::uint16_t port() const {
        return port_;
    }
    const std::vector<std::string> &headers() const {
        return headers_;
    }
    const std::vector<std::vector<std::uint8_t>> &payloads() const {
        return payloads_;
    }
    bool hadError() const {
        return hadError_;
    }

   private:
    void serve() {
        for (std::size_t i = 0; i < payloadSizes_.size(); ++i) {
            pollfd pfd{};
            pfd.fd = listenFd_;
            pfd.events = POLLIN;
            if (::poll(&pfd, 1, 3000) <= 0) {
                hadError_ = true;
                return;
            }

            int client = ::accept(listenFd_, nullptr, nullptr);
            if (client < 0) {
                hadError_ = true;
                return;
            }

            std::vector<std::uint8_t> header;
            if (!recvExact(client, header, kPcmHeaderSize)) {
                hadError_ = true;
                ::close(client);
                return;
            }
            headers_.emplace_back(reinterpret_cast<const char *>(header.data()), header.size());

            std::vector<std::uint8_t> payload;
            if (!recvExact(client, payload, payloadSizes_[i])) {
                hadError_ = true;
                ::close(client);
                return;
            }
            payloads_.push_back(std::move(payload));

            if (closeAfterFirst_ && i + 1 < payloadSizes_.size()) {
                ::close(client);
                continue;
            }

            ::close(client);
        }
    }

    int listenFd_{-1};
    std::uint16_t port_{0};
    std::vector<std::size_t> payloadSizes_;
    bool closeAfterFirst_{false};
    std::vector<std::string> headers_;
    std::vector<std::vector<std::uint8_t>> payloads_;
    bool hadError_{false};
    std::thread serverThread_;
};

}  // namespace

TEST(TcpClientTest, SendsHeaderAndPayload) {
    PcmHeader header;
    header.sampleRate = 96000;
    header.channels = 2;
    header.format = 2;

    const std::vector<std::uint8_t> payload(256, 0xAB);

    DummyTcpServer server({payload.size()}, false);

    TcpClient client;
    ASSERT_TRUE(client.configure("127.0.0.1", server.port(), header));
    ASSERT_TRUE(client.sendPcmChunk(payload));

    // give server thread a moment to finish
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_FALSE(server.hadError());
    ASSERT_EQ(server.headers().size(), 1u);
    EXPECT_EQ(server.headers()[0], headerToString(header));
    ASSERT_EQ(server.payloads().size(), 1u);
    EXPECT_EQ(server.payloads()[0], payload);
}

TEST(TcpClientTest, ReconnectsAndResendsHeader) {
    PcmHeader header;
    header.sampleRate = 44100;
    header.channels = 2;
    header.format = 1;

    const std::vector<std::uint8_t> firstPayload(64, 0x11);
    const std::vector<std::uint8_t> secondPayload(32, 0x22);

    DummyTcpServer server({firstPayload.size(), secondPayload.size()}, true);

    TcpClient client;
    ASSERT_TRUE(client.configure("127.0.0.1", server.port(), header));
    ASSERT_TRUE(client.sendPcmChunk(firstPayload));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ASSERT_TRUE(client.sendPcmChunk(secondPayload));

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    ASSERT_FALSE(server.hadError());
    ASSERT_EQ(server.headers().size(), 2u);
    EXPECT_EQ(server.headers()[0], headerToString(header));
    EXPECT_EQ(server.headers()[1], headerToString(header));

    ASSERT_EQ(server.payloads().size(), 2u);
    EXPECT_EQ(server.payloads()[0], firstPayload);
    EXPECT_EQ(server.payloads()[1], secondPayload);
}
