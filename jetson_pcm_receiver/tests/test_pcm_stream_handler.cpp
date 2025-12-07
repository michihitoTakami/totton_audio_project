#include "alsa_playback.h"
#include "pcm_stream_handler.h"
#include "tcp_server.h"

#include <cstring>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

PcmHeader makeValidHeader() {
    PcmHeader h{};
    std::memcpy(h.magic, "PCMA", 4);
    h.version = 1;
    h.sample_rate = 48000;
    h.channels = 2;
    h.format = 1;
    return h;
}

}  // namespace

TEST(PcmStreamHandler, AcceptsValidHeaderOverSocketPair) {
    AlsaPlayback playback("hw:Loopback,0,0");
    TcpServer server(0);
    PcmStreamHandler handler(playback, server);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));

    EXPECT_TRUE(handler.handleClientForTest(fds[1]));

    ::close(fds[0]);
    ::close(fds[1]);
}

TEST(PcmStreamHandler, RejectsInvalidHeader) {
    AlsaPlayback playback("hw:Loopback,0,0");
    TcpServer server(0);
    PcmStreamHandler handler(playback, server);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    std::memcpy(header.magic, "BAD!", 4);
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));

    EXPECT_FALSE(handler.handleClientForTest(fds[1]));

    ::close(fds[0]);
    ::close(fds[1]);
}

TEST(PcmStreamHandler, DisconnectsOnPartialHeader) {
    AlsaPlayback playback("hw:Loopback,0,0");
    TcpServer server(0);
    PcmStreamHandler handler(playback, server);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    ASSERT_EQ(::send(fds[0], &header, sizeof(header) / 2, 0),
              static_cast<ssize_t>(sizeof(header) / 2));
    ::close(fds[0]);  // simulate client closing early

    EXPECT_FALSE(handler.handleClientForTest(fds[1]));

    ::close(fds[1]);
}
