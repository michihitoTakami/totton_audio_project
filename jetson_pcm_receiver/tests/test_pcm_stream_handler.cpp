#include "alsa_playback.h"
#include "audio/pcm_format_set.h"
#include "pcm_stream_handler.h"
#include "status_tracker.h"
#include "tcp_server.h"

#include <arpa/inet.h>
#include <atomic>
#include <cstring>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

class FakeAlsaPlayback : public AlsaPlayback {
   public:
    FakeAlsaPlayback() : AlsaPlayback("fake") {}

    bool open(uint32_t sampleRate, uint16_t channels, uint16_t format) override {
        openCalled = true;
        lastRate = sampleRate;
        lastChannels = channels;
        lastFormat = format;
        xrunStorm = false;

        if (shouldFailOpen) {
            return false;
        }
        if (!isSupportedRate(sampleRate) || channels != 2) {
            return false;
        }
        if (format != 1 && format != 2 && format != 4) {
            return false;
        }
        opened = true;
        return true;
    }

    bool write(const void* /*data*/, std::size_t frames) override {
        writeCalled = true;
        if (!opened) {
            return false;
        }
        if (failWriteAsXrun) {
            xrunStorm = true;
            return false;
        }
        if (shouldFailWrite) {
            return false;
        }
        totalFramesWritten += frames;
        return true;
    }

    void close() override {
        closeCalled = true;
        opened = false;
    }

    bool wasXrunStorm() const override {
        return xrunStorm;
    }

    static bool isSupportedRate(uint32_t rate) {
        return PcmFormatSet::isAllowedSampleRate(rate);
    }

    bool shouldFailOpen{false};
    bool shouldFailWrite{false};
    bool failWriteAsXrun{false};
    bool openCalled{false};
    bool writeCalled{false};
    bool closeCalled{false};
    bool opened{false};
    bool xrunStorm{false};
    std::size_t totalFramesWritten{0};
    uint32_t lastRate{0};
    uint16_t lastChannels{0};
    uint16_t lastFormat{0};
};

PcmHeader makeValidHeader() {
    PcmHeader h{};
    std::memcpy(h.magic, "PCMA", 4);
    h.version = 1;
    h.sample_rate = 48000;
    h.channels = 2;
    h.format = 1;
    return h;
}

int connectLoopback(uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

}  // namespace

TEST(PcmStreamHandler, DetectsMidStreamFormatChangeAndDisconnects) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    StatusTracker status;
    PcmStreamHandler handler(playback, server, stop, cfg, nullptr, &status);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto firstHeader = makeValidHeader();
    ASSERT_EQ(::send(fds[0], &firstHeader, sizeof(firstHeader), 0), sizeof(firstHeader));

    std::vector<std::uint8_t> pcm(32, 0);
    ASSERT_EQ(::send(fds[0], pcm.data(), pcm.size(), 0), static_cast<ssize_t>(pcm.size()));

    PcmHeader newHeader = makeValidHeader();
    newHeader.sample_rate = 96000;
    newHeader.format = 4;
    ASSERT_EQ(::send(fds[0], &newHeader, sizeof(newHeader), 0), sizeof(newHeader));

    std::vector<std::uint8_t> pcm2(64, 0);
    ASSERT_EQ(::send(fds[0], pcm2.data(), pcm2.size(), 0), static_cast<ssize_t>(pcm2.size()));
    ::close(fds[0]);

    EXPECT_FALSE(handler.handleClientForTest(fds[1]));
    auto snap = status.snapshot();
    EXPECT_EQ(snap.disconnectReason, "format_changed");
    EXPECT_TRUE(playback.closeCalled);

    ::close(fds[1]);
}

TEST(PcmStreamHandler, RejectsUnsupportedHeaderAndReportsStatus) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    StatusTracker status;
    PcmStreamHandler handler(playback, server, stop, cfg, nullptr, &status);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    header.sample_rate = 32000;  // not in shared allowlist
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));
    ::shutdown(fds[0], SHUT_WR);

    EXPECT_FALSE(handler.handleClientForTest(fds[1]));
    auto snap = status.snapshot();
    EXPECT_EQ(snap.disconnectReason, "sample_rate unsupported");

    ::close(fds[0]);
    ::close(fds[1]);
}

TEST(PcmStreamHandler, ClearsDisconnectReasonAfterRenegotiation) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    StatusTracker status;
    PcmStreamHandler handler(playback, server, stop, cfg, nullptr, &status);

    int first[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, first), 0);

    auto header = makeValidHeader();
    ASSERT_EQ(::send(first[0], &header, sizeof(header), 0), sizeof(header));

    // Send mid-stream header with different format to trigger disconnect_reason
    PcmHeader midHeader = makeValidHeader();
    midHeader.format = 4;
    ASSERT_EQ(::send(first[0], &midHeader, sizeof(midHeader), 0), sizeof(midHeader));
    ::close(first[0]);

    EXPECT_FALSE(handler.handleClientForTest(first[1]));
    EXPECT_EQ(status.snapshot().disconnectReason, "format_changed");
    ::close(first[1]);

    // Reset playback flags for the next negotiation
    playback.openCalled = false;
    playback.writeCalled = false;
    playback.closeCalled = false;
    playback.totalFramesWritten = 0;

    int second[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, second), 0);

    PcmHeader newHeader = makeValidHeader();
    newHeader.sample_rate = 96000;
    newHeader.format = 4;  // S32_LE (8 bytes per frame for stereo)
    ASSERT_EQ(::send(second[0], &newHeader, sizeof(newHeader), 0), sizeof(newHeader));

    std::vector<std::uint8_t> pcm(64, 0);  // 8 frames for S32_LE stereo
    ASSERT_EQ(::send(second[0], pcm.data(), pcm.size(), 0), static_cast<ssize_t>(pcm.size()));
    ::close(second[0]);

    EXPECT_TRUE(handler.handleClientForTest(second[1]));
    auto snap = status.snapshot();
    EXPECT_EQ(snap.disconnectReason, "client_closed");
    EXPECT_EQ(snap.header.header.sample_rate, newHeader.sample_rate);
    EXPECT_TRUE(playback.openCalled);
    EXPECT_TRUE(playback.writeCalled);
    EXPECT_TRUE(playback.closeCalled);
    EXPECT_EQ(playback.totalFramesWritten, 8u);

    ::close(second[1]);
}

TEST(PcmStreamHandler, AcceptsValidHeaderAndWritesFrames) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    PcmStreamHandler handler(playback, server, stop, cfg);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));

    // 8 frames of silence (S16_LE, 2ch = 4 bytes/frame -> 32 bytes)
    std::vector<std::uint8_t> pcm(32, 0);
    ASSERT_EQ(::send(fds[0], pcm.data(), pcm.size(), 0), static_cast<ssize_t>(pcm.size()));
    ::close(fds[0]);

    EXPECT_TRUE(handler.handleClientForTest(fds[1]));
    EXPECT_TRUE(playback.openCalled);
    EXPECT_TRUE(playback.writeCalled);
    EXPECT_TRUE(playback.closeCalled);
    EXPECT_EQ(playback.totalFramesWritten, 8u);

    ::close(fds[0]);
    ::close(fds[1]);
}

TEST(PcmStreamHandler, RingBufferDropsOldFramesWhenOverflow) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    cfg.ringBufferFrames = 4;  // very small buffer
    cfg.watermarkFrames = 3;
    PcmStreamHandler handler(playback, server, stop, cfg);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));

    // 8 frames (should drop 4 when buffer is 4 frames)
    std::vector<std::uint8_t> pcm(32, 0);  // S16_LE stereo -> 4 bytes/frame
    ASSERT_EQ(::send(fds[0], pcm.data(), pcm.size(), 0), static_cast<ssize_t>(pcm.size()));
    ::close(fds[0]);

    EXPECT_TRUE(handler.handleClientForTest(fds[1]));
    EXPECT_TRUE(playback.openCalled);
    EXPECT_TRUE(playback.writeCalled);
    EXPECT_EQ(playback.totalFramesWritten, 4u);  // only capacity frames survive

    ::close(fds[1]);
}

TEST(PcmStreamHandler, AcceptsS24AndHighRate) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    PcmStreamHandler handler(playback, server, stop, cfg);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    PcmHeader header = makeValidHeader();
    header.sample_rate = 176400;  // 44.1k * 4
    header.format = 2;            // S24_3LE
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));

    // 4 frames, 3 bytes/sample * 2ch = 24 bytes
    std::vector<std::uint8_t> pcm(24, 0);
    ASSERT_EQ(::send(fds[0], pcm.data(), pcm.size(), 0), static_cast<ssize_t>(pcm.size()));
    ::close(fds[0]);

    EXPECT_TRUE(handler.handleClientForTest(fds[1]));
    EXPECT_EQ(playback.lastRate, 176400u);
    EXPECT_EQ(playback.lastFormat, 2u);
    EXPECT_EQ(playback.totalFramesWritten, 4u);

    ::close(fds[1]);
}

TEST(PcmStreamHandler, DisconnectsOnFrameBytesMismatch) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    StatusTracker status;
    PcmStreamHandler handler(playback, server, stop, cfg, nullptr, &status);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    header.format = 2;  // expect 3 bytes/sample (6 bytes per frame for stereo)
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));

    std::vector<std::uint8_t> pcm(4096, 0);  // not divisible by 6 bytes/frame
    ASSERT_EQ(::send(fds[0], pcm.data(), pcm.size(), 0), static_cast<ssize_t>(pcm.size()));
    ::close(fds[0]);

    EXPECT_FALSE(handler.handleClientForTest(fds[1]));
    auto snap = status.snapshot();
    EXPECT_EQ(snap.disconnectReason, "frame_bytes_mismatch");
    EXPECT_TRUE(playback.openCalled);
    EXPECT_FALSE(playback.writeCalled);
    EXPECT_TRUE(playback.closeCalled);
    EXPECT_FALSE(snap.header.present);

    ::close(fds[1]);
}

TEST(PcmStreamHandler, RejectsUnsupportedRate) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    PcmStreamHandler handler(playback, server, stop, cfg);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    header.sample_rate = 50000;  // not in supported multiples
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));
    ::close(fds[0]);

    EXPECT_FALSE(handler.handleClientForTest(fds[1]));
    EXPECT_FALSE(playback.openCalled);
    EXPECT_FALSE(playback.writeCalled);

    ::close(fds[1]);
}

TEST(PcmStreamHandler, RejectsWhenPlaybackOpenFailsForFormat) {
    FakeAlsaPlayback playback;
    playback.shouldFailOpen = true;  // force open failure after validation passes
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    PcmStreamHandler handler(playback, server, stop, cfg);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();  // valid header; open will be forced to fail
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));
    ::close(fds[0]);

    EXPECT_FALSE(handler.handleClientForTest(fds[1]));
    EXPECT_TRUE(playback.openCalled);
    EXPECT_FALSE(playback.writeCalled);
    EXPECT_FALSE(playback.closeCalled);

    ::close(fds[1]);
}

TEST(PcmStreamHandler, RejectsInvalidHeader) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    PcmStreamHandler handler(playback, server, stop, cfg);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    std::memcpy(header.magic, "BAD!", 4);
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));

    EXPECT_FALSE(handler.handleClientForTest(fds[1]));
    EXPECT_FALSE(playback.openCalled);
    EXPECT_FALSE(playback.closeCalled);

    ::close(fds[0]);
    ::close(fds[1]);
}

TEST(PcmStreamHandler, DisconnectsAfterRepeatedTimeouts) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    cfg.recvTimeoutMs = 5;
    cfg.recvTimeoutSleepMs = 1;
    cfg.acceptCooldownMs = 1;
    cfg.maxConsecutiveTimeouts = 3;
    StatusTracker status;
    PcmStreamHandler handler(playback, server, stop, cfg, nullptr, &status);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));

    EXPECT_FALSE(handler.handleClientForTest(fds[1]));
    EXPECT_EQ(status.snapshot().disconnectReason, "recv_timeout");
    EXPECT_TRUE(playback.openCalled);
    EXPECT_TRUE(playback.closeCalled);
    EXPECT_FALSE(playback.writeCalled);

    ::close(fds[0]);
    ::close(fds[1]);
}

TEST(PcmStreamHandler, SignalsClientCloseAndAllowsReconnect) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    StatusTracker status;
    PcmStreamHandler handler(playback, server, stop, cfg, nullptr, &status);

    int first[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, first), 0);

    auto header = makeValidHeader();
    ASSERT_EQ(::send(first[0], &header, sizeof(header), 0), sizeof(header));
    std::vector<std::uint8_t> pcm(32, 0);
    ASSERT_EQ(::send(first[0], pcm.data(), pcm.size(), 0), static_cast<ssize_t>(pcm.size()));
    ::close(first[0]);

    EXPECT_TRUE(handler.handleClientForTest(first[1]));
    auto snapAfterClose = status.snapshot();
    EXPECT_EQ(snapAfterClose.disconnectReason, "client_closed");
    EXPECT_FALSE(snapAfterClose.streaming);
    EXPECT_FALSE(snapAfterClose.header.present);
    ::close(first[1]);

    playback.openCalled = false;
    playback.writeCalled = false;
    playback.closeCalled = false;
    playback.totalFramesWritten = 0;

    int second[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, second), 0);

    PcmHeader newHeader = makeValidHeader();
    newHeader.sample_rate = 96000;
    newHeader.format = 4;
    ASSERT_EQ(::send(second[0], &newHeader, sizeof(newHeader), 0), sizeof(newHeader));
    std::vector<std::uint8_t> pcm2(64, 0);
    ASSERT_EQ(::send(second[0], pcm2.data(), pcm2.size(), 0), static_cast<ssize_t>(pcm2.size()));
    ::close(second[0]);

    EXPECT_TRUE(handler.handleClientForTest(second[1]));
    auto snapAfterReconnect = status.snapshot();
    EXPECT_EQ(snapAfterReconnect.disconnectReason, "client_closed");
    EXPECT_FALSE(snapAfterReconnect.header.present);
    EXPECT_EQ(snapAfterReconnect.header.header.sample_rate, newHeader.sample_rate);
    EXPECT_TRUE(playback.openCalled);
    EXPECT_TRUE(playback.writeCalled);
    EXPECT_TRUE(playback.closeCalled);

    ::close(second[1]);
}

TEST(PcmStreamHandler, SetsDisconnectReasonOnXrunStorm) {
    FakeAlsaPlayback playback;
    playback.failWriteAsXrun = true;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    StatusTracker status;
    PcmStreamHandler handler(playback, server, stop, cfg, nullptr, &status);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    ASSERT_EQ(::send(fds[0], &header, sizeof(header), 0), sizeof(header));

    std::vector<std::uint8_t> pcm(32, 0);  // enough to trigger write
    ASSERT_EQ(::send(fds[0], pcm.data(), pcm.size(), 0), static_cast<ssize_t>(pcm.size()));
    ::close(fds[0]);

    EXPECT_FALSE(handler.handleClientForTest(fds[1]));
    auto snap = status.snapshot();
    EXPECT_EQ(snap.disconnectReason, "xrun_storm");
    EXPECT_TRUE(playback.writeCalled);
    EXPECT_TRUE(playback.closeCalled);

    ::close(fds[1]);
}

TEST(PcmStreamHandler, DisconnectsOnPartialHeader) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    PcmStreamHandler handler(playback, server, stop, cfg);

    int fds[2]{-1, -1};
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

    auto header = makeValidHeader();
    ASSERT_EQ(::send(fds[0], &header, sizeof(header) / 2, 0),
              static_cast<ssize_t>(sizeof(header) / 2));
    ::close(fds[0]);  // simulate client closing early

    EXPECT_FALSE(handler.handleClientForTest(fds[1]));
    EXPECT_FALSE(playback.openCalled);
    EXPECT_FALSE(playback.closeCalled);

    ::close(fds[1]);
}

TEST(PcmStreamHandler, TakeoverInterruptsCurrentClient) {
    FakeAlsaPlayback playback;
    TcpServerOptions opts;
    opts.connectionMode = ConnectionMode::Takeover;
    opts.backlog = 4;
    TcpServer server(0, opts);
    ASSERT_TRUE(server.start());

    std::atomic_bool stop{false};
    PcmStreamConfig cfg{};
    cfg.connectionMode = ConnectionMode::Takeover;
    cfg.maxConsecutiveTimeouts = 10;
    PcmStreamHandler handler(playback, server, stop, cfg);

    int client1 = connectLoopback(server.boundPort());
    ASSERT_GE(client1, 0);
    auto firstAccept = server.acceptClient();
    ASSERT_TRUE(firstAccept.accepted);

    auto header = makeValidHeader();
    ASSERT_EQ(::send(client1, &header, sizeof(header), 0), sizeof(header));
    std::vector<std::uint8_t> pcm(32, 0);
    ASSERT_EQ(::send(client1, pcm.data(), pcm.size(), 0), static_cast<ssize_t>(pcm.size()));

    std::thread handlerThread(
        [&]() { EXPECT_FALSE(handler.handleClientForTest(server.clientFd())); });

    int client2 = connectLoopback(server.boundPort());
    ASSERT_GE(client2, 0);
    auto secondAccept = server.acceptClient();
    EXPECT_TRUE(secondAccept.takeoverPending);
    ASSERT_TRUE(server.hasPendingClient());

    auto header2 = makeValidHeader();
    ASSERT_EQ(::send(client2, &header2, sizeof(header2), 0), sizeof(header2));
    ASSERT_EQ(::send(client2, pcm.data(), pcm.size(), 0), static_cast<ssize_t>(pcm.size()));
    ::close(client2);  // new client finishes sending

    handlerThread.join();
    EXPECT_TRUE(server.hasPendingClient());
    server.promotePendingClient();

    std::thread handlerThread2(
        [&]() { EXPECT_TRUE(handler.handleClientForTest(server.clientFd())); });
    handlerThread2.join();

    server.closeClient();
    server.stop();
    ::close(client1);
    ::close(client2);
}
