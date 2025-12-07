#include "alsa_playback.h"
#include "pcm_stream_handler.h"
#include "tcp_server.h"

#include <cstring>
#include <gtest/gtest.h>
#include <sys/socket.h>
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

    static bool isSupportedRate(uint32_t rate) {
        constexpr uint32_t base[] = {44100, 48000};
        constexpr uint32_t mul[] = {1, 2, 4, 8, 16};
        for (auto b : base) {
            for (auto m : mul) {
                if (b * m == rate) {
                    return true;
                }
            }
        }
        return false;
    }

    bool shouldFailOpen{false};
    bool shouldFailWrite{false};
    bool openCalled{false};
    bool writeCalled{false};
    bool closeCalled{false};
    bool opened{false};
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

}  // namespace

TEST(PcmStreamHandler, AcceptsValidHeaderAndWritesFrames) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    PcmStreamHandler handler(playback, server);

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

TEST(PcmStreamHandler, AcceptsS24AndHighRate) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    PcmStreamHandler handler(playback, server);

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

TEST(PcmStreamHandler, RejectsUnsupportedRate) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    PcmStreamHandler handler(playback, server);

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
    PcmStreamHandler handler(playback, server);

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
    PcmStreamHandler handler(playback, server);

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

TEST(PcmStreamHandler, DisconnectsOnPartialHeader) {
    FakeAlsaPlayback playback;
    TcpServer server(0);
    PcmStreamHandler handler(playback, server);

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
