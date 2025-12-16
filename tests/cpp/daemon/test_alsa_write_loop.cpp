#include "daemon/output/alsa_write_loop.h"

#include <cerrno>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace {

TEST(AlsaWriteLoopTest, HandlesShortWritesAndEagainWithCorrectPointerOffset) {
    // 2ch interleaved: [L0 R0 L1 R1 ...]
    constexpr unsigned int kChannels = 2;
    constexpr size_t kFrames = 8;
    std::vector<int32_t> interleaved(kFrames * kChannels, 0);

    std::vector<const int32_t*> ptrs;
    std::vector<size_t> reqFrames;
    int yieldCount = 0;
    int call = 0;

    auto writeFn = [&](const int32_t* ptr, size_t frames) -> long {
        ptrs.push_back(ptr);
        reqFrames.push_back(frames);
        // Write pattern:
        // 1) short write 3 frames
        // 2) EAGAIN
        // 3) 0 (also treated as retry)
        // 4) remaining frames
        ++call;
        if (call == 1) {
            return 3;
        }
        if (call == 2) {
            return -EAGAIN;
        }
        if (call == 3) {
            return 0;
        }
        return static_cast<long>(frames);
    };

    auto recoverFn = [&](long) -> long {
        // Should not be called in this test.
        return -1;
    };

    auto yieldFn = [&]() { ++yieldCount; };
    auto runningFn = [&]() { return true; };
    auto onXrun = [&]() { FAIL() << "onXrun should not be called"; };

    long written = daemon_output::alsa_write_loop::writeAllInterleaved(
        interleaved.data(), kFrames, kChannels, writeFn, recoverFn, yieldFn, runningFn, onXrun,
        EAGAIN, EPIPE);
    EXPECT_EQ(written, static_cast<long>(kFrames));
    EXPECT_EQ(yieldCount, 2);
    ASSERT_GE(ptrs.size(), 2u);

    // First call starts at frame 0
    EXPECT_EQ(ptrs[0], interleaved.data());
    // Second call is retry at frame 3 (because 3 frames were written)
    EXPECT_EQ(ptrs[1], interleaved.data() + static_cast<size_t>(3 * kChannels));
}

TEST(AlsaWriteLoopTest, CallsOnXrunAndUsesRecoverThenContinues) {
    constexpr unsigned int kChannels = 2;
    constexpr size_t kFrames = 4;
    std::vector<int32_t> interleaved(kFrames * kChannels, 0);

    int xrunCount = 0;
    int recoverCount = 0;
    int call = 0;

    auto writeFn = [&](const int32_t*, size_t frames) -> long {
        ++call;
        if (call == 1) {
            return -EPIPE;
        }
        return static_cast<long>(frames);
    };

    auto recoverFn = [&](long err) -> long {
        ++recoverCount;
        EXPECT_EQ(err, -EPIPE);
        return 0;  // recovered
    };

    auto yieldFn = [&]() {};
    auto runningFn = [&]() { return true; };
    auto onXrun = [&]() { ++xrunCount; };

    long written = daemon_output::alsa_write_loop::writeAllInterleaved(
        interleaved.data(), kFrames, kChannels, writeFn, recoverFn, yieldFn, runningFn, onXrun,
        EAGAIN, EPIPE);
    EXPECT_EQ(written, static_cast<long>(kFrames));
    EXPECT_EQ(xrunCount, 1);
    EXPECT_EQ(recoverCount, 1);
}

TEST(AlsaWriteLoopTest, ReturnsOriginalErrorIfRecoverFails) {
    constexpr unsigned int kChannels = 2;
    constexpr size_t kFrames = 4;
    std::vector<int32_t> interleaved(kFrames * kChannels, 0);

    auto writeFn = [&](const int32_t*, size_t) -> long { return -123; };
    auto recoverFn = [&](long) -> long { return -1; };
    auto yieldFn = [&]() {};
    auto runningFn = [&]() { return true; };
    auto onXrun = [&]() {};

    long written = daemon_output::alsa_write_loop::writeAllInterleaved(
        interleaved.data(), kFrames, kChannels, writeFn, recoverFn, yieldFn, runningFn, onXrun,
        EAGAIN, EPIPE);
    EXPECT_EQ(written, -123);
}

}  // namespace
