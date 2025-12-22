#include "daemon/output/playback_buffer_manager.h"
#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST(PlaybackBufferManagerTest, ThrottleBlocksUntilQueueDrains) {
    constexpr std::size_t kCapacity = 100;
    constexpr int kOutputRate = 1000;

    daemon_output::PlaybackBufferManager manager([]() { return kCapacity; });

    std::vector<float> data(kCapacity, 1.0f);
    std::size_t stored = 0;
    std::size_t dropped = 0;
    ASSERT_TRUE(manager.enqueue(data.data(), data.data(), kCapacity, kOutputRate, stored, dropped));
    EXPECT_EQ(manager.queuedFramesLocked(), kCapacity);

    std::atomic<bool> running{true};
    std::atomic<bool> throttling{false};

    std::thread producer([&]() {
        throttling.store(true, std::memory_order_release);
        manager.throttleProducerIfFull(running, []() { return kOutputRate; });
        throttling.store(false, std::memory_order_release);
    });

    std::this_thread::sleep_for(50ms);
    EXPECT_TRUE(throttling.load(std::memory_order_acquire));

    std::vector<float> outLeft(kCapacity);
    std::vector<float> outRight(kCapacity);
    ASSERT_TRUE(manager.readPlanar(outLeft.data(), outRight.data(), 80));

    producer.join();
    EXPECT_FALSE(throttling.load(std::memory_order_acquire));
    EXPECT_LT(manager.queuedFramesLocked(), kCapacity);
}
