/**
 * @file test_audio_common.cpp
 * @brief Unit tests for common audio utilities (AudioRingBuffer, AudioUtils)
 *
 * Tests the shared audio components extracted in Issue #105.
 */

#include "audio/audio_utils.h"
#include "io/audio_ring_buffer.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

// ============================================================================
// AudioRingBuffer Tests
// ============================================================================

class AudioRingBufferTest : public ::testing::Test {
   protected:
    AudioRingBuffer buffer_;
};

// Basic functionality tests

TEST_F(AudioRingBufferTest, Init_SetsCapacity) {
    buffer_.init(1024);
    EXPECT_EQ(buffer_.capacity(), 1024);
}

TEST_F(AudioRingBufferTest, Init_StartsEmpty) {
    buffer_.init(1024);
    EXPECT_EQ(buffer_.availableToRead(), 0);
    EXPECT_EQ(buffer_.availableToWrite(), 1024);
}

TEST_F(AudioRingBufferTest, Write_UpdatesAvailable) {
    buffer_.init(1024);
    std::vector<float> data(100, 1.0f);

    EXPECT_TRUE(buffer_.write(data.data(), 100));
    EXPECT_EQ(buffer_.availableToRead(), 100);
    EXPECT_EQ(buffer_.availableToWrite(), 924);
}

TEST_F(AudioRingBufferTest, Write_FailsWhenFull) {
    buffer_.init(100);
    std::vector<float> data(100, 1.0f);

    EXPECT_TRUE(buffer_.write(data.data(), 100));
    EXPECT_FALSE(buffer_.write(data.data(), 1));  // Buffer is full
}

TEST_F(AudioRingBufferTest, Write_FailsWhenOverCapacity) {
    buffer_.init(100);
    std::vector<float> data(101, 1.0f);

    EXPECT_FALSE(buffer_.write(data.data(), 101));
}

TEST_F(AudioRingBufferTest, Read_UpdatesAvailable) {
    buffer_.init(1024);
    std::vector<float> writeData(100, 1.0f);
    std::vector<float> readData(100);

    buffer_.write(writeData.data(), 100);
    EXPECT_TRUE(buffer_.read(readData.data(), 50));
    EXPECT_EQ(buffer_.availableToRead(), 50);
    EXPECT_EQ(buffer_.availableToWrite(), 974);
}

TEST_F(AudioRingBufferTest, Read_FailsWhenEmpty) {
    buffer_.init(1024);
    std::vector<float> data(100);

    EXPECT_FALSE(buffer_.read(data.data(), 1));  // Buffer is empty
}

TEST_F(AudioRingBufferTest, Read_FailsWhenUnderAvailable) {
    buffer_.init(1024);
    std::vector<float> writeData(50, 1.0f);
    std::vector<float> readData(100);

    buffer_.write(writeData.data(), 50);
    EXPECT_FALSE(buffer_.read(readData.data(), 100));  // Only 50 available
}

TEST_F(AudioRingBufferTest, ReadWrite_DataIntegrity) {
    buffer_.init(1024);
    std::vector<float> writeData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> readData(5);

    buffer_.write(writeData.data(), 5);
    buffer_.read(readData.data(), 5);

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(readData[i], writeData[i]);
    }
}

TEST_F(AudioRingBufferTest, WrapAround_WriteThenRead) {
    buffer_.init(10);
    std::vector<float> data(8, 1.0f);
    std::vector<float> readData(8);

    // Fill most of buffer
    buffer_.write(data.data(), 8);
    buffer_.read(readData.data(), 8);

    // Write again - this should wrap around
    std::vector<float> wrapData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    EXPECT_TRUE(buffer_.write(wrapData.data(), 6));

    std::vector<float> wrapReadData(6);
    EXPECT_TRUE(buffer_.read(wrapReadData.data(), 6));

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(wrapReadData[i], wrapData[i]);
    }
}

TEST_F(AudioRingBufferTest, Clear_ResetsBuffer) {
    buffer_.init(1024);
    std::vector<float> data(100, 1.0f);

    buffer_.write(data.data(), 100);
    buffer_.clear();

    EXPECT_EQ(buffer_.availableToRead(), 0);
    EXPECT_EQ(buffer_.availableToWrite(), 1024);
}

TEST_F(AudioRingBufferTest, MultipleWriteRead_Cycles) {
    buffer_.init(256);

    for (int cycle = 0; cycle < 100; ++cycle) {
        std::vector<float> writeData(64);
        for (size_t i = 0; i < 64; ++i) {
            writeData[i] = static_cast<float>(static_cast<size_t>(cycle * 64) + i);
        }

        EXPECT_TRUE(buffer_.write(writeData.data(), 64));

        std::vector<float> readData(64);
        EXPECT_TRUE(buffer_.read(readData.data(), 64));

        for (size_t i = 0; i < 64; ++i) {
            EXPECT_FLOAT_EQ(readData[i], writeData[i]);
        }
    }
}

TEST_F(AudioRingBufferTest, UninitiailizedBuffer_WriteReturnsFalse) {
    // Default constructed buffer has capacity 0
    std::vector<float> data(10, 1.0f);
    EXPECT_FALSE(buffer_.write(data.data(), 10));
}

TEST_F(AudioRingBufferTest, UninitiailizedBuffer_ReadReturnsFalse) {
    std::vector<float> data(10);
    EXPECT_FALSE(buffer_.read(data.data(), 10));
}

// Thread safety test (basic smoke test)
TEST_F(AudioRingBufferTest, ConcurrentAccess_SPSCPattern) {
    buffer_.init(4096);
    const size_t blockSize = 64;
    const size_t totalSamples = blockSize * 150;  // 9600 samples (divisible by blockSize)
    std::atomic<bool> producerDone{false};
    std::atomic<size_t> samplesWritten{0};
    std::atomic<size_t> samplesRead{0};

    // Producer thread
    std::thread producer([&]() {
        std::vector<float> data(blockSize);
        size_t written = 0;
        while (written < totalSamples) {
            for (size_t i = 0; i < blockSize; ++i) {
                data[i] = static_cast<float>(written + i);
            }
            if (buffer_.write(data.data(), blockSize)) {
                written += blockSize;
                samplesWritten.store(written, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
        producerDone.store(true, std::memory_order_release);
    });

    // Consumer thread
    std::thread consumer([&]() {
        std::vector<float> data(blockSize);
        size_t read = 0;
        while (read < totalSamples) {
            if (buffer_.read(data.data(), blockSize)) {
                // Verify data integrity
                for (size_t i = 0; i < blockSize; ++i) {
                    EXPECT_FLOAT_EQ(data[i], static_cast<float>(read + i));
                }
                read += blockSize;
                samplesRead.store(read, std::memory_order_relaxed);
            } else {
                if (producerDone.load(std::memory_order_acquire) &&
                    buffer_.availableToRead() == 0) {
                    break;
                }
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(samplesWritten.load(), totalSamples);
    EXPECT_EQ(samplesRead.load(), totalSamples);
}

TEST_F(AudioRingBufferTest, ConcurrentAccess_StressSequence) {
    buffer_.init(4096);
    const size_t totalSamples = 1 << 18;
    const size_t maxChunk = 128;
    std::atomic<bool> producerDone{false};
    std::atomic<bool> mismatch{false};
    std::atomic<size_t> samplesWritten{0};
    std::atomic<size_t> samplesRead{0};

    std::thread producer([&]() {
        std::minstd_rand rng(12345);
        std::uniform_int_distribution<size_t> dist(1, maxChunk);
        std::vector<float> data(maxChunk);
        size_t written = 0;
        while (written < totalSamples && !mismatch.load(std::memory_order_relaxed)) {
            size_t chunk = std::min(dist(rng), totalSamples - written);
            if (buffer_.availableToWrite() < chunk) {
                std::this_thread::yield();
                continue;
            }
            for (size_t i = 0; i < chunk; ++i) {
                data[i] = static_cast<float>(written + i);
            }
            if (buffer_.write(data.data(), chunk)) {
                written += chunk;
                samplesWritten.store(written, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
        producerDone.store(true, std::memory_order_release);
    });

    std::thread consumer([&]() {
        std::minstd_rand rng(67890);
        std::uniform_int_distribution<size_t> dist(1, maxChunk);
        std::vector<float> data(maxChunk);
        size_t read = 0;
        while (read < totalSamples && !mismatch.load(std::memory_order_relaxed)) {
            size_t chunk = std::min(dist(rng), totalSamples - read);
            if (buffer_.availableToRead() < chunk) {
                if (producerDone.load(std::memory_order_acquire) &&
                    buffer_.availableToRead() == 0) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }
            if (buffer_.read(data.data(), chunk)) {
                for (size_t i = 0; i < chunk; ++i) {
                    if (data[i] != static_cast<float>(read + i)) {
                        mismatch.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
                read += chunk;
                samplesRead.store(read, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_FALSE(mismatch.load());
    EXPECT_EQ(samplesWritten.load(), totalSamples);
    EXPECT_EQ(samplesRead.load(), totalSamples);
}

// ============================================================================
// AudioUtils Tests
// ============================================================================

class AudioUtilsTest : public ::testing::Test {};

TEST_F(AudioUtilsTest, DeinterleaveStereo_CorrectSeparation) {
    // Interleaved: L0, R0, L1, R1, L2, R2, L3, R3
    std::vector<float> interleaved = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> left(4);
    std::vector<float> right(4);

    AudioUtils::deinterleaveStereo(interleaved.data(), left.data(), right.data(), 4);

    EXPECT_FLOAT_EQ(left[0], 1.0f);
    EXPECT_FLOAT_EQ(left[1], 3.0f);
    EXPECT_FLOAT_EQ(left[2], 5.0f);
    EXPECT_FLOAT_EQ(left[3], 7.0f);

    EXPECT_FLOAT_EQ(right[0], 2.0f);
    EXPECT_FLOAT_EQ(right[1], 4.0f);
    EXPECT_FLOAT_EQ(right[2], 6.0f);
    EXPECT_FLOAT_EQ(right[3], 8.0f);
}

TEST_F(AudioUtilsTest, InterleaveStereo_CorrectMerge) {
    std::vector<float> left = {1.0f, 3.0f, 5.0f, 7.0f};
    std::vector<float> right = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> interleaved(8);

    AudioUtils::interleaveStereo(left.data(), right.data(), interleaved.data(), 4);

    EXPECT_FLOAT_EQ(interleaved[0], 1.0f);
    EXPECT_FLOAT_EQ(interleaved[1], 2.0f);
    EXPECT_FLOAT_EQ(interleaved[2], 3.0f);
    EXPECT_FLOAT_EQ(interleaved[3], 4.0f);
    EXPECT_FLOAT_EQ(interleaved[4], 5.0f);
    EXPECT_FLOAT_EQ(interleaved[5], 6.0f);
    EXPECT_FLOAT_EQ(interleaved[6], 7.0f);
    EXPECT_FLOAT_EQ(interleaved[7], 8.0f);
}

TEST_F(AudioUtilsTest, InterleaveStereoWithGain_AppliesGain) {
    std::vector<float> left = {1.0f, 2.0f};
    std::vector<float> right = {3.0f, 4.0f};
    std::vector<float> interleaved(4);

    AudioUtils::interleaveStereoWithGain(left.data(), right.data(), interleaved.data(), 2, 0.5f);

    EXPECT_FLOAT_EQ(interleaved[0], 0.5f);  // 1.0 * 0.5
    EXPECT_FLOAT_EQ(interleaved[1], 1.5f);  // 3.0 * 0.5
    EXPECT_FLOAT_EQ(interleaved[2], 1.0f);  // 2.0 * 0.5
    EXPECT_FLOAT_EQ(interleaved[3], 2.0f);  // 4.0 * 0.5
}

TEST_F(AudioUtilsTest, InterleaveStereoWithGain_ZeroGain) {
    std::vector<float> left = {1.0f, 2.0f};
    std::vector<float> right = {3.0f, 4.0f};
    std::vector<float> interleaved(4);

    AudioUtils::interleaveStereoWithGain(left.data(), right.data(), interleaved.data(), 2, 0.0f);

    for (float sample : interleaved) {
        EXPECT_FLOAT_EQ(sample, 0.0f);
    }
}

TEST_F(AudioUtilsTest, RoundTrip_DeinterleaveInterleave) {
    std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> left(3);
    std::vector<float> right(3);
    std::vector<float> restored(6);

    AudioUtils::deinterleaveStereo(original.data(), left.data(), right.data(), 3);
    AudioUtils::interleaveStereo(left.data(), right.data(), restored.data(), 3);

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(restored[i], original[i]);
    }
}

TEST_F(AudioUtilsTest, DeinterleaveStereo_ZeroFrames) {
    std::vector<float> interleaved;
    std::vector<float> left;
    std::vector<float> right;

    // Should not crash
    AudioUtils::deinterleaveStereo(interleaved.data(), left.data(), right.data(), 0);
}

TEST_F(AudioUtilsTest, InterleaveStereo_ZeroFrames) {
    std::vector<float> left;
    std::vector<float> right;
    std::vector<float> interleaved;

    // Should not crash
    AudioUtils::interleaveStereo(left.data(), right.data(), interleaved.data(), 0);
}
