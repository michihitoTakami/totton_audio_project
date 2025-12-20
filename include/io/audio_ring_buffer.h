#ifndef AUDIO_RING_BUFFER_H
#define AUDIO_RING_BUFFER_H

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

// Lock-free ring buffer (single producer/single consumer) for audio samples
// Issue #105: Daemon common code extraction
//
// Usage:
//   AudioRingBuffer buffer;
//   buffer.init(capacity);  // capacity must be > 0
//   buffer.write(data, count);
//   buffer.read(dst, count);
//
// Thread safety:
//   Safe for single producer / single consumer pattern with atomic operations.
//   Producer thread calls write(), consumer thread calls read().
//   For multi-producer or multi-consumer, use mutex externally.
//
// Memory ordering / invariants:
//   - SPSC only: producer is the sole writer of tail_, consumer is the sole
//     writer of head_ (both use relaxed operations).
//   - size_ is the synchronization point between threads:
//       * write(): sample writes happen-before size_.fetch_add(..., release)
//       * read(): size_.load(acquire) happens-before reading samples
//   - clear() must be externally synchronized and called only when both threads
//     are stopped or paused (no concurrent write/read), otherwise data races.

class AudioRingBuffer {
   public:
    AudioRingBuffer() = default;

    void init(size_t capacity) {
        assert(capacity > 0 && "AudioRingBuffer capacity must be > 0");
        buffer_.assign(capacity, 0.0f);
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        size_.store(0, std::memory_order_relaxed);
    }

    size_t capacity() const {
        return buffer_.size();
    }

    size_t availableToRead() const {
        return size_.load(std::memory_order_acquire);
    }

    size_t availableToWrite() const {
        return capacity() - size_.load(std::memory_order_acquire);
    }

    // Producer thread calls this
    bool write(const float* data, size_t count) {
        size_t cap = capacity();
        if (cap == 0 || count > availableToWrite()) {
            return false;
        }
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t first = std::min(count, cap - tail);
        std::memcpy(buffer_.data() + tail, data, first * sizeof(float));
        size_t remaining = count - first;
        if (remaining > 0) {
            std::memcpy(buffer_.data(), data + first, remaining * sizeof(float));
        }
        tail_.store((tail + count) % cap, std::memory_order_relaxed);
        size_.fetch_add(count, std::memory_order_release);
        return true;
    }

    // Consumer thread calls this
    bool read(float* dst, size_t count) {
        size_t cap = capacity();
        if (cap == 0 || count > availableToRead()) {
            return false;
        }
        size_t head = head_.load(std::memory_order_relaxed);
        size_t first = std::min(count, cap - head);
        std::memcpy(dst, buffer_.data() + head, first * sizeof(float));
        size_t remaining = count - first;
        if (remaining > 0) {
            std::memcpy(dst + first, buffer_.data(), remaining * sizeof(float));
        }
        head_.store((head + count) % cap, std::memory_order_relaxed);
        size_.fetch_sub(count, std::memory_order_release);
        return true;
    }

    void clear() {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        size_.store(0, std::memory_order_release);
    }

   private:
    std::vector<float> buffer_;
    std::atomic<size_t> head_{0};  // read position (consumer updates)
    std::atomic<size_t> tail_{0};  // write position (producer updates)
    std::atomic<size_t> size_{0};  // samples stored (both read, respective owner updates)
};

#endif  // AUDIO_RING_BUFFER_H
