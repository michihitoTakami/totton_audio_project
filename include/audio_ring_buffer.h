#ifndef AUDIO_RING_BUFFER_H
#define AUDIO_RING_BUFFER_H

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

// Simple lock-free ring buffer (single producer/consumer) for audio samples
// Issue #105: Daemon common code extraction
//
// Usage:
//   AudioRingBuffer buffer;
//   buffer.init(capacity);
//   buffer.write(data, count);
//   buffer.read(dst, count);
//
// Thread safety:
//   Safe for single producer / single consumer pattern.
//   For multi-producer or multi-consumer, use mutex externally.

class AudioRingBuffer {
   public:
    AudioRingBuffer() = default;

    void init(size_t capacity) {
        buffer_.assign(capacity, 0.0f);
        head_ = tail_ = size_ = 0;
    }

    size_t capacity() const {
        return buffer_.size();
    }
    size_t availableToRead() const {
        return size_;
    }
    size_t availableToWrite() const {
        return capacity() - size_;
    }

    bool write(const float* data, size_t count) {
        if (count > availableToWrite()) {
            return false;
        }
        size_t cap = capacity();
        size_t first = std::min(count, cap - tail_);
        std::memcpy(buffer_.data() + tail_, data, first * sizeof(float));
        size_t remaining = count - first;
        if (remaining > 0) {
            std::memcpy(buffer_.data(), data + first, remaining * sizeof(float));
        }
        tail_ = (tail_ + count) % cap;
        size_ += count;
        return true;
    }

    bool read(float* dst, size_t count) {
        if (count > size_) {
            return false;
        }
        size_t cap = capacity();
        size_t first = std::min(count, cap - head_);
        std::memcpy(dst, buffer_.data() + head_, first * sizeof(float));
        size_t remaining = count - first;
        if (remaining > 0) {
            std::memcpy(dst + first, buffer_.data(), remaining * sizeof(float));
        }
        head_ = (head_ + count) % cap;
        size_ -= count;
        return true;
    }

    void clear() {
        head_ = tail_ = size_ = 0;
    }

   private:
    std::vector<float> buffer_;
    size_t head_ = 0;  // read position
    size_t tail_ = 0;  // write position
    size_t size_ = 0;  // samples stored
};

#endif  // AUDIO_RING_BUFFER_H
