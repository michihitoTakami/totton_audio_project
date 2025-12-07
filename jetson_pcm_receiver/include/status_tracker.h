#pragma once

#include "pcm_header.h"

#include <cstddef>
#include <cstdint>
#include <mutex>

struct RingBufferSnapshot {
    std::size_t configuredFrames{0};
    std::size_t watermarkFrames{0};
    std::size_t bufferedFrames{0};
    std::size_t maxBufferedFrames{0};
    std::size_t droppedFrames{0};
};

struct HeaderSnapshot {
    bool present{false};
    PcmHeader header{};
};

struct StatusSnapshot {
    bool listening{false};
    uint16_t boundPort{0};
    bool clientConnected{false};
    bool streaming{false};
    std::size_t xrunCount{0};
    RingBufferSnapshot ring;
    HeaderSnapshot header;
};

// ストリーミング状態を集約し、ZMQレスポンスなどで使えるスナップショットを提供する。
class StatusTracker {
   public:
    void setListening(uint16_t port);
    void clearListening();
    void setClientConnected(bool connected);
    void setStreaming(bool streaming);
    void setHeader(const PcmHeader& header);
    void updateRingConfig(std::size_t ringFrames, std::size_t watermarkFrames);
    void updateRingBuffer(std::size_t bufferedFrames, std::size_t maxBufferedFrames,
                          std::size_t droppedFrames);
    void incrementXrun();

    StatusSnapshot snapshot() const;

   private:
    mutable std::mutex mutex_;
    StatusSnapshot status_;
};
