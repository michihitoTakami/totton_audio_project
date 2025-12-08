#include "status_tracker.h"

void StatusTracker::setListening(uint16_t port) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.listening = true;
    status_.boundPort = port;
}

void StatusTracker::clearListening() {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.listening = false;
    status_.boundPort = 0;
}

void StatusTracker::setClientConnected(bool connected) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.clientConnected = connected;
    if (!connected) {
        status_.streaming = false;
    }
}

void StatusTracker::setStreaming(bool streaming) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.streaming = streaming;
}

void StatusTracker::setHeader(const PcmHeader& header) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.header.present = true;
    status_.header.header = header;
}

void StatusTracker::updateRingConfig(std::size_t ringFrames, std::size_t watermarkFrames) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.ring.configuredFrames = ringFrames;
    status_.ring.watermarkFrames = watermarkFrames;
}

void StatusTracker::updateRingBuffer(std::size_t bufferedFrames, std::size_t maxBufferedFrames,
                                     std::size_t droppedFrames) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.ring.bufferedFrames = bufferedFrames;
    status_.ring.maxBufferedFrames = maxBufferedFrames;
    status_.ring.droppedFrames = droppedFrames;
}

void StatusTracker::incrementXrun() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++status_.xrunCount;
}

StatusSnapshot StatusTracker::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

void StatusTracker::setDisconnectReason(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.disconnectReason = reason;
}

void StatusTracker::clearDisconnectReason() {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.disconnectReason.clear();
}
