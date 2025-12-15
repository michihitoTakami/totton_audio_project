#ifndef DAEMON_CONSTANTS_H
#define DAEMON_CONSTANTS_H

#include <cstddef>  // for size_t

// Common constants shared across daemon components
// Issue #105: Daemon common code extraction

namespace DaemonConstants {

// Audio format constants
constexpr int DEFAULT_INPUT_SAMPLE_RATE = 44100;
constexpr int DEFAULT_UPSAMPLE_RATIO = 16;
constexpr int DEFAULT_BLOCK_SIZE = 4096;
constexpr int CHANNELS = 2;

// Derived constants
constexpr int DEFAULT_OUTPUT_SAMPLE_RATE =
    DEFAULT_INPUT_SAMPLE_RATE * DEFAULT_UPSAMPLE_RATIO;  // 705600 Hz

// Output buffer safety limits
constexpr float MAX_OUTPUT_BUFFER_SECONDS = 2.0f;  // Keep at most 2 seconds of audio
constexpr size_t DEFAULT_MAX_OUTPUT_BUFFER_FRAMES =
    static_cast<size_t>(DEFAULT_OUTPUT_SAMPLE_RATE * MAX_OUTPUT_BUFFER_SECONDS);

// Peak limiter defaults
constexpr float DEFAULT_HEADROOM_TARGET = 0.92f;
constexpr float MIN_HEADROOM_TARGET = 0.5f;
constexpr float MAX_HEADROOM_TARGET = 0.999f;

// Soft mute constants (Issue #266)
constexpr int DEFAULT_SOFT_MUTE_FADE_MS = 50;  // Default fade duration for shutdown/reload
constexpr int FILTER_SWITCH_FADE_MS = 1500;    // Fade duration for filter switching (1.5 seconds)
constexpr int FILTER_SWITCH_FADE_TIMEOUT_MS =
    2250;  // Timeout for fade completion (1.5x fade duration)

// ZeroMQ endpoints
constexpr const char* ZEROMQ_IPC_PATH = "ipc:///tmp/gpu_os.sock";
constexpr const char* ZEROMQ_PUB_SUFFIX = ".pub";

}  // namespace DaemonConstants

#endif  // DAEMON_CONSTANTS_H
