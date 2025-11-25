#ifndef DAEMON_CONSTANTS_H
#define DAEMON_CONSTANTS_H

// Common constants shared between alsa_daemon and pipewire_daemon
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

// Peak limiter defaults
constexpr float DEFAULT_HEADROOM_TARGET = 0.92f;
constexpr float MIN_HEADROOM_TARGET = 0.5f;
constexpr float MAX_HEADROOM_TARGET = 0.999f;

// Soft mute constants (Issue #266)
constexpr int DEFAULT_SOFT_MUTE_FADE_MS = 50;  // Default fade duration for shutdown/reload
constexpr int FILTER_SWITCH_FADE_MS = 1500;    // Fade duration for filter switching (1.5 seconds)
constexpr int FILTER_SWITCH_FADE_TIMEOUT_MS = 2250;  // Timeout for fade completion (1.5x fade duration)

}  // namespace DaemonConstants

#endif  // DAEMON_CONSTANTS_H
