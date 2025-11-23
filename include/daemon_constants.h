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

}  // namespace DaemonConstants

#endif  // DAEMON_CONSTANTS_H
