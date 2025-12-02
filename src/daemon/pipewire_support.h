#pragma once

#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>

namespace pipewire_support {

struct InputContext {
    struct pw_main_loop* loop = nullptr;
    struct pw_stream* input_stream = nullptr;
    struct spa_source* signal_check_timer = nullptr;  // Timer for checking signal flags
    bool gpu_ready = false;
};

bool handle_rate_change(int detected_sample_rate);

extern const struct pw_stream_events kInputStreamEvents;

}  // namespace pipewire_support

