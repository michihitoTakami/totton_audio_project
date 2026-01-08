#pragma once

#include "logging/logger.h"

#include <cstdlib>
#include <cstring>
#include <string>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace daemon_core {

inline void elevateRealtimePriority(const char* name, int priority = 65) {
#ifdef __linux__
    // RT scheduling can freeze remote shells if something spins.
    // Allow disabling via env for containerized debugging.
    // - TOTTON_AUDIO_ENABLE_RT=0 disables SCHED_FIFO attempts.
    const char* enableRt = std::getenv("TOTTON_AUDIO_ENABLE_RT");
    if (enableRt && std::string(enableRt) == "0") {
        LOG_WARN("[RT] {} thread: SCHED_FIFO disabled via TOTTON_AUDIO_ENABLE_RT=0", name);
        return;
    }
    sched_param params{};
    params.sched_priority = priority;
    int ret = pthread_setschedparam(pthread_self(), SCHED_FIFO, &params);
    if (ret != 0) {
        LOG_WARN("[RT] Failed to set {} thread to SCHED_FIFO (errno={}): {}", name, ret,
                 std::strerror(ret));
    } else {
        LOG_INFO("[RT] {} thread priority set to SCHED_FIFO {}", name, priority);
    }
#else
    (void)name;
    (void)priority;
#endif
}

}  // namespace daemon_core
