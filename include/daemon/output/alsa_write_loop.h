#pragma once

#include <cstdint>
#include <functional>
#include <type_traits>

namespace daemon_output {
namespace alsa_write_loop {

// A small, testable helper for writing interleaved int32 frames in a loop.
//
// - Handles short writes (0 < written < frames)
// - Treats 0 / -EAGAIN as "try again later" via yieldFn()
// - Calls onXrun() on -EPIPE before attempting recoverFn()
// - Uses recoverFn(err) to recover; if recoverFn returns < 0, the original error is returned.
//
// This is intentionally ALSA-agnostic to allow unit testing without snd_pcm_t.
inline long writeAllInterleaved(const int32_t* interleaved, size_t frames, unsigned int channels,
                                const std::function<long(const int32_t*, size_t)>& writeFn,
                                const std::function<long(long)>& recoverFn,
                                const std::function<void()>& yieldFn,
                                const std::function<bool()>& runningFn,
                                const std::function<void()>& onXrun, long eagainErrno,
                                long epipeErrno) {
    if (!interleaved || frames == 0 || channels == 0) {
        return 0;
    }

    size_t totalWritten = 0;
    while (totalWritten < frames && (!runningFn || runningFn())) {
        const int32_t* ptr = interleaved + totalWritten * static_cast<size_t>(channels);
        const size_t remaining = frames - totalWritten;
        const long written = writeFn ? writeFn(ptr, remaining) : 0;

        if (written > 0) {
            totalWritten += static_cast<size_t>(written);
            continue;
        }

        if (written == 0 || written == -eagainErrno) {
            if (yieldFn) {
                yieldFn();
            }
            continue;
        }

        if (written == -epipeErrno) {
            if (onXrun) {
                onXrun();
            }
        }

        const long rec = recoverFn ? recoverFn(written) : -1;
        if (rec < 0) {
            return written;
        }
    }

    return static_cast<long>(totalWritten);
}

}  // namespace alsa_write_loop
}  // namespace daemon_output
