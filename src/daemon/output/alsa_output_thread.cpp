#include "daemon/output/alsa_output_thread.h"

#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/core/thread_priority.h"
#include "daemon/output/alsa_pcm_controller.h"
#include "daemon/output/playback_buffer_access.h"
#include "io/playback_buffer.h"
#include "logging/logger.h"

#include <algorithm>
#include <alsa/asoundlib.h>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace daemon_output {

static void maybe_restore_soft_mute_params(daemon_app::RuntimeState& state) {
    if (!state.softMute.controller) {
        return;
    }
    if (!state.softMute.restorePending.load(std::memory_order_acquire)) {
        return;
    }
    SoftMute::MuteState st = state.softMute.controller->getState();
    if (st != SoftMute::MuteState::PLAYING && st != SoftMute::MuteState::MUTED) {
        return;
    }

    std::lock_guard<std::mutex> lock(state.softMute.opMutex);
    if (!state.softMute.controller) {
        return;
    }
    int fadeMs = state.softMute.restoreFadeMs.load(std::memory_order_relaxed);
    int sr = state.softMute.restoreSampleRate.load(std::memory_order_relaxed);
    state.softMute.controller->setFadeDuration(fadeMs);
    state.softMute.controller->setSampleRate(sr);
    state.softMute.restorePending.store(false, std::memory_order_release);
}

// Helper function for soft mute during filter switching (Issue #266)
// Fade-out: 1.5 seconds, perform filter switch, fade-in: 1.5 seconds
//
// Thread safety & responsiveness:
// - Called from ZeroMQ command thread, guarded by a mutex to serialize parameter updates
// - Non-blocking: start fade-out, perform switch, then trigger fade-in with minimal wait
// - Original fade parameters are restored in the audio thread once the transition settles

static size_t get_playback_ready_threshold(daemon_app::RuntimeState& state, size_t period_size) {
    bool crossfeedActive = false;
    size_t crossfeedBlockSize = 0;
    size_t producerBlockSize = 0;

    if (state.crossfeed.enabled.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> cf_lock(state.crossfeed.crossfeedMutex);
        if (state.crossfeed.processor) {
            crossfeedActive = true;
            crossfeedBlockSize = state.crossfeed.processor->getStreamValidInputPerBlock();
        }
    }

    if (state.upsampler) {
        // StreamValidInputPerBlock() is in input frames. Multiply by upsample ratio to obtain the
        // number of samples the producer actually contributes to the playback ring per block so the
        // ALSA thread can wake up as soon as a full GPU block finishes.
        size_t streamBlock = state.upsampler->getStreamValidInputPerBlock();
        int upsampleRatio = state.upsampler->getUpsampleRatio();
        if (streamBlock > 0 && upsampleRatio > 0) {
            producerBlockSize = streamBlock * static_cast<size_t>(upsampleRatio);
        }
    }

    return PlaybackBuffer::computeReadyThreshold(period_size, crossfeedActive, crossfeedBlockSize,
                                                 producerBlockSize);
}

// Fallback manager (Issue #139)

// Crossfeed enable/disable safety (Issue #888)
// - Avoid mixing pre/post switch audio by clearing playback + streaming caches.
// - Do not touch SoftMute here; caller wraps this with a fade-out/in.

void alsaOutputThread(daemon_app::RuntimeState& state) {
    daemon_core::elevateRealtimePriority("ALSA output");

    daemon_output::AlsaPcmController pcmController(daemon_output::AlsaPcmControllerDependencies{
        .config = &state.config,
        .dacManager = state.managers.dacManager.get(),
        .streamingCacheManager = state.managers.streamingCacheManager.get(),
        .fallbackManager = state.fallbackManager,
        .running = &state.flags.running,
        .outputReady = &state.flags.outputReady,
        .currentOutputRate =
            [&state]() { return state.rates.currentOutputRate.load(std::memory_order_acquire); },
    });

    if (!pcmController.openSelected()) {
        return;
    }

    auto period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
    if (period_size == 0) {
        period_size = static_cast<snd_pcm_uframes_t>(
            (state.config.periodSize > 0) ? state.config.periodSize : 32768);
    }
    std::vector<int32_t> interleaved_buffer(period_size * DaemonConstants::CHANNELS);
    std::vector<float> float_buffer(period_size *
                                    DaemonConstants::CHANNELS);  // for soft mute processing
    auto& bufferManager = daemon_output::playbackBuffer(state);

    // Main playback loop
    while (state.flags.running) {
        // Heartbeat check every few hundred loops
        static int alive_counter = 0;
        if (++alive_counter > 200) {  // ~200 iterations ~ a few seconds depending on buffer wait
            alive_counter = 0;
            if (!pcmController.alive()) {
                LOG_EVERY_N(WARN, 5, "[ALSA] PCM disconnected/suspended, attempting reopen...");
                pcmController.close();
                while (state.flags.running && !pcmController.openSelected()) {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                }
                period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                if (period_size == 0) {
                    period_size = static_cast<snd_pcm_uframes_t>(
                        (state.config.periodSize > 0) ? state.config.periodSize : 32768);
                }
                interleaved_buffer.resize(period_size * DaemonConstants::CHANNELS);
                float_buffer.resize(period_size * DaemonConstants::CHANNELS);
                continue;
            }
        }

        // Issue #219: Check for pending ALSA reconfiguration (output rate changed)
        if (state.rates.alsaReconfigureNeeded.exchange(false, std::memory_order_acquire)) {
            int new_output_rate = state.rates.alsaNewOutputRate.load(std::memory_order_acquire);
            if (new_output_rate > 0) {
                LOG_INFO("[Main] Reconfiguring ALSA for new output rate {} Hz", new_output_rate);

                // Reconfigure ALSA with new rate
                if (pcmController.reconfigure(new_output_rate)) {
                    period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                    if (period_size == 0) {
                        period_size = static_cast<snd_pcm_uframes_t>(
                            (state.config.periodSize > 0) ? state.config.periodSize : 32768);
                    }
                    interleaved_buffer.resize(period_size * DaemonConstants::CHANNELS);
                    float_buffer.resize(period_size * DaemonConstants::CHANNELS);

                    // Update soft mute sample rate
                    if (state.softMute.controller) {
                        state.softMute.controller->setSampleRate(new_output_rate);
                    }

                    LOG_INFO("[Main] ALSA reconfiguration successful");
                } else {
                    // Failed to reconfigure - try to reopen with old rate
                    LOG_ERROR("[Main] ALSA reconfiguration failed, attempting recovery...");
                    int old_rate = state.rates.currentOutputRate.load(std::memory_order_acquire);
                    if (!pcmController.reconfigure(old_rate)) {
                        LOG_ERROR("[Main] ALSA recovery failed, waiting for reconnect...");
                    }
                }
            }
        }

        if (auto pendingDevice = state.managers.dacManager->consumePendingChange()) {
            std::string nextDevice = *pendingDevice;
            if (!nextDevice.empty() && nextDevice != pcmController.device()) {
                LOG_INFO("[ALSA] Switching output to {}", nextDevice);
                if (!pcmController.switchDevice(nextDevice)) {
                    continue;
                }
                period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
                if (period_size == 0) {
                    period_size = static_cast<snd_pcm_uframes_t>(
                        (state.config.periodSize > 0) ? state.config.periodSize : 32768);
                }
                interleaved_buffer.resize(period_size * DaemonConstants::CHANNELS);
                float_buffer.resize(period_size * DaemonConstants::CHANNELS);
            }
        }

        // Wait for GPU processed data (dynamic threshold to avoid underflow with crossfeed)
        std::unique_lock<std::mutex> lock(bufferManager.mutex());
        size_t ready_threshold =
            get_playback_ready_threshold(state, static_cast<size_t>(period_size));
        int outputRate = state.rates.currentOutputRate.load(std::memory_order_acquire);
        if (outputRate <= 0) {
            outputRate = DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE;
        }
        double periodMs =
            (outputRate > 0 && period_size > 0)
                ? (1000.0 * static_cast<double>(period_size) / static_cast<double>(outputRate))
                : 10.0;
        auto waitDuration =
            std::chrono::milliseconds(static_cast<int64_t>(std::clamp(periodMs, 5.0, 50.0)));

        // Issue #1232: wake roughly once per ALSA period even when input is silent, so we keep
        // feeding zeros and avoid miscounting silence as XRUN underflow.
        bufferManager.cv().wait_for(lock, waitDuration, [&bufferManager, ready_threshold, &state] {
            return bufferManager.queuedFramesLocked() >= ready_threshold || !state.flags.running;
        });

        if (!state.flags.running) {
            break;
        }

        lock.unlock();

        audio_pipeline::RenderResult renderResult;
        if (state.audioPipeline) {
            renderResult = state.audioPipeline->renderOutput(static_cast<size_t>(period_size),
                                                             interleaved_buffer, float_buffer,
                                                             state.softMute.controller);
        } else {
            renderResult.framesRequested = period_size;
            renderResult.framesRendered = period_size;
            renderResult.wroteSilence = true;
            std::fill(interleaved_buffer.begin(), interleaved_buffer.end(), 0);
        }

        // Wake any producers that are throttling on "space available".
        // renderOutput() advances the output read position under its own lock; notifying here is
        // safe.
        bufferManager.cv().notify_all();

        // Apply pending soft mute parameter restoration once transition completes
        maybe_restore_soft_mute_params(state);

        // Write to ALSA device
        long frames_written = pcmController.writeInterleaved(interleaved_buffer.data(),
                                                             static_cast<size_t>(period_size));
        if (frames_written < 0) {
            // Device may be gone; attempt reopen
            LOG_EVERY_N(ERROR, 10, "[ALSA] Write error: {}, retrying reopen...",
                        snd_strerror(frames_written));
            pcmController.close();
            while (state.flags.running && !pcmController.openSelected()) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
            if (!pcmController.isOpen()) {
                continue;
            }
            period_size = static_cast<snd_pcm_uframes_t>(pcmController.periodFrames());
            if (period_size == 0) {
                period_size = static_cast<snd_pcm_uframes_t>(
                    (state.config.periodSize > 0) ? state.config.periodSize : 32768);
            }
            interleaved_buffer.resize(period_size * DaemonConstants::CHANNELS);
            float_buffer.resize(period_size * DaemonConstants::CHANNELS);
        }
    }

    // Cleanup
    pcmController.close();
    LOG_INFO("[ALSA] Output thread terminated");
}

}  // namespace daemon_output
