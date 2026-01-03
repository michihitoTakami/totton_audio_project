#include "daemon/audio_pipeline/switch_actions.h"

#include "audio/soft_mute.h"
#include "core/daemon_constants.h"
#include "daemon/audio/crossfeed_manager.h"
#include "daemon/audio_pipeline/audio_pipeline.h"
#include "daemon/audio_pipeline/stream_buffer_sizing.h"
#include "daemon/output/playback_buffer_access.h"
#include "logging/logger.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace audio_pipeline {

// #region agent log
static void agent_debug_log(const char* location, const char* message, const std::string& dataJson,
                            const char* hypothesisId) {
    std::ofstream ofs("/home/michihito/Working/gpu_os/.cursor/debug.log", std::ios::app);
    if (!ofs.is_open()) {
        return;
    }
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    ofs << "{\"sessionId\":\"debug-session\",\"runId\":\"pre-fix\",\"hypothesisId\":\""
        << (hypothesisId ? hypothesisId : "") << "\",\"location\":\"" << location
        << "\",\"message\":\"" << message << "\",\"data\":" << dataJson
        << ",\"timestamp\":" << nowMs << "}\n";
}
// #endregion

void applySoftMuteForFilterSwitch(daemon_app::RuntimeState& state,
                                  std::function<bool()> filterSwitchFunc) {
    using namespace DaemonConstants;

    if (!state.softMute.controller) {
        // If soft mute not initialized, perform switch without mute
        filterSwitchFunc();
        return;
    }

    std::lock_guard<std::mutex> lock(state.softMute.opMutex);

    // Cancel any stale pending restore (new switch supersedes)
    state.softMute.restorePending.store(false, std::memory_order_release);

    // Save current fade duration for restoration
    int originalFadeDuration = state.softMute.controller->getFadeDuration();
    int outputSampleRate = state.softMute.controller->getSampleRate();

    // Update fade duration for filter switching
    // Note: This is called from command thread, but audio thread may be processing.
    // The fade calculation will use the new duration from the next audio frame.
    state.softMute.controller->setFadeDuration(FILTER_SWITCH_FADE_MS);
    state.softMute.controller->setSampleRate(outputSampleRate);

    std::cout << "[Filter Switch] Starting fade-out (" << (FILTER_SWITCH_FADE_MS / 1000.0)
              << "s)..." << '\n';
    state.softMute.controller->startFadeOut();

    // Wait until near-silent (or timeout) before switching to avoid audible glitches.
    // NOTE: This runs on the command thread. Use a bounded timeout to avoid hanging the UI
    // if the audio thread is not advancing (e.g., output not running).
    auto fade_start = std::chrono::steady_clock::now();
    const auto fade_timeout = std::chrono::milliseconds(FILTER_SWITCH_FADE_TIMEOUT_MS);
    while (true) {
        SoftMute::MuteState st = state.softMute.controller->getState();
        float gain = state.softMute.controller->getCurrentGain();
        if (st == SoftMute::MuteState::MUTED || gain <= 0.001f) {
            break;
        }
        if (std::chrono::steady_clock::now() - fade_start > fade_timeout) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Perform filter switch while fade-out is progressing
    bool pauseOk = true;
    if (state.audioPipeline) {
        state.audioPipeline->requestRtPause();
        pauseOk = state.audioPipeline->waitForRtQuiescent(std::chrono::milliseconds(500));
        if (!pauseOk) {
            LOG_ERROR("[Filter Switch] RT pause handshake timed out (aborting switch)");
        }
    }

    bool switch_success = false;
    if (pauseOk) {
        switch_success = filterSwitchFunc();
    }

    if (state.audioPipeline) {
        state.audioPipeline->resumeRtPause();
    }

    if (switch_success) {
        // Start fade-in after filter switch
        std::cout << "[Filter Switch] Starting fade-in (" << (FILTER_SWITCH_FADE_MS / 1000.0)
                  << "s)..." << '\n';
        state.softMute.controller->startFadeIn();

        // Mark pending restoration to be applied once transition completes
        state.softMute.restoreFadeMs.store(originalFadeDuration, std::memory_order_relaxed);
        state.softMute.restoreSampleRate.store(outputSampleRate, std::memory_order_relaxed);
        state.softMute.restorePending.store(true, std::memory_order_release);
    } else {
        // If switch failed, restore original state immediately
        std::cerr << "[Filter Switch] Switch failed, restoring audio state" << '\n';
        state.softMute.controller->setPlaying();
        state.softMute.controller->setFadeDuration(originalFadeDuration);
        state.softMute.controller->setSampleRate(outputSampleRate);
    }
}

// Audio buffer for thread communication (managed component)

bool resetStreamingCachesForSwitch(daemon_app::RuntimeState& state) {
    struct RtPauseGuard {
        audio_pipeline::AudioPipeline* pipeline = nullptr;
        bool ok = true;
        explicit RtPauseGuard(audio_pipeline::AudioPipeline* p) : pipeline(p) {
            if (!pipeline) {
                return;
            }
            pipeline->requestRtPause();
            ok = pipeline->waitForRtQuiescent(std::chrono::milliseconds(500));
            if (!ok) {
                pipeline->resumeRtPause();
            }
        }
        ~RtPauseGuard() {
            if (pipeline && ok) {
                pipeline->resumeRtPause();
            }
        }
    } pauseGuard(state.audioPipeline.get());

    if (!pauseGuard.ok) {
        return false;
    }

    daemon_output::playbackBuffer(state).reset();
    daemon_output::playbackBuffer(state).cv().notify_all();

    {
        std::lock_guard<std::mutex> lock(state.streaming.streamingMutex);
        if (!state.streaming.streamInputLeft.empty()) {
            std::fill(state.streaming.streamInputLeft.begin(),
                      state.streaming.streamInputLeft.end(), 0.0f);
        }
        if (!state.streaming.streamInputRight.empty()) {
            std::fill(state.streaming.streamInputRight.begin(),
                      state.streaming.streamInputRight.end(), 0.0f);
        }
        state.streaming.streamAccumulatedLeft = 0;
        state.streaming.streamAccumulatedRight = 0;
        state.streaming.upsamplerOutputLeft.clear();
        state.streaming.upsamplerOutputRight.clear();
        if (state.upsampler) {
            state.upsampler->resetStreaming();
        }
    }

    { state.crossfeed.resetRequested.store(true, std::memory_order_release); }

    return true;
}

bool reinitializeStreamingForLegacyMode(daemon_app::RuntimeState& state) {
    if (!state.upsampler) {
        return false;
    }

    std::lock_guard<std::mutex> streamLock(state.streaming.streamingMutex);
    state.upsampler->resetStreaming();
    daemon_output::playbackBuffer(state).reset();

    state.streaming.streamInputLeft.clear();
    state.streaming.streamInputRight.clear();
    state.streaming.streamAccumulatedLeft = 0;
    state.streaming.streamAccumulatedRight = 0;
    state.streaming.upsamplerOutputLeft.clear();
    state.streaming.upsamplerOutputRight.clear();

    // Rebuild legacy streams so the buffers match the full FFT (avoids invalid cudaMemset after
    // disabling partitions).
    if (!state.upsampler->initializeStreaming()) {
        std::cerr << "[Partition] Failed to initialize legacy streaming buffers" << '\n';
        return false;
    }

    size_t buffer_capacity = audio_pipeline::computeStreamBufferCapacity(
        state, state.upsampler->getStreamValidInputPerBlock());
    state.streaming.streamInputLeft.resize(buffer_capacity, 0.0f);
    state.streaming.streamInputRight.resize(buffer_capacity, 0.0f);
    state.streaming.streamAccumulatedLeft = 0;
    state.streaming.streamAccumulatedRight = 0;

    size_t upsampler_output_capacity =
        buffer_capacity * static_cast<size_t>(state.upsampler->getUpsampleRatio());
    state.streaming.upsamplerOutputLeft.reserve(upsampler_output_capacity);
    state.streaming.upsamplerOutputRight.reserve(upsampler_output_capacity);

    return true;
}

bool handleRateSwitch(daemon_app::RuntimeState& state, int newInputRate) {
    // #region agent log
    {
        std::ostringstream data;
        data << "{\"newInputRate\":" << newInputRate
             << ",\"upsamplerNull\":" << (state.upsampler ? "false" : "true") << "}";
        agent_debug_log("switch_actions.cpp:handleRateSwitch:entry", "enter", data.str(), "H1");
    }
    // #endregion

    if (!state.upsampler || !state.upsampler->isMultiRateEnabled()) {
        std::cerr << "[Rate] Multi-rate mode not enabled (attempting non-multi-rate switch)"
                  << '\n';
        // #region agent log
        {
            const bool hasUpsampler = (state.upsampler != nullptr);
            const bool isMulti =
                (state.upsampler != nullptr) ? state.upsampler->isMultiRateEnabled() : false;
            std::ostringstream data;
            data << "{\"hasUpsampler\":" << (hasUpsampler ? "true" : "false")
                 << ",\"isMultiRate\":" << (isMulti ? "true" : "false") << "}";
            agent_debug_log("switch_actions.cpp:handleRateSwitch:early", "non_multi_rate_path",
                            data.str(), "H1");
        }
        // #endregion
        if (!state.upsampler) {
            return false;
        }
    }

    int currentRate = state.upsampler->getCurrentInputRate();
    if (currentRate == newInputRate) {
        std::cout << "[Rate] Already at target rate: " << newInputRate << " Hz" << '\n';
        // #region agent log
        {
            std::ostringstream data;
            data << "{\"currentRate\":" << currentRate << ",\"newInputRate\":" << newInputRate
                 << "}";
            agent_debug_log("switch_actions.cpp:handleRateSwitch:noop", "already_at_target",
                            data.str(), "H2");
        }
        // #endregion
        return true;
    }

    std::cout << "[Rate] Switching: " << currentRate << " Hz -> " << newInputRate << " Hz" << '\n';

    int savedRate = currentRate;
    ConvolutionEngine::RateFamily targetFamily = ConvolutionEngine::detectRateFamily(newInputRate);
    if (targetFamily == ConvolutionEngine::RateFamily::RATE_UNKNOWN) {
        targetFamily = ConvolutionEngine::RateFamily::RATE_44K;
    }

    if (state.softMute.controller) {
        state.softMute.controller->startFadeOut();
        auto startTime = std::chrono::steady_clock::now();
        while (state.softMute.controller->isTransitioning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto elapsed = std::chrono::steady_clock::now() - startTime;
            if (elapsed > std::chrono::milliseconds(200)) {
                std::cerr << "[Rate] Warning: Fade-out timeout" << '\n';
                break;
            }
        }
    }

    int newOutputRate = state.upsampler->getOutputSampleRate();
    int newUpsampleRatio = state.upsampler->getUpsampleRatio();
    size_t buffer_capacity = 0;

    struct RtPauseGuard {
        audio_pipeline::AudioPipeline* pipeline = nullptr;
        bool ok = true;
        explicit RtPauseGuard(audio_pipeline::AudioPipeline* p) : pipeline(p) {
            if (!pipeline) {
                return;
            }
            pipeline->requestRtPause();
            ok = pipeline->waitForRtQuiescent(std::chrono::milliseconds(500));
            if (!ok) {
                LOG_ERROR("[Rate] RT pause handshake timed out (aborting rate switch)");
            }
        }
        ~RtPauseGuard() {
            if (pipeline) {
                pipeline->resumeRtPause();
            }
        }
    } pauseGuard(state.audioPipeline.get());

    if (!pauseGuard.ok) {
        if (state.softMute.controller) {
            state.softMute.controller->setPlaying();
        }
        return false;
    }

    {
        std::lock_guard<std::mutex> streamLock(state.streaming.streamingMutex);

        state.upsampler->resetStreaming();

        daemon_output::playbackBuffer(state).reset();
        state.streaming.streamInputLeft.clear();
        state.streaming.streamInputRight.clear();
        state.streaming.streamAccumulatedLeft = 0;
        state.streaming.streamAccumulatedRight = 0;

        if (!state.upsampler->switchToInputRate(newInputRate)) {
            std::cerr << "[Rate] Failed to switch rate, rolling back" << '\n';
            // #region agent log
            {
                std::ostringstream data;
                data << "{\"newInputRate\":" << newInputRate << ",\"savedRate\":" << savedRate
                     << "}";
                agent_debug_log("switch_actions.cpp:handleRateSwitch:switch",
                                "switchToInputRate_failed", data.str(), "H3");
            }
            // #endregion
            if (state.upsampler->switchToInputRate(savedRate)) {
                std::cout << "[Rate] Rollback successful: restored to " << savedRate << " Hz"
                          << '\n';
            } else {
                std::cerr << "[Rate] ERROR: Rollback failed!" << '\n';
            }
            if (state.softMute.controller) {
                state.softMute.controller->startFadeIn();
            }
            return false;
        }

        if (!state.upsampler->initializeStreaming()) {
            std::cerr << "[Rate] Failed to re-initialize streaming mode, rolling back" << '\n';
            if (state.upsampler->switchToInputRate(savedRate)) {
                if (state.upsampler->initializeStreaming()) {
                    std::cout << "[Rate] Rollback successful: restored to " << savedRate << " Hz"
                              << '\n';
                }
            }
            if (state.softMute.controller) {
                state.softMute.controller->startFadeIn();
            }
            return false;
        }

        state.rates.inputSampleRate = newInputRate;
        state.rates.currentRateFamilyInt.store(static_cast<int>(targetFamily),
                                               std::memory_order_release);
        state.rates.activeRateFamily = targetFamily;
        newOutputRate = state.upsampler->getOutputSampleRate();
        newUpsampleRatio = state.upsampler->getUpsampleRatio();

        buffer_capacity = audio_pipeline::computeStreamBufferCapacity(
            state, state.upsampler->getStreamValidInputPerBlock());
        state.streaming.streamInputLeft.resize(buffer_capacity, 0.0f);
        state.streaming.streamInputRight.resize(buffer_capacity, 0.0f);
        state.streaming.streamAccumulatedLeft = 0;
        state.streaming.streamAccumulatedRight = 0;
        size_t upsampler_output_capacity =
            buffer_capacity * static_cast<size_t>(state.upsampler->getUpsampleRatio());
        state.streaming.upsamplerOutputLeft.reserve(upsampler_output_capacity);
        state.streaming.upsamplerOutputRight.reserve(upsampler_output_capacity);

        if (state.softMute.controller) {
            delete state.softMute.controller;
        }
        state.softMute.controller = new SoftMute::Controller(50, newOutputRate);
    }

    if (state.crossfeed.processor) {
        std::lock_guard<std::mutex> cfLock(state.crossfeed.crossfeedMutex);
        auto status = daemon_audio::switchCrossfeedRateFamilyLocked(state.crossfeed, state.config,
                                                                    targetFamily);
        if (status == daemon_audio::CrossfeedSwitchStatus::Failed) {
            std::cerr << "[Rate] Warning: Failed to switch crossfeed HRTF rate family" << '\n';
        }
    }

    if (state.softMute.controller) {
        state.softMute.controller->startFadeIn();
    }

    std::cout << "[Rate] Switch complete: " << newInputRate << " Hz (" << newUpsampleRatio
              << "x -> " << newOutputRate << " Hz)" << '\n';
    std::cout << "[Rate] Streaming buffers re-initialized: " << buffer_capacity
              << " samples capacity" << '\n';

    // #region agent log
    {
        std::ostringstream data;
        data << "{\"newInputRate\":" << newInputRate << ",\"newOutputRate\":" << newOutputRate
             << ",\"newUpsampleRatio\":" << newUpsampleRatio << "}";
        agent_debug_log("switch_actions.cpp:handleRateSwitch:done", "success", data.str(), "H4");
    }
    // #endregion

    return true;
}

}  // namespace audio_pipeline
