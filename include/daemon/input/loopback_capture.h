#pragma once

#include "core/config_loader.h"

#include <alsa/asoundlib.h>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace audio_pipeline {
class AudioPipeline;
}  // namespace audio_pipeline

namespace daemon_output {
class PlaybackBufferManager;
}  // namespace daemon_output

namespace daemon_input {

struct LoopbackCaptureDependencies {
    daemon_output::PlaybackBufferManager* playbackBuffer = nullptr;
    std::atomic<bool>* running = nullptr;
    std::atomic<int>* currentOutputRate = nullptr;
    audio_pipeline::AudioPipeline** audioPipeline = nullptr;
    std::mutex* handleMutex = nullptr;
    snd_pcm_t** handle = nullptr;
    std::atomic<bool>* captureRunning = nullptr;
    std::atomic<bool>* captureReady = nullptr;
};

snd_pcm_format_t parseLoopbackFormat(const std::string& formatStr);
bool validateLoopbackConfig(const AppConfig& cfg);
bool convertPcmToFloat(const void* src, snd_pcm_format_t format, size_t frames,
                       unsigned int channels, std::vector<float>& dst);
void waitForCaptureReady(snd_pcm_t* handle);

void loopbackCaptureThread(const std::string& device, snd_pcm_format_t format, unsigned int rate,
                           unsigned int channels, snd_pcm_uframes_t periodFrames,
                           const LoopbackCaptureDependencies& deps);

}  // namespace daemon_input
