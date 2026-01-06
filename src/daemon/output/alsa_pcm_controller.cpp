#include "daemon/output/alsa_pcm_controller.h"

#include "daemon/output/alsa_write_loop.h"
#include "logging/logger.h"
#include "logging/metrics.h"

#include <algorithm>
#include <alsa/asoundlib.h>
#include <chrono>
#include <thread>

namespace daemon_output {
namespace {

constexpr std::uint32_t kChannels = 2;

bool isRunning(const std::atomic<bool>* running) {
    return !running || running->load(std::memory_order_acquire);
}

}  // namespace

AlsaPcmController::AlsaPcmController(AlsaPcmControllerDependencies deps) : deps_(std::move(deps)) {
    channels_ = kChannels;
}

AlsaPcmController::~AlsaPcmController() {
    close();
}

void AlsaPcmController::markConnected(const std::string& device, const char* logMessage) {
    if (deps_.outputReady) {
        deps_.outputReady->store(true, std::memory_order_release);
    }
    if (logMessage && *logMessage) {
        LOG_INFO("{}", logMessage);
    }
    if (deps_.dacManager) {
        deps_.dacManager->markActiveDevice(device, true);
    }
}

void AlsaPcmController::markDisconnected(const std::string& device, const char* logMessage) {
    if (deps_.outputReady) {
        deps_.outputReady->store(false, std::memory_order_release);
    }
    if (deps_.streamingCacheManager) {
        deps_.streamingCacheManager->flushCaches();
        (void)deps_.streamingCacheManager->drainFlushRequests();
    }
    if (logMessage && *logMessage) {
        LOG_INFO("{}", logMessage);
    }
    if (deps_.dacManager) {
        deps_.dacManager->markActiveDevice(device, false);
    }
}

void AlsaPcmController::refreshPeriodFrames() {
    periodFrames_ = 0;
    if (!pcmHandle_) {
        return;
    }
    auto* pcm = static_cast<snd_pcm_t*>(pcmHandle_);
    snd_pcm_hw_params_t* curParams;
    snd_pcm_hw_params_alloca(&curParams);
    if (snd_pcm_hw_params_current(pcm, curParams) != 0) {
        return;
    }
    snd_pcm_uframes_t detected = 0;
    if (snd_pcm_hw_params_get_period_size(curParams, &detected, nullptr) == 0 && detected > 0) {
        periodFrames_ = static_cast<std::uint64_t>(detected);
    }
}

void AlsaPcmController::close() {
    if (!pcmHandle_) {
        return;
    }
    auto* pcm = static_cast<snd_pcm_t*>(pcmHandle_);
    snd_pcm_drop(pcm);
    snd_pcm_close(pcm);
    pcmHandle_ = nullptr;
    if (!currentDevice_.empty()) {
        markDisconnected(currentDevice_, "DAC disconnected - stopping input processing");
    }
    currentDevice_.clear();
    periodFrames_ = 0;
}

bool AlsaPcmController::openForDevice(const std::string& device, int forcedSampleRate) {
    if (!deps_.config) {
        LOG_ERROR("[ALSA] Cannot open PCM: config not set");
        return false;
    }
    if (device.empty()) {
        LOG_ERROR("[ALSA] Cannot open PCM: empty device");
        return false;
    }

    snd_pcm_t* pcm = nullptr;
    int err = snd_pcm_open(&pcm, device.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        LOG_ERROR("[ALSA] Cannot open device {}: {}", device, snd_strerror(err));
        return false;
    }

    snd_pcm_hw_params_t* hwParams;
    snd_pcm_hw_params_alloca(&hwParams);
    snd_pcm_hw_params_any(pcm, hwParams);

    if ((err = snd_pcm_hw_params_set_access(pcm, hwParams, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0 ||
        (err = snd_pcm_hw_params_set_format(pcm, hwParams, SND_PCM_FORMAT_S32_LE)) < 0) {
        LOG_ERROR("[ALSA] Cannot set access/format: {}", snd_strerror(err));
        snd_pcm_close(pcm);
        return false;
    }

    int configuredRate = forcedSampleRate;
    if (configuredRate <= 0 && deps_.currentOutputRate) {
        configuredRate = deps_.currentOutputRate();
    }
    if (configuredRate <= 0) {
        configuredRate = static_cast<int>(deps_.config->upsampleRatio) * 44100;
    }
    auto rate = static_cast<unsigned int>(configuredRate);
    if ((err = snd_pcm_hw_params_set_rate_near(pcm, hwParams, &rate, nullptr)) < 0) {
        LOG_ERROR("[ALSA] Cannot set sample rate: {}", snd_strerror(err));
        snd_pcm_close(pcm);
        return false;
    }
    if (rate != static_cast<unsigned int>(configuredRate)) {
        LOG_ERROR("[ALSA] Requested sample rate {} not supported (got {})", configuredRate, rate);
        snd_pcm_close(pcm);
        return false;
    }

    if ((err = snd_pcm_hw_params_set_channels(pcm, hwParams, channels_)) < 0) {
        LOG_ERROR("[ALSA] Cannot set channel count: {}", snd_strerror(err));
        snd_pcm_close(pcm);
        return false;
    }

    auto bufferSize = static_cast<snd_pcm_uframes_t>(deps_.config->bufferSize);
    auto periodSize = static_cast<snd_pcm_uframes_t>(deps_.config->periodSize);
    if ((err = snd_pcm_hw_params_set_period_size_near(pcm, hwParams, &periodSize, nullptr)) < 0) {
        LOG_ERROR("[ALSA] Cannot set period size: {}", snd_strerror(err));
        snd_pcm_close(pcm);
        return false;
    }
    bufferSize = std::max<snd_pcm_uframes_t>(bufferSize, periodSize * 2);
    if ((err = snd_pcm_hw_params_set_buffer_size_near(pcm, hwParams, &bufferSize)) < 0) {
        LOG_ERROR("[ALSA] Cannot set buffer size: {}", snd_strerror(err));
        snd_pcm_close(pcm);
        return false;
    }

    if ((err = snd_pcm_hw_params(pcm, hwParams)) < 0) {
        LOG_ERROR("[ALSA] Cannot set hardware parameters: {}", snd_strerror(err));
        snd_pcm_close(pcm);
        return false;
    }

    snd_pcm_hw_params_get_period_size(hwParams, &periodSize, nullptr);
    snd_pcm_hw_params_get_buffer_size(hwParams, &bufferSize);

    if ((err = snd_pcm_prepare(pcm)) < 0) {
        LOG_ERROR("[ALSA] Cannot prepare device: {}", snd_strerror(err));
        snd_pcm_close(pcm);
        return false;
    }

    snd_pcm_sw_params_t* swParams;
    snd_pcm_sw_params_alloca(&swParams);
    if (snd_pcm_sw_params_current(pcm, swParams) == 0) {
        snd_pcm_sw_params_set_start_threshold(pcm, swParams, bufferSize);
        snd_pcm_sw_params_set_avail_min(pcm, swParams, periodSize);
        if (snd_pcm_sw_params(pcm, swParams) < 0) {
            LOG_WARN("[ALSA] Failed to set software parameters");
        }
    }

    // Swap in the handle only on success.
    if (pcmHandle_) {
        auto* old = static_cast<snd_pcm_t*>(pcmHandle_);
        snd_pcm_drop(old);
        snd_pcm_close(old);
    }
    pcmHandle_ = pcm;
    currentDevice_ = device;
    refreshPeriodFrames();

    LOG_INFO(
        "[ALSA] Output device {} configured ({} Hz, 32-bit int, {}ch) buffer {} frames, "
        "period {} frames",
        device, rate, channels_, bufferSize, periodSize);
    return true;
}

bool AlsaPcmController::openSelected() {
    if (!deps_.dacManager) {
        LOG_ERROR("[ALSA] Cannot open selected device: DacManager missing");
        return false;
    }

    std::string device = deps_.dacManager->waitForSelection();
    while (isRunning(deps_.running) && pcmHandle_ == nullptr) {
        if (device.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            device = deps_.dacManager->waitForSelection();
            continue;
        }
        if (openForDevice(device, /*forcedSampleRate=*/0)) {
            markConnected(device, "DAC connected - input processing enabled");
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        device = deps_.dacManager->waitForSelection();
    }
    return pcmHandle_ != nullptr;
}

bool AlsaPcmController::alive() const {
    if (!pcmHandle_) {
        return false;
    }
    auto* pcm = static_cast<snd_pcm_t*>(pcmHandle_);
    snd_pcm_status_t* status;
    snd_pcm_status_alloca(&status);
    if (snd_pcm_status(pcm, status) < 0) {
        return false;
    }
    snd_pcm_state_t st = snd_pcm_status_get_state(status);
    if (st == SND_PCM_STATE_DISCONNECTED || st == SND_PCM_STATE_SUSPENDED) {
        return false;
    }
    return true;
}

bool AlsaPcmController::reconfigure(int newSampleRate) {
    if (!isRunning(deps_.running)) {
        return false;
    }
    if (currentDevice_.empty()) {
        LOG_ERROR("[ALSA] Cannot reconfigure: no active device");
        return false;
    }

    // Drop/close current handle before attempting reopen.
    if (pcmHandle_) {
        auto* pcm = static_cast<snd_pcm_t*>(pcmHandle_);
        snd_pcm_drop(pcm);
        snd_pcm_close(pcm);
        pcmHandle_ = nullptr;
        LOG_INFO("[ALSA] Closed PCM handle for reconfiguration");
    }
    bool ok = openForDevice(currentDevice_, newSampleRate);
    if (ok) {
        LOG_INFO("[ALSA] Reconfigured for {} Hz", newSampleRate);
    }
    return ok;
}

bool AlsaPcmController::switchDevice(const std::string& nextDevice) {
    if (!isRunning(deps_.running)) {
        return false;
    }
    if (nextDevice.empty() || nextDevice == currentDevice_) {
        return false;
    }

    if (pcmHandle_) {
        auto* pcm = static_cast<snd_pcm_t*>(pcmHandle_);
        snd_pcm_drop(pcm);
        snd_pcm_close(pcm);
        pcmHandle_ = nullptr;
    }
    if (!currentDevice_.empty()) {
        markDisconnected(currentDevice_, "DAC disconnected - stopping input processing");
    }

    currentDevice_ = nextDevice;
    while (isRunning(deps_.running) && pcmHandle_ == nullptr) {
        if (openForDevice(currentDevice_, /*forcedSampleRate=*/0)) {
            markConnected(currentDevice_, "DAC reconnected - resuming input processing");
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return pcmHandle_ != nullptr;
}

long AlsaPcmController::writeInterleaved(const std::int32_t* interleaved, std::size_t frames) {
    if (!pcmHandle_ || !interleaved || frames == 0 || channels_ == 0) {
        return 0;
    }
    auto* pcm = static_cast<snd_pcm_t*>(pcmHandle_);

    auto writeFn = [&](const std::int32_t* ptr, std::size_t requestedFrames) -> long {
        return static_cast<long>(
            snd_pcm_writei(pcm, ptr, static_cast<snd_pcm_uframes_t>(requestedFrames)));
    };
    auto recoverFn = [&](long err) -> long {
        return static_cast<long>(snd_pcm_recover(pcm, err, 0));
    };
    auto yieldFn = [&]() { std::this_thread::sleep_for(std::chrono::milliseconds(1)); };
    auto runningFn = [&]() { return isRunning(deps_.running); };
    auto onXrun = [&]() {
        if (deps_.fallbackManager) {
            deps_.fallbackManager->notifyXrun();
        }
        gpu_upsampler::metrics::recordOutputXrun();
        const char* device = currentDevice_.empty() ? "unknown" : currentDevice_.c_str();
        LOG_WARN("ALSA: XRUN detected at output buffer (device: {})", device);
    };

    return daemon_output::alsa_write_loop::writeAllInterleaved(interleaved, frames, channels_,
                                                               writeFn, recoverFn, yieldFn,
                                                               runningFn, onXrun, EAGAIN, EPIPE);
}

}  // namespace daemon_output
