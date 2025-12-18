#include "daemon/control/control_plane.h"

#include "audio/audio_utils.h"
#include "audio/eq_parser.h"
#include "audio/eq_to_fir.h"
#include "convolution_engine.h"
#include "daemon/api/events.h"
#include "logging/logger.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <thread>
#include <utility>

namespace daemon_control {
namespace {

std::string buildOkResponse(const daemon_ipc::ZmqRequest& request, const std::string& message = "",
                            const nlohmann::json& data = {}) {
    if (request.isJson) {
        nlohmann::json resp;
        resp["status"] = "ok";
        if (!message.empty()) {
            resp["message"] = message;
        }
        if (!data.is_null() && !data.empty()) {
            resp["data"] = data;
        }
        return resp.dump();
    }

    if (!data.is_null() && !data.empty()) {
        return "OK:" + data.dump();
    }
    if (!message.empty()) {
        return "OK:" + message;
    }
    return "OK";
}

std::string buildErrorResponse(const daemon_ipc::ZmqRequest& request, const std::string& code,
                               const std::string& message) {
    if (request.isJson) {
        nlohmann::json resp;
        resp["status"] = "error";
        resp["error_code"] = code;
        resp["message"] = message;
        return resp.dump();
    }
    return "ERR:" + message;
}

std::string normalizeOutputMode(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool isSupportedOutputMode(const std::string& normalized) {
    static constexpr std::array<const char*, 1> kSupportedOutputModes = {"usb"};
    for (const auto* mode : kSupportedOutputModes) {
        if (normalized == mode) {
            return true;
        }
    }
    return false;
}

ConvolutionEngine::FourChannelFIR* crossfeedProcessor(const CrossfeedControls& controls) {
    if (!controls.processor) {
        return nullptr;
    }
    return *controls.processor;
}

ConvolutionEngine::GPUUpsampler* upsampler(ControlPlaneDependencies& deps) {
    if (!deps.upsampler) {
        return nullptr;
    }
    return *deps.upsampler;
}

SoftMute::Controller* softMute(ControlPlaneDependencies& deps) {
    if (!deps.softMute) {
        return nullptr;
    }
    return *deps.softMute;
}

bool applySoftMuteForCrossfeedSwitch(ControlPlaneDependencies& deps,
                                     const std::function<bool()>& switchFunc) {
    auto* mute = softMute(deps);
    if (!mute) {
        return switchFunc();
    }

    mute->startFadeOut();

    auto fadeStart = std::chrono::steady_clock::now();
    const auto quickTimeout = std::chrono::milliseconds(250);
    while (true) {
        SoftMute::MuteState st = mute->getState();
        float gain = mute->getCurrentGain();
        if (st == SoftMute::MuteState::MUTED || gain < 0.05f) {
            break;
        }
        if (std::chrono::steady_clock::now() - fadeStart > quickTimeout) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    bool ok = switchFunc();
    if (ok) {
        mute->startFadeIn();
    } else {
        mute->setPlaying();
    }
    return ok;
}

}  // namespace

ControlPlane::ControlPlane(ControlPlaneDependencies deps) : deps_(std::move(deps)) {}

ControlPlane::~ControlPlane() {
    stop();
}

bool ControlPlane::start() {
    zmqServer_ = std::make_unique<daemon_ipc::ZmqCommandServer>();
    registerHandlers();

    if (zmqServer_->start()) {
        startStatsThread();
        return true;
    }

    if (deps_.zmqBindFailed) {
        deps_.zmqBindFailed->store(true, std::memory_order_release);
    }
    if (deps_.runningFlag) {
        deps_.runningFlag->store(false, std::memory_order_release);
    }
    if (deps_.quitMainLoop) {
        deps_.quitMainLoop();
    }
    return false;
}

void ControlPlane::stop() {
    stopStatsThread();
    if (zmqServer_) {
        zmqServer_->stop();
        zmqServer_.reset();
    }
}

std::function<void(const nlohmann::json&)> ControlPlane::eventPublisher() {
    return [this](const nlohmann::json& payload) { publish(payload); };
}

void ControlPlane::publish(const nlohmann::json& payload) {
    if (!zmqServer_) {
        return;
    }
    zmqServer_->publish(payload.dump());
}

void ControlPlane::registerHandlers() {
    zmqServer_->registerCommand("PING", [this](const auto& req) { return handlePing(req); });
    zmqServer_->registerCommand("RELOAD", [this](const auto& req) { return handleReload(req); });
    zmqServer_->registerCommand("STATS", [this](const auto& req) { return handleStats(req); });
    zmqServer_->registerCommand("CROSSFEED_ENABLE",
                                [this](const auto& req) { return handleCrossfeedEnable(req); });
    zmqServer_->registerCommand("CROSSFEED_DISABLE",
                                [this](const auto& req) { return handleCrossfeedDisable(req); });
    zmqServer_->registerCommand(
        "CROSSFEED_STATUS", [this](const auto& req) { return handleCrossfeedStatus(req, false); });
    zmqServer_->registerCommand("CROSSFEED_GET_STATUS",
                                [this](const auto& req) { return handleCrossfeedGetStatus(req); });
    zmqServer_->registerCommand("CROSSFEED_SET_COMBINED", [this](const auto& req) {
        return handleCrossfeedSetCombined(req);
    });
    zmqServer_->registerCommand("CROSSFEED_GENERATE_WOODWORTH",
                                [this](const auto& req) { return handleCrossfeedGenerate(req); });
    zmqServer_->registerCommand("CROSSFEED_SET_SIZE",
                                [this](const auto& req) { return handleCrossfeedSetSize(req); });
    zmqServer_->registerCommand("DAC_LIST", [this](const auto& req) { return handleDacList(req); });
    zmqServer_->registerCommand("DAC_STATUS",
                                [this](const auto& req) { return handleDacStatus(req); });
    zmqServer_->registerCommand("DAC_SELECT",
                                [this](const auto& req) { return handleDacSelect(req); });
    zmqServer_->registerCommand("DAC_RESCAN",
                                [this](const auto& req) { return handleDacRescan(req); });
    zmqServer_->registerCommand("OUTPUT_MODE_GET",
                                [this](const auto& req) { return handleOutputModeGet(req); });
    zmqServer_->registerCommand("OUTPUT_MODE_SET",
                                [this](const auto& req) { return handleOutputModeSet(req); });
    zmqServer_->registerCommand("PHASE_TYPE_GET",
                                [this](const auto& req) { return handlePhaseTypeGet(req); });
    zmqServer_->registerCommand("PHASE_TYPE_SET",
                                [this](const auto& req) { return handlePhaseTypeSet(req); });
}

void ControlPlane::startStatsThread() {
    if (!deps_.buildRuntimeStats || !deps_.bufferCapacityFrames || deps_.statsFilePath.empty()) {
        return;
    }
    if (statsThreadRunning_.exchange(true)) {
        return;
    }
    statsThread_ = std::thread([this]() {
        size_t lastLoggedClips = 0;
        while (statsThreadRunning_.load(std::memory_order_acquire)) {
            runtime_stats::writeStatsFile(deps_.buildRuntimeStats(), deps_.bufferCapacityFrames(),
                                          deps_.statsFilePath);
            size_t total = runtime_stats::totalSamples();
            size_t clips = runtime_stats::clipCount();
            if (clips > lastLoggedClips && total > 0) {
                LOG_WARN("Clipping detected: {} samples out of {} ({}%)", clips, total,
                         (100.0 * static_cast<double>(clips) / static_cast<double>(total)));
                lastLoggedClips = clips;
            }

            if (deps_.runningFlag && !deps_.runningFlag->load(std::memory_order_acquire)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });
}

void ControlPlane::stopStatsThread() {
    if (!statsThreadRunning_.exchange(false)) {
        return;
    }
    if (statsThread_.joinable()) {
        statsThread_.join();
    }
}

std::string ControlPlane::handlePing(const daemon_ipc::ZmqRequest& request) {
    return buildOkResponse(request);
}

std::string ControlPlane::handleReload(const daemon_ipc::ZmqRequest& request) {
    if (deps_.reloadRequested) {
        deps_.reloadRequested->store(true);
    }
    auto* mute = softMute(deps_);
    if (mute) {
        mute->startFadeOut();
    }
    if (deps_.quitMainLoop) {
        deps_.quitMainLoop();
    }
    return buildOkResponse(request);
}

std::string ControlPlane::handleStats(const daemon_ipc::ZmqRequest& request) {
    auto stats = runtime_stats::collect(deps_.buildRuntimeStats(), deps_.bufferCapacityFrames());
    return buildOkResponse(request, "", stats);
}

std::string ControlPlane::handleCrossfeedEnable(const daemon_ipc::ZmqRequest& request) {
    if (deps_.config && deps_.config->partitionedConvolution.enabled) {
        return buildErrorResponse(request, "CROSSFEED_DISABLED",
                                  "Crossfeed not available in low-latency mode");
    }

    // Preflight: avoid starting fade if enable will fail.
    {
        std::lock_guard<std::mutex> cfLock(*deps_.crossfeed.mutex);
        auto* processor = crossfeedProcessor(deps_.crossfeed);
        if (!processor || !processor->isInitialized()) {
            return buildErrorResponse(request, "CROSSFEED_NOT_INITIALIZED",
                                      "HRTF processor not initialized");
        }
    }

    bool enabled = applySoftMuteForCrossfeedSwitch(deps_, [&]() {
        if (deps_.resetStreamingCachesForSwitch) {
            if (!deps_.resetStreamingCachesForSwitch()) {
                return false;
            }
        }

        std::lock_guard<std::mutex> cfLock(*deps_.crossfeed.mutex);
        auto* processor = crossfeedProcessor(deps_.crossfeed);
        if (!processor || !processor->isInitialized()) {
            return false;
        }

        if (deps_.crossfeed.resetStreamingState) {
            deps_.crossfeed.resetStreamingState();
        }
        processor->setEnabled(true);
        if (deps_.crossfeed.enabledFlag) {
            deps_.crossfeed.enabledFlag->store(true);
        }
        return true;
    });

    if (!enabled) {
        return buildErrorResponse(request, "CROSSFEED_ENABLE_FAILED",
                                  "Failed to enable crossfeed safely");
    }
    return buildOkResponse(request, "Crossfeed enabled");
}

std::string ControlPlane::handleCrossfeedDisable(const daemon_ipc::ZmqRequest& request) {
    bool disabled = applySoftMuteForCrossfeedSwitch(deps_, [&]() {
        if (deps_.resetStreamingCachesForSwitch) {
            if (!deps_.resetStreamingCachesForSwitch()) {
                return false;
            }
        }

        std::lock_guard<std::mutex> cfLock(*deps_.crossfeed.mutex);
        if (deps_.crossfeed.enabledFlag) {
            deps_.crossfeed.enabledFlag->store(false);
        }
        auto* processor = crossfeedProcessor(deps_.crossfeed);
        if (processor) {
            processor->setEnabled(false);
        }
        if (deps_.crossfeed.resetStreamingState) {
            deps_.crossfeed.resetStreamingState();
        }
        return true;
    });

    if (!disabled) {
        return buildErrorResponse(request, "CROSSFEED_DISABLE_FAILED",
                                  "Failed to disable crossfeed safely");
    }
    return buildOkResponse(request, "Crossfeed disabled");
}

std::string ControlPlane::handleCrossfeedStatus(const daemon_ipc::ZmqRequest& request,
                                                bool includeHeadSize) {
    std::lock_guard<std::mutex> cfLock(*deps_.crossfeed.mutex);
    auto* processor = crossfeedProcessor(deps_.crossfeed);
    bool enabled =
        deps_.crossfeed.enabledFlag && deps_.crossfeed.enabledFlag->load(std::memory_order_relaxed);
    bool initialized = (processor != nullptr && processor->isInitialized());

    nlohmann::json data;
    data["enabled"] = enabled;
    data["initialized"] = initialized;

    if (includeHeadSize) {
        if (processor != nullptr) {
            ConvolutionEngine::HeadSize currentSize = processor->getCurrentHeadSize();
            data["head_size"] = ConvolutionEngine::headSizeToString(currentSize);
        } else {
            data["head_size"] = nullptr;
        }
    }

    return buildOkResponse(request, "", data);
}

std::string ControlPlane::handleCrossfeedGetStatus(const daemon_ipc::ZmqRequest& request) {
    return handleCrossfeedStatus(request, true);
}

std::string ControlPlane::handleCrossfeedSetCombined(const daemon_ipc::ZmqRequest& request) {
    return buildErrorResponse(request, "CROSSFEED_NOT_IMPLEMENTED",
                              "Combined filter is not supported in the integrated crossfeed path");
}

std::string ControlPlane::handleCrossfeedGenerate(const daemon_ipc::ZmqRequest& request) {
    return buildErrorResponse(request, "CROSSFEED_NOT_IMPLEMENTED",
                              "Woodworth HRTF generation is not supported in this path");
}

std::string ControlPlane::handleCrossfeedSetSize(const daemon_ipc::ZmqRequest& request) {
    if (!request.json || !request.json->contains("params")) {
        return buildErrorResponse(request, "IPC_INVALID_PARAMS", "Missing params field");
    }

    {
        std::lock_guard<std::mutex> cfLock(*deps_.crossfeed.mutex);
        auto* processor = crossfeedProcessor(deps_.crossfeed);
        if (!processor || !processor->isInitialized()) {
            return buildErrorResponse(request, "CROSSFEED_NOT_INITIALIZED",
                                      "HRTF processor not initialized");
        }
    }

    auto params = (*request.json)["params"];
    std::string sizeStr = params.value("head_size", "");
    if (sizeStr.empty()) {
        return buildErrorResponse(request, "IPC_INVALID_PARAMS", "Missing head_size parameter");
    }

    ConvolutionEngine::HeadSize targetSize = ConvolutionEngine::stringToHeadSize(sizeStr);
    bool switchSuccess = false;
    if (deps_.applySoftMuteForFilterSwitch) {
        deps_.applySoftMuteForFilterSwitch([&]() {
            std::lock_guard<std::mutex> cfLock(*deps_.crossfeed.mutex);
            auto* processor = crossfeedProcessor(deps_.crossfeed);
            if (!processor || !processor->isInitialized()) {
                return false;
            }
            switchSuccess = processor->switchHeadSize(targetSize);
            return switchSuccess;
        });
    } else {
        std::lock_guard<std::mutex> cfLock(*deps_.crossfeed.mutex);
        auto* processor = crossfeedProcessor(deps_.crossfeed);
        if (processor) {
            switchSuccess = processor->switchHeadSize(targetSize);
        }
    }

    if (!switchSuccess) {
        return buildErrorResponse(request, "CROSSFEED_SIZE_SWITCH_FAILED",
                                  "Failed to switch head size");
    }

    {
        std::lock_guard<std::mutex> cfLock(*deps_.crossfeed.mutex);
        if (deps_.crossfeed.resetStreamingState) {
            deps_.crossfeed.resetStreamingState();
        }
    }
    nlohmann::json data;
    data["head_size"] = ConvolutionEngine::headSizeToString(targetSize);
    return buildOkResponse(request, "", data);
}

std::string ControlPlane::handleDacList(const daemon_ipc::ZmqRequest& request) {
    if (!deps_.dacManager) {
        return buildErrorResponse(request, "IPC_INTERNAL_ERROR", "DAC manager unavailable");
    }
    return buildOkResponse(request, "", deps_.dacManager->buildDevicesJson());
}

std::string ControlPlane::handleDacStatus(const daemon_ipc::ZmqRequest& request) {
    if (!deps_.dacManager) {
        return buildErrorResponse(request, "IPC_INTERNAL_ERROR", "DAC manager unavailable");
    }
    nlohmann::json data = deps_.dacManager->buildStatusJson();
    if (deps_.currentOutputRate) {
        data["output_rate"] = deps_.currentOutputRate->load(std::memory_order_acquire);
    }
    return buildOkResponse(request, "", data);
}

std::string ControlPlane::handleDacSelect(const daemon_ipc::ZmqRequest& request) {
    if (!deps_.dacManager) {
        return buildErrorResponse(request, "IPC_INTERNAL_ERROR", "DAC manager unavailable");
    }
    if (!request.json || !request.json->contains("params") ||
        !(*request.json)["params"].contains("device")) {
        return buildErrorResponse(request, "IPC_INVALID_PARAMS", "Missing params.device field");
    }

    std::string targetDevice = (*request.json)["params"]["device"].get<std::string>();
    if (!deps_.dacManager->isValidDeviceName(targetDevice)) {
        return buildErrorResponse(request, "IPC_INVALID_PARAMS", "Invalid ALSA device name");
    }

    deps_.dacManager->requestDevice(targetDevice);

    if (deps_.dispatcher) {
        daemon_core::api::DeviceChangeRequested event;
        event.preferredDevice = targetDevice;
        event.mode = OutputMode::Usb;
        deps_.dispatcher->publish(event);
    }

    return buildOkResponse(request, "Preferred ALSA device updated",
                           deps_.dacManager->buildDevicesJson());
}

std::string ControlPlane::handleDacRescan(const daemon_ipc::ZmqRequest& request) {
    if (!deps_.dacManager) {
        return buildErrorResponse(request, "IPC_INTERNAL_ERROR", "DAC manager unavailable");
    }
    deps_.dacManager->requestRescan();
    return buildOkResponse(request, "DAC rescan scheduled", deps_.dacManager->buildDevicesJson());
}

std::string ControlPlane::handleOutputModeGet(const daemon_ipc::ZmqRequest& request) {
    nlohmann::json data;
    if (deps_.config) {
        data["mode"] = outputModeToString(deps_.config->output.mode);

        nlohmann::json modes = nlohmann::json::array();
        modes.push_back("usb");
        data["available_modes"] = modes;

        nlohmann::json options;
        options["usb"]["preferred_device"] = deps_.config->output.usb.preferredDevice;
        data["options"] = options;
    }
    return buildOkResponse(request, "", data);
}

std::string ControlPlane::handleOutputModeSet(const daemon_ipc::ZmqRequest& request) {
    if (!deps_.config || !deps_.dacManager) {
        return buildErrorResponse(request, "IPC_INTERNAL_ERROR", "Configuration unavailable");
    }
    if (!request.json || !request.json->contains("params") ||
        !(*request.json)["params"].is_object()) {
        return buildErrorResponse(request, "IPC_INVALID_PARAMS", "Missing params object");
    }

    const auto& params = (*request.json)["params"];
    std::string requestedMode = outputModeToString(deps_.config->output.mode);
    if (params.contains("mode") && params["mode"].is_string()) {
        requestedMode = params["mode"].get<std::string>();
    }
    std::string normalizedMode = normalizeOutputMode(requestedMode);
    if (!isSupportedOutputMode(normalizedMode)) {
        return buildErrorResponse(request, "ERR_UNSUPPORTED_MODE",
                                  "Output mode '" + requestedMode + "' is not supported");
    }

    std::string preferredDevice = deps_.config->output.usb.preferredDevice;
    if (params.contains("options") && params["options"].is_object()) {
        const auto& options = params["options"];
        if (options.contains("usb") && options["usb"].is_object()) {
            const auto& usb = options["usb"];
            if (usb.contains("preferred_device") && usb["preferred_device"].is_string()) {
                preferredDevice = usb["preferred_device"].get<std::string>();
            } else if (usb.contains("preferredDevice") && usb["preferredDevice"].is_string()) {
                preferredDevice = usb["preferredDevice"].get<std::string>();
            }
        }
    }

    if (preferredDevice.empty()) {
        preferredDevice = deps_.defaultAlsaDevice ? deps_.defaultAlsaDevice : "default";
    }

    if (!deps_.dacManager->isValidDeviceName(preferredDevice)) {
        return buildErrorResponse(request, "IPC_INVALID_PARAMS", "Invalid ALSA device name");
    }

    if (deps_.setPreferredOutputDevice) {
        deps_.setPreferredOutputDevice(*deps_.config, preferredDevice);
    }
    deps_.dacManager->requestDevice(preferredDevice);

    if (deps_.dispatcher) {
        daemon_core::api::DeviceChangeRequested event;
        event.preferredDevice = preferredDevice;
        event.mode = OutputMode::Usb;
        deps_.dispatcher->publish(event);
    }

    nlohmann::json options;
    options["usb"]["preferred_device"] = preferredDevice;

    nlohmann::json modes = nlohmann::json::array();
    modes.push_back("usb");

    nlohmann::json data;
    data["mode"] = outputModeToString(deps_.config->output.mode);
    data["available_modes"] = modes;
    data["options"] = options;
    data["devices"] = deps_.dacManager->buildDevicesJson();

    return buildOkResponse(request, "Output mode updated", data);
}

std::string ControlPlane::handlePhaseTypeGet(const daemon_ipc::ZmqRequest& request) {
    auto* up = upsampler(deps_);
    if (!up) {
        return buildErrorResponse(request, "IPC_INVALID_COMMAND", "Upsampler not initialized");
    }
    PhaseType pt = up->getPhaseType();
    std::string ptStr = (pt == PhaseType::Minimum) ? "minimum" : "linear";
    nlohmann::json data;
    data["phase_type"] = ptStr;
    return buildOkResponse(request, "", data);
}

std::string ControlPlane::handlePhaseTypeSet(const daemon_ipc::ZmqRequest& request) {
    std::string phaseStr = request.payload;
    if (phaseStr.empty()) {
        return buildErrorResponse(request, "IPC_INVALID_PARAMS",
                                  "Invalid phase type (use 'minimum' or 'linear')");
    }

    auto* up = upsampler(deps_);
    if (!up) {
        return buildErrorResponse(request, "IPC_INVALID_COMMAND", "Upsampler not initialized");
    }
    if (phaseStr != "minimum" && phaseStr != "linear") {
        return buildErrorResponse(request, "IPC_INVALID_PARAMS",
                                  "Invalid phase type (use 'minimum' or 'linear')");
    }

    PhaseType newPhase = (phaseStr == "minimum") ? PhaseType::Minimum : PhaseType::Linear;
    PhaseType oldPhase = up->getPhaseType();
    bool disablePartitionForLinear = (newPhase == PhaseType::Linear && deps_.config &&
                                      deps_.config->partitionedConvolution.enabled);

    if (oldPhase == newPhase) {
        return buildOkResponse(request, "Phase type already " + phaseStr);
    }

    bool switchSuccess = false;
    auto switchFunc = [&]() {
        if (disablePartitionForLinear && deps_.config) {
            std::cout << "[Partition] Linear phase selected, disabling low-latency partitioned "
                         "convolution."
                      << '\n';
            deps_.config->partitionedConvolution.enabled = false;
            up->setPartitionedConvolutionConfig(deps_.config->partitionedConvolution);
            if (deps_.reinitializeStreamingForLegacyMode &&
                !deps_.reinitializeStreamingForLegacyMode()) {
                deps_.config->partitionedConvolution.enabled = true;
                up->setPartitionedConvolutionConfig(deps_.config->partitionedConvolution);
                return false;
            }
        }

        switchSuccess = up->switchPhaseType(newPhase);
        if (switchSuccess && deps_.activePhaseType) {
            *deps_.activePhaseType = newPhase;
            if (deps_.refreshHeadroom) {
                deps_.refreshHeadroom("phase switch");
            }
            if (deps_.config && deps_.config->eqEnabled && !deps_.config->eqProfilePath.empty()) {
                EQ::EqProfile eqProfile;
                if (EQ::parseEqFile(deps_.config->eqProfilePath, eqProfile)) {
                    size_t filterFftSize = up->getFilterFftSize();
                    size_t fullFftSize = up->getFullFftSize();
                    double outputSampleRate =
                        static_cast<double>(*deps_.inputSampleRate) * deps_.config->upsampleRatio;
                    auto eqMagnitude = EQ::computeEqMagnitudeForFft(filterFftSize, fullFftSize,
                                                                    outputSampleRate, eqProfile);
                    if (up->applyEqMagnitude(eqMagnitude)) {
                        std::cout << "ZeroMQ: EQ re-applied with " << phaseStr << " phase" << '\n';
                    } else {
                        LOG_WARN("ZeroMQ: EQ re-apply failed");
                    }
                } else {
                    LOG_WARN("ZeroMQ: Failed to parse EQ profile: {}", deps_.config->eqProfilePath);
                }
            }
        }
        return switchSuccess;
    };

    if (deps_.applySoftMuteForFilterSwitch) {
        deps_.applySoftMuteForFilterSwitch(switchFunc);
    } else {
        switchFunc();
    }

    if (!switchSuccess) {
        return buildErrorResponse(request, "IPC_PROTOCOL_ERROR", "Failed to switch phase type");
    }

    return buildOkResponse(request, "Phase type set to " + phaseStr);
}

}  // namespace daemon_control
