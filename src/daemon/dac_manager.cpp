#include "daemon/dac_manager.h"

#include "logging/logger.h"

#include <algorithm>
#include <alsa/asoundlib.h>
#include <cctype>
#include <chrono>
#include <cstring>
#include <iostream>
#include <utility>

namespace dac {

DacManager::DacManager(Dependencies deps) : deps_(std::move(deps)) {}

DacManager::~DacManager() {
    stop();
}

bool DacManager::isValidDeviceName(const std::string& device) const {
    if (device.empty())
        return false;
    if (device == "default")
        return true;

    auto checkPrefix = [](const std::string& value, const std::string& prefix) -> bool {
        if (value.rfind(prefix, 0) != 0)
            return false;
        const std::string rest = value.substr(prefix.size());
        if (rest.empty())
            return false;
        for (char c : rest) {
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == ',' || c == '_' ||
                  c == ':')) {
                return false;
            }
        }
        return true;
    };

    if (checkPrefix(device, "hw:") || checkPrefix(device, "plughw:") ||
        checkPrefix(device, "sysdefault:")) {
        return true;
    }
    return false;
}

bool DacManager::isRunning() const {
    return !deps_.runningFlag || deps_.runningFlag->load(std::memory_order_acquire);
}

void DacManager::addDeviceEntry(std::vector<DacDeviceRuntimeInfo>& list,
                                std::unordered_set<std::string>& seen, const std::string& id,
                                const std::string& card, const std::string& name,
                                const std::string& description) const {
    if (id.empty())
        return;
    if (!seen.insert(id).second)
        return;
    list.push_back({id, card, name, description});
}

std::vector<DacDeviceRuntimeInfo> DacManager::enumerateDevices() const {
    std::vector<DacDeviceRuntimeInfo> devices;
    std::unordered_set<std::string> seen;
    addDeviceEntry(devices, seen, "default", "System", "Default Output",
                   "System default ALSA route");

    int card = -1;
    if (snd_card_next(&card) < 0) {
        return devices;
    }

    while (card >= 0) {
        std::string cardNum = std::to_string(card);
        std::string cardBaseId = "hw:" + cardNum;
        snd_ctl_t* handle = nullptr;
        if (snd_ctl_open(&handle, cardBaseId.c_str(), 0) < 0) {
            snd_card_next(&card);
            continue;
        }

        snd_ctl_card_info_t* cardInfo;
        snd_ctl_card_info_alloca(&cardInfo);
        if (snd_ctl_card_info(handle, cardInfo) < 0) {
            snd_ctl_close(handle);
            snd_card_next(&card);
            continue;
        }

        std::string cardName = snd_ctl_card_info_get_name(cardInfo);
        std::string cardLong = snd_ctl_card_info_get_longname(cardInfo);
        std::string cardId = snd_ctl_card_info_get_id(cardInfo);
        if (cardLong.empty())
            cardLong = cardName;

        addDeviceEntry(devices, seen, cardBaseId, cardName, "hw", cardLong);
        addDeviceEntry(devices, seen, "plughw:" + cardNum, cardName, "plughw", cardLong);
        if (!cardId.empty()) {
            addDeviceEntry(devices, seen, "hw:" + cardId, cardName, "hw", cardLong);
            addDeviceEntry(devices, seen, "plughw:" + cardId, cardName, "plughw", cardLong);
        }

        int device = -1;
        while (snd_ctl_pcm_next_device(handle, &device) >= 0 && device >= 0) {
            snd_pcm_info_t* pcmInfo;
            snd_pcm_info_alloca(&pcmInfo);
            snd_pcm_info_set_device(pcmInfo, device);
            snd_pcm_info_set_subdevice(pcmInfo, 0);
            snd_pcm_info_set_stream(pcmInfo, SND_PCM_STREAM_PLAYBACK);
            if (snd_ctl_pcm_info(handle, pcmInfo) < 0)
                continue;

            std::string pcmName = snd_pcm_info_get_name(pcmInfo);
            std::string desc = cardName;
            if (!pcmName.empty())
                desc += " / " + pcmName;

            std::string deviceSuffix = "," + std::to_string(device);
            addDeviceEntry(devices, seen, cardBaseId + deviceSuffix, cardName, pcmName, desc);
            addDeviceEntry(devices, seen, "plughw:" + cardNum + deviceSuffix, cardName, pcmName,
                           desc);
            if (!cardId.empty()) {
                addDeviceEntry(devices, seen, "hw:" + cardId + deviceSuffix, cardName, pcmName,
                               desc);
                addDeviceEntry(devices, seen, "plughw:" + cardId + deviceSuffix, cardName, pcmName,
                               desc);
            }
        }

        snd_ctl_close(handle);
        snd_card_next(&card);
    }

    return devices;
}

std::string DacManager::pickPreferredDeviceLocked(
    const std::vector<DacDeviceRuntimeInfo>& devices) const {
    if (!requestedDevice_.empty()) {
        auto it = std::find_if(devices.begin(), devices.end(),
                               [&](const auto& info) { return info.id == requestedDevice_; });
        if (it != devices.end()) {
            return requestedDevice_;
        }
    }

    for (const auto& dev : devices) {
        if (dev.id == "default")
            continue;
        return dev.id;
    }
    return requestedDevice_;
}

nlohmann::json DacManager::capabilityToJson(const DacCapability::Capability& cap) const {
    nlohmann::json capJson;
    capJson["device"] = cap.deviceName;
    capJson["is_valid"] = cap.isValid;
    capJson["min_rate"] = cap.minSampleRate;
    capJson["max_rate"] = cap.maxSampleRate;
    capJson["max_channels"] = cap.maxChannels;
    capJson["supported_rates"] = cap.supportedRates;
    if (!cap.errorMessage.empty()) {
        capJson["error_message"] = cap.errorMessage;
    }
    if (cap.alsaErrno != 0) {
        capJson["alsa_errno"] = cap.alsaErrno;
    }
    return capJson;
}

nlohmann::json DacManager::buildDevicesJsonLocked() {
    nlohmann::json devicesJson = nlohmann::json::array();
    for (const auto& dev : devices_) {
        nlohmann::json entry;
        entry["id"] = dev.id;
        entry["card"] = dev.card;
        entry["name"] = dev.name;
        entry["description"] = dev.description;
        entry["is_requested"] = (dev.id == requestedDevice_);
        entry["is_selected"] = (dev.id == selectedDevice_);
        entry["is_active"] = (dev.id == activePlaybackDevice_);
        devicesJson.push_back(entry);
    }

    nlohmann::json root;
    root["devices"] = devicesJson;
    root["requested_device"] = requestedDevice_;
    root["selected_device"] = selectedDevice_;
    root["active_device"] = activePlaybackDevice_;
    root["change_pending"] = changePending_.load(std::memory_order_acquire);
    return root;
}

void DacManager::setSelectedDeviceLocked(const std::string& device, const std::string& reason) {
    if (selectedDevice_ == device)
        return;
    selectedDevice_ = device;
    changePending_.store(true, std::memory_order_release);
    cv_.notify_all();
    std::cout << "[DAC] Selected ALSA device: " << (device.empty() ? "<none>" : device) << " ("
              << reason << ")" << std::endl;
}

void DacManager::updateCapabilityAndNotify(const std::string& device, const std::string& reason) {
    if (device.empty())
        return;
    auto cap = DacCapability::scan(device);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        activeCapability_ = cap;
    }
    nlohmann::json data = buildDevicesJson();
    data["capability"] = capabilityToJson(cap);
    data["reason"] = reason;
    emitEvent("dac_selected", data);
}

void DacManager::emitEvent(const std::string& type, const nlohmann::json& data) {
    if (!deps_.timestampProvider) {
        return;
    }

    nlohmann::json payload;
    payload["type"] = type;
    payload["timestamp"] = deps_.timestampProvider();
    if (!data.is_null() && !data.empty()) {
        payload["data"] = data;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        lastEvent_ = payload;
    }

    if (deps_.eventPublisher) {
        deps_.eventPublisher(payload);
    }
}

void DacManager::initialize() {
    stop();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        requestedDevice_ = (deps_.config && !deps_.config->alsaDevice.empty())
                               ? deps_.config->alsaDevice
                               : deps_.defaultDevice;
        selectedDevice_.clear();
        activePlaybackDevice_.clear();
        activeCapability_ = DacCapability::Capability();
        lastEvent_ = nlohmann::json::object();
        changePending_.store(false, std::memory_order_release);
    }

    auto snapshot = enumerateDevices();
    std::string initialSelection;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        devices_ = snapshot;
        initialSelection = pickPreferredDeviceLocked(devices_);
        if (!initialSelection.empty()) {
            selectedDevice_ = initialSelection;
            changePending_.store(true, std::memory_order_release);
        }
    }

    emitEvent("dac_devices_updated", buildDevicesJson());
    if (!initialSelection.empty()) {
        updateCapabilityAndNotify(initialSelection, "initial");
    }
}

void DacManager::start() {
    if (monitorRunning_.exchange(true))
        return;
    monitorThread_ = std::thread(&DacManager::monitorLoop, this);
}

void DacManager::stop() {
    bool wasRunning = monitorRunning_.exchange(false);
    forceRescan_.store(true, std::memory_order_release);
    cv_.notify_all();
    if (wasRunning && monitorThread_.joinable()) {
        monitorThread_.join();
    }
}

std::string DacManager::waitForSelection() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() {
        return !isRunning() || !monitorRunning_.load(std::memory_order_acquire) ||
               !selectedDevice_.empty();
    });
    return selectedDevice_;
}

std::string DacManager::getSelectedDevice() {
    std::lock_guard<std::mutex> lock(mutex_);
    return selectedDevice_;
}

std::optional<std::string> DacManager::consumePendingChange() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!changePending_.exchange(false, std::memory_order_acq_rel)) {
        return std::nullopt;
    }
    return selectedDevice_;
}

void DacManager::markActiveDevice(const std::string& device, bool active) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active) {
        activePlaybackDevice_ = device;
    } else if (activePlaybackDevice_ == device) {
        activePlaybackDevice_.clear();
    }
}

void DacManager::requestDevice(const std::string& device) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        requestedDevice_ = device;
        if (deps_.config) {
            deps_.config->alsaDevice = device;
        }
        std::string candidate = pickPreferredDeviceLocked(devices_);
        setSelectedDeviceLocked(candidate, "manual_select");
    }
    forceRescan_.store(true, std::memory_order_release);
}

void DacManager::requestRescan() {
    forceRescan_.store(true, std::memory_order_release);
}

nlohmann::json DacManager::buildDevicesJson() {
    std::lock_guard<std::mutex> lock(mutex_);
    return buildDevicesJsonLocked();
}

nlohmann::json DacManager::buildStatusJson() {
    std::lock_guard<std::mutex> lock(mutex_);
    nlohmann::json data = buildDevicesJsonLocked();
    data["capability"] = capabilityToJson(activeCapability_);
    if (!lastEvent_.is_null() && !lastEvent_.empty()) {
        data["last_event"] = lastEvent_;
    }
    return data;
}

nlohmann::json DacManager::buildStatsSummaryJson() {
    std::lock_guard<std::mutex> lock(mutex_);
    nlohmann::json dacJson;
    dacJson["requested_device"] = requestedDevice_;
    dacJson["selected_device"] = selectedDevice_;
    dacJson["active_device"] = activePlaybackDevice_;
    dacJson["device_count"] = devices_.size();
    return dacJson;
}

void DacManager::monitorLoop() {
    while (monitorRunning_.load(std::memory_order_acquire) && isRunning()) {
        bool rescan = forceRescan_.exchange(false, std::memory_order_acq_rel);
        auto devices = enumerateDevices();
        std::string newlySelected;
        bool devicesChanged = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (devices != devices_) {
                devices_ = devices;
                devicesChanged = true;
            }
            std::string candidate = pickPreferredDeviceLocked(devices_);
            if (candidate != selectedDevice_) {
                setSelectedDeviceLocked(candidate, "monitor");
                newlySelected = candidate;
            } else if (rescan && !candidate.empty()) {
                newlySelected = candidate;
            }
        }

        if (devicesChanged) {
            emitEvent("dac_devices_updated", buildDevicesJson());
        }
        if (!newlySelected.empty()) {
            updateCapabilityAndNotify(newlySelected, rescan ? "manual_rescan" : "auto");
        }

        for (int i = 0; i < 10 && monitorRunning_.load(std::memory_order_acquire) && isRunning();
             ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (forceRescan_.load(std::memory_order_acquire)) {
                break;
            }
        }
    }
}

}  // namespace dac
