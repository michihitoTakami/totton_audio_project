#pragma once

#include "core/config_loader.h"
#include "io/dac_capability.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace dac {

struct DacDeviceRuntimeInfo {
    std::string id;
    std::string card;
    std::string name;
    std::string description;

    bool operator==(const DacDeviceRuntimeInfo& other) const {
        return id == other.id && card == other.card && name == other.name &&
               description == other.description;
    }
};

class DacManager {
   public:
    struct Dependencies {
        AppConfig* config = nullptr;
        std::atomic<bool>* runningFlag = nullptr;
        std::function<int64_t()> timestampProvider;
        std::function<void(const nlohmann::json&)> eventPublisher;
        std::string defaultDevice;
    };

    explicit DacManager(Dependencies deps);
    ~DacManager();

    void initialize();
    void start();
    void stop();

    std::string waitForSelection();
    std::string getSelectedDevice();
    std::optional<std::string> consumePendingChange();
    void markActiveDevice(const std::string& device, bool active);

    void requestDevice(const std::string& device);
    void requestRescan();

    nlohmann::json buildDevicesJson();
    nlohmann::json buildStatusJson();
    nlohmann::json buildStatsSummaryJson();

    bool isValidDeviceName(const std::string& device) const;
    void setEventPublisher(std::function<void(const nlohmann::json&)> eventPublisher);

   private:
    bool isRunning() const;
    void addDeviceEntry(std::vector<DacDeviceRuntimeInfo>& list,
                        std::unordered_set<std::string>& seen, const std::string& id,
                        const std::string& card, const std::string& name,
                        const std::string& description) const;
    std::vector<DacDeviceRuntimeInfo> enumerateDevices() const;
    std::string pickPreferredDeviceLocked(const std::vector<DacDeviceRuntimeInfo>& devices) const;
    nlohmann::json capabilityToJson(const DacCapability::Capability& cap) const;
    nlohmann::json buildDevicesJsonLocked();
    void setSelectedDeviceLocked(const std::string& device, const std::string& reason);
    void updateCapabilityAndNotify(const std::string& device, const std::string& reason);
    void emitEvent(const std::string& type, const nlohmann::json& data);
    void monitorLoop();

    Dependencies deps_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<DacDeviceRuntimeInfo> devices_;
    std::string requestedDevice_;
    std::string selectedDevice_;
    std::string activePlaybackDevice_;
    DacCapability::Capability activeCapability_;
    std::atomic<bool> changePending_{false};
    std::atomic<bool> monitorRunning_{false};
    std::atomic<bool> forceRescan_{false};
    std::thread monitorThread_;
    nlohmann::json lastEvent_;
};

}  // namespace dac
