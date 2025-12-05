#include "daemon/rtp/rtp_engine_coordinator.h"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

using namespace std::chrono_literals;

namespace {

void send_rtp_packet(uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    ASSERT_GE(fd, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    ASSERT_EQ(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr), 1);

    uint8_t packet[12] = {};
    packet[0] = 0x80;        // Version 2
    packet[1] = 127 & 0x7F;  // Payload type 127

    ssize_t sent =
        ::sendto(fd, packet, sizeof(packet), 0, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    ASSERT_EQ(sent, static_cast<ssize_t>(sizeof(packet)));
    ::close(fd);
}

}  // namespace

TEST(RtpDiscovery, ReportsListeningPort) {
    constexpr uint16_t kListenPort = 49000;

    AppConfig config;
    config.rtp.enabled = true;
    config.rtp.autoStart = false;
    config.rtp.discoveryPorts = {kListenPort};
    config.rtp.discoveryScanDurationMs = 500;
    config.rtp.discoveryEnableMulticast = false;
    config.rtp.discoveryEnableUnicast = true;
    config.rtp.sampleRate = 44100;
    config.rtp.channels = 2;
    config.rtp.payloadType = 127;

    std::atomic<bool> running{true};
    std::atomic<int> inputRate{44100};
    std::atomic<int> outputRate{705600};
    std::atomic<bool> alsaReconfig{false};

    rtp_engine::RtpEngineCoordinator::Dependencies deps{};
    deps.config = &config;
    deps.runningFlag = &running;
    deps.currentInputRate = &inputRate;
    deps.currentOutputRate = &outputRate;
    deps.alsaReconfigureNeeded = &alsaReconfig;
    deps.handleRateChange = [](int) { return true; };
    deps.isUpsamplerReady = [] { return true; };
    deps.isMultiRateEnabled = [] { return true; };
    deps.getUpsampleRatio = [] { return 16; };
    deps.getInputSampleRate = [] { return 44100; };
    deps.processInterleaved = [](const float*, size_t, uint32_t) {};
    deps.ptpProvider = [] { return Network::PtpSyncState{}; };
    deps.telemetry = [](const Network::SessionMetrics&) {};

    rtp_engine::RtpEngineCoordinator coordinator(deps);

    std::thread sender([&]() {
        std::this_thread::sleep_for(50ms);
        send_rtp_packet(kListenPort);
    });

    nlohmann::json result = rtp_engine::RtpEngineCoordinatorTestHook::runDiscoveryScan(coordinator);
    sender.join();

    ASSERT_EQ(result["status"], "ok");
    auto streams = result["data"]["streams"];
    ASSERT_FALSE(streams.empty());
    EXPECT_EQ(streams[0]["port"].get<uint16_t>(), kListenPort);
}
