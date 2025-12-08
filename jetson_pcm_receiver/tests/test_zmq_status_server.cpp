#include "status_tracker.h"
#include "zmq_status_server.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <gtest/gtest.h>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <zmq.hpp>

namespace {

std::string makeEndpoint() {
    std::ostringstream oss;
    static std::atomic<int> counter{0};
    oss << "ipc:///tmp/jetson_pcm_zmq_status_" << ::getpid() << "_" << counter++ << ".sock";
    return oss.str();
}

class ZmqStatusServerTest : public ::testing::Test {
   protected:
    void TearDown() override {
        server.stop();
    }

    void startServer(int pubIntervalMs = 0, const std::string& token = "") {
        options.enabled = true;
        options.endpoint = makeEndpoint();
        options.publishIntervalMs = pubIntervalMs;
        options.token = token;
        ASSERT_TRUE(server.start(options));
    }

    std::string sendCommand(const nlohmann::json& payload) {
        zmq::context_t ctx(1);
        zmq::socket_t req(ctx, zmq::socket_type::req);
        req.connect(server.endpoint());
        std::string raw = payload.dump();
        req.send(zmq::buffer(raw), zmq::send_flags::none);
        zmq::message_t reply;
        auto res = req.recv(reply, zmq::recv_flags::none);
        if (!res) {
            return {};
        }
        return std::string(static_cast<char*>(reply.data()), reply.size());
    }

    StatusTracker status;
    PcmStreamConfig config{};
    std::mutex configMutex;
    std::atomic_bool stopFlag{false};
    ZmqStatusServer server{status, config, configMutex, stopFlag};
    ZmqStatusServer::Options options{};
};

}  // namespace

TEST_F(ZmqStatusServerTest, ReturnsStatusSnapshot) {
    startServer();

    PcmHeader hdr{};
    std::memcpy(hdr.magic, "PCMA", 4);
    hdr.version = 1;
    hdr.sample_rate = 48000;
    hdr.channels = 2;
    hdr.format = 1;

    status.setListening(5555);
    status.setClientConnected(true);
    status.setHeader(hdr);
    status.updateRingConfig(64, 48);
    status.updateRingBuffer(32, 48, 1);
    status.incrementXrun();
    status.setDisconnectReason("format_changed");

    nlohmann::json cmd;
    cmd["cmd"] = "STATUS";
    auto reply = sendCommand(cmd);
    auto resp = nlohmann::json::parse(reply);
    ASSERT_EQ(resp["status"], "ok");
    auto data = resp["data"];
    EXPECT_TRUE(data["listening"]);
    EXPECT_EQ(data["bound_port"], 5555);
    EXPECT_TRUE(data["client_connected"]);
    EXPECT_EQ(data["ring_buffer_frames"], 64);
    EXPECT_EQ(data["buffered_frames"], 32);
    EXPECT_EQ(data["dropped_frames"], 1);
    EXPECT_EQ(data["xrun_count"], 1);
    EXPECT_EQ(data["last_header"]["sample_rate"], 48000);
    EXPECT_EQ(data["disconnect_reason"], "format_changed");
}

TEST_F(ZmqStatusServerTest, UpdatesRingConfigFromSetCache) {
    startServer();
    config.ringBufferFrames = 8;
    config.watermarkFrames = 0;

    nlohmann::json cmd;
    cmd["cmd"] = "SET_CACHE";
    cmd["params"]["ring_buffer_frames"] = 512;
    cmd["params"]["watermark_frames"] = 256;

    auto reply = sendCommand(cmd);
    auto resp = nlohmann::json::parse(reply);
    ASSERT_EQ(resp["status"], "ok");
    EXPECT_EQ(resp["data"]["ring_buffer_frames"], 512);
    EXPECT_EQ(resp["data"]["watermark_frames"], 256);

    std::lock_guard<std::mutex> lock(configMutex);
    EXPECT_EQ(config.ringBufferFrames, 512u);
    EXPECT_EQ(config.watermarkFrames, 256u);

    auto snap = status.snapshot();
    EXPECT_EQ(snap.ring.configuredFrames, 512u);
    EXPECT_EQ(snap.ring.watermarkFrames, 256u);
}

TEST_F(ZmqStatusServerTest, RejectsInvalidToken) {
    startServer(0, "secret");

    nlohmann::json cmd;
    cmd["cmd"] = "STATUS";
    auto reply = sendCommand(cmd);
    auto resp = nlohmann::json::parse(reply);
    ASSERT_EQ(resp["status"], "error");
    EXPECT_EQ(resp["error_code"], "IPC_UNAUTHORIZED");

    cmd["token"] = "secret";
    reply = sendCommand(cmd);
    resp = nlohmann::json::parse(reply);
    EXPECT_EQ(resp["status"], "ok");
}

TEST_F(ZmqStatusServerTest, RestartSetsStopFlagAndPublishes) {
    startServer(50);

    nlohmann::json cmd;
    cmd["cmd"] = "RESTART";
    auto reply = sendCommand(cmd);
    auto resp = nlohmann::json::parse(reply);
    ASSERT_EQ(resp["status"], "ok");
    EXPECT_TRUE(stopFlag.load());
    EXPECT_TRUE(server.restartRequested());

    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.set(zmq::sockopt::subscribe, "");
    sub.set(zmq::sockopt::rcvtimeo, 500);
    sub.connect(server.pubEndpoint());

    bool received = false;
    for (int i = 0; i < 5 && !received; ++i) {
        zmq::message_t msg;
        if (sub.recv(msg, zmq::recv_flags::none)) {
            auto payload =
                nlohmann::json::parse(std::string(static_cast<char*>(msg.data()), msg.size()));
            if (payload.contains("event") && payload["event"] == "status") {
                received = true;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    EXPECT_TRUE(received);
}

TEST_F(ZmqStatusServerTest, PublishesHeaderChangeEvent) {
    startServer(0);

    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.set(zmq::sockopt::subscribe, "");
    sub.set(zmq::sockopt::rcvtimeo, 500);
    sub.connect(server.pubEndpoint());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    PcmHeader hdr{};
    std::memcpy(hdr.magic, "PCMA", 4);
    hdr.version = 1;
    hdr.sample_rate = 96000;
    hdr.channels = 2;
    hdr.format = 4;

    ASSERT_TRUE(server.publishHeaderChange(hdr));

    zmq::message_t msg;
    ASSERT_TRUE(sub.recv(msg, zmq::recv_flags::none));
    auto payload = nlohmann::json::parse(
        std::string(static_cast<char*>(msg.data()), static_cast<std::size_t>(msg.size())));
    EXPECT_EQ(payload["event"], "pcm_header_changed");
    EXPECT_EQ(payload["header"]["sample_rate"], 96000);
    EXPECT_EQ(payload["header"]["channels"], 2);
    EXPECT_EQ(payload["header"]["format"], 4);
    EXPECT_FALSE(payload.contains("previous_header"));
}
