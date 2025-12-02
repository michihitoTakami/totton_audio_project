#include "daemon/zmq_server.h"

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <zmq.hpp>

namespace {

std::string make_ipc_endpoint() {
    static std::atomic<int> counter{0};
    std::ostringstream oss;
    oss << "ipc:///tmp/gpu_os_zmq_server_test_" << ::getpid() << "_" << counter++ << ".sock";
    return oss.str();
}

class ZmqServerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        endpoint_ = make_ipc_endpoint();
        server_ = std::make_unique<daemon_ipc::ZmqCommandServer>(endpoint_);
    }

    std::string endpoint_;
    std::unique_ptr<daemon_ipc::ZmqCommandServer> server_;
};

std::string read_message(zmq::socket_t& socket) {
    zmq::message_t reply;
    auto result = socket.recv(reply, zmq::recv_flags::none);
    if (!result) {
        return {};
    }
    return std::string(static_cast<char*>(reply.data()), reply.size());
}

}  // namespace

TEST_F(ZmqServerTest, DispatchesRawAndJsonCommands) {
    server_->registerCommand("PING", [](const daemon_ipc::ZmqRequest& request) {
        (void)request;
        return std::string("PONG");
    });
    server_->registerCommand("HELLO", [](const daemon_ipc::ZmqRequest& request) {
        nlohmann::json resp;
        resp["status"] = "ok";
        std::string name;
        if (request.json && request.json->contains("params")) {
            const auto& params = (*request.json)["params"];
            if (params.is_object()) {
                name = params.value("name", "");
            }
        }
        resp["message"] = "Hello " + (name.empty() ? std::string("world") : name);
        return resp.dump();
    });
    ASSERT_TRUE(server_->start());

    zmq::context_t ctx(1);
    zmq::socket_t req(ctx, zmq::socket_type::req);
    req.connect(endpoint_);

    req.send(zmq::buffer("PING"), zmq::send_flags::none);
    EXPECT_EQ(read_message(req), "PONG");

    nlohmann::json cmd;
    cmd["cmd"] = "HELLO";
    cmd["params"]["name"] = "ZMQ";
    req.send(zmq::buffer(cmd.dump()), zmq::send_flags::none);
    auto reply = read_message(req);
    ASSERT_FALSE(reply.empty());
    auto respJson = nlohmann::json::parse(reply);
    EXPECT_EQ(respJson["status"], "ok");
    EXPECT_EQ(respJson["message"], "Hello ZMQ");
}

TEST_F(ZmqServerTest, PublishesEvents) {
    ASSERT_TRUE(server_->start());

    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.set(zmq::sockopt::subscribe, "");
    sub.set(zmq::sockopt::rcvtimeo, 200);
    sub.connect(server_->pubEndpoint());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const std::string payload = "{\"hello\":true}";
    ASSERT_TRUE(server_->publish(payload));

    zmq::message_t msg;
    bool received = false;
    for (int i = 0; i < 5 && !received; ++i) {
        auto result = sub.recv(msg, zmq::recv_flags::none);
        if (result) {
            received = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    ASSERT_TRUE(received);
    EXPECT_EQ(std::string(static_cast<char*>(msg.data()), msg.size()), payload);
}

TEST_F(ZmqServerTest, ReturnsJsonErrorOnParseFailure) {
    ASSERT_TRUE(server_->start());

    zmq::context_t ctx(1);
    zmq::socket_t req(ctx, zmq::socket_type::req);
    req.connect(endpoint_);

    req.send(zmq::buffer("{invalid"), zmq::send_flags::none);
    auto reply = read_message(req);
    ASSERT_FALSE(reply.empty());

    auto resp = nlohmann::json::parse(reply);
    EXPECT_EQ(resp["status"], "error");
    EXPECT_EQ(resp["error_code"], "IPC_PROTOCOL_ERROR");
}
