/**
 * @file test_zeromq_interface.cpp
 * @brief Unit tests for ZeroMQ communication interface
 */

#include "network/zeromq_interface.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace ZMQComm;

// Atomic counter for generating unique socket paths across tests
static std::atomic<int> g_socketCounter{0};

class ZMQInterfaceTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {
        // Clean up any socket files created during the test
        for (const auto& path : socketsToCleanup_) {
            std::remove(path.c_str());
            std::remove((path + ".pub").c_str());
        }
        socketsToCleanup_.clear();
    }

    // Generate a unique IPC endpoint and register for cleanup
    std::string makeUniqueEndpoint(const std::string& baseName) {
        int id = g_socketCounter.fetch_add(1);
        std::string socketPath = "/tmp/zmq_test_" + baseName + "_" + std::to_string(id) + ".sock";
        // Remove any existing socket file before use
        std::remove(socketPath.c_str());
        std::remove((socketPath + ".pub").c_str());
        socketsToCleanup_.push_back(socketPath);
        return "ipc://" + socketPath;
    }

   private:
    std::vector<std::string> socketsToCleanup_;
};

// ============================================================
// String Conversion Tests
// ============================================================

TEST_F(ZMQInterfaceTest, CommandTypeToString) {
    EXPECT_STREQ(commandTypeToString(CommandType::LOAD_IR), "LOAD_IR");
    EXPECT_STREQ(commandTypeToString(CommandType::SET_GAIN), "SET_GAIN");
    EXPECT_STREQ(commandTypeToString(CommandType::SOFT_RESET), "SOFT_RESET");
    EXPECT_STREQ(commandTypeToString(CommandType::GET_STATUS), "GET_STATUS");
    EXPECT_STREQ(commandTypeToString(CommandType::SWITCH_RATE), "SWITCH_RATE");
    EXPECT_STREQ(commandTypeToString(CommandType::APPLY_EQ), "APPLY_EQ");
    EXPECT_STREQ(commandTypeToString(CommandType::RESTORE_EQ), "RESTORE_EQ");
    EXPECT_STREQ(commandTypeToString(CommandType::SHUTDOWN), "SHUTDOWN");
    EXPECT_STREQ(commandTypeToString(CommandType::OUTPUT_MODE_GET), "OUTPUT_MODE_GET");
    EXPECT_STREQ(commandTypeToString(CommandType::OUTPUT_MODE_SET), "OUTPUT_MODE_SET");
}

TEST_F(ZMQInterfaceTest, StringToCommandType) {
    EXPECT_EQ(stringToCommandType("LOAD_IR"), CommandType::LOAD_IR);
    EXPECT_EQ(stringToCommandType("SET_GAIN"), CommandType::SET_GAIN);
    EXPECT_EQ(stringToCommandType("SOFT_RESET"), CommandType::SOFT_RESET);
    EXPECT_EQ(stringToCommandType("GET_STATUS"), CommandType::GET_STATUS);
    EXPECT_EQ(stringToCommandType("SWITCH_RATE"), CommandType::SWITCH_RATE);
    EXPECT_EQ(stringToCommandType("APPLY_EQ"), CommandType::APPLY_EQ);
    EXPECT_EQ(stringToCommandType("RESTORE_EQ"), CommandType::RESTORE_EQ);
    EXPECT_EQ(stringToCommandType("SHUTDOWN"), CommandType::SHUTDOWN);
    EXPECT_EQ(stringToCommandType("OUTPUT_MODE_GET"), CommandType::OUTPUT_MODE_GET);
    EXPECT_EQ(stringToCommandType("OUTPUT_MODE_SET"), CommandType::OUTPUT_MODE_SET);
}

TEST_F(ZMQInterfaceTest, StringToCommandTypeInvalid) {
    EXPECT_THROW(stringToCommandType("INVALID"), std::invalid_argument);
    EXPECT_THROW(stringToCommandType(""), std::invalid_argument);
}

TEST_F(ZMQInterfaceTest, ResponseStatusToString) {
    EXPECT_STREQ(responseStatusToString(ResponseStatus::OK), "ok");
    EXPECT_STREQ(responseStatusToString(ResponseStatus::ERROR), "error");
    EXPECT_STREQ(responseStatusToString(ResponseStatus::INVALID_COMMAND), "invalid_command");
    EXPECT_STREQ(responseStatusToString(ResponseStatus::INVALID_PARAMS), "invalid_params");
}

// ============================================================
// JSON Utility Tests
// ============================================================

TEST_F(ZMQInterfaceTest, BuildCommandNoParams) {
    std::string json = JSON::buildCommand(CommandType::SOFT_RESET, "");
    EXPECT_NE(json.find("\"cmd\":\"SOFT_RESET\""), std::string::npos);
}

TEST_F(ZMQInterfaceTest, BuildCommandWithParams) {
    std::string json = JSON::buildCommand(CommandType::LOAD_IR, R"({"path":"/test.bin"})");
    EXPECT_NE(json.find("\"cmd\":\"LOAD_IR\""), std::string::npos);
    EXPECT_NE(json.find("\"path\":\"/test.bin\""), std::string::npos);
}

TEST_F(ZMQInterfaceTest, ParseCommandValid) {
    std::string json = R"({"cmd":"LOAD_IR","params":{"path":"/test.bin"}})";
    CommandType type;
    std::string params;

    EXPECT_TRUE(JSON::parseCommand(json, type, params));
    EXPECT_EQ(type, CommandType::LOAD_IR);
    EXPECT_NE(params.find("/test.bin"), std::string::npos);
}

TEST_F(ZMQInterfaceTest, ParseCommandInvalidJson) {
    std::string json = "not valid json";
    CommandType type;
    std::string params;

    EXPECT_FALSE(JSON::parseCommand(json, type, params));
}

TEST_F(ZMQInterfaceTest, ParseCommandMissingCmd) {
    std::string json = R"({"params":{}})";
    CommandType type;
    std::string params;

    EXPECT_FALSE(JSON::parseCommand(json, type, params));
}

TEST_F(ZMQInterfaceTest, BuildResponse) {
    std::string json = JSON::buildResponse(ResponseStatus::OK, "Success", R"({"loaded":true})");
    EXPECT_NE(json.find("\"status\":\"ok\""), std::string::npos);
    EXPECT_NE(json.find("\"message\":\"Success\""), std::string::npos);
    EXPECT_NE(json.find("\"loaded\":true"), std::string::npos);
}

TEST_F(ZMQInterfaceTest, ParseResponseValid) {
    std::string json = R"({"status":"ok","message":"done","data":{"result":123}})";
    ResponseStatus status;
    std::string message, data;

    EXPECT_TRUE(JSON::parseResponse(json, status, message, data));
    EXPECT_EQ(status, ResponseStatus::OK);
    EXPECT_EQ(message, "done");
    EXPECT_NE(data.find("123"), std::string::npos);
}

TEST_F(ZMQInterfaceTest, ParseResponseError) {
    std::string json = R"({"status":"error","message":"failed"})";
    ResponseStatus status;
    std::string message, data;

    EXPECT_TRUE(JSON::parseResponse(json, status, message, data));
    EXPECT_EQ(status, ResponseStatus::ERROR);
    EXPECT_EQ(message, "failed");
}

TEST_F(ZMQInterfaceTest, BuildStatus) {
    EngineStatus status;
    status.inputSampleRate = 44100;
    status.outputSampleRate = 705600;
    status.gpuLoad = 15.5;
    status.bufferLevel = 0.85;
    status.eqApplied = true;
    status.currentRateFamily = "44k";
    status.framesProcessed = 1000000;

    std::string json = JSON::buildStatus(status);
    EXPECT_NE(json.find("\"input_rate\":44100"), std::string::npos);
    EXPECT_NE(json.find("\"output_rate\":705600"), std::string::npos);
    EXPECT_NE(json.find("\"eq_applied\":true"), std::string::npos);
    EXPECT_NE(json.find("\"rate_family\":\"44k\""), std::string::npos);
}

TEST_F(ZMQInterfaceTest, ParseStatus) {
    std::string json = R"({
        "type":"status",
        "data":{
            "input_rate":48000,
            "output_rate":768000,
            "gpu_load":20.0,
            "buffer_level":0.9,
            "eq_applied":false,
            "rate_family":"48k",
            "frames_processed":500000
        }
    })";

    EngineStatus status;
    EXPECT_TRUE(JSON::parseStatus(json, status));
    EXPECT_EQ(status.inputSampleRate, 48000);
    EXPECT_EQ(status.outputSampleRate, 768000);
    EXPECT_DOUBLE_EQ(status.gpuLoad, 20.0);
    EXPECT_DOUBLE_EQ(status.bufferLevel, 0.9);
    EXPECT_FALSE(status.eqApplied);
    EXPECT_EQ(status.currentRateFamily, "48k");
    EXPECT_EQ(status.framesProcessed, 500000u);
}

// ============================================================
// EngineStatus Structure Tests
// ============================================================

TEST_F(ZMQInterfaceTest, EngineStatusDefaults) {
    EngineStatus status;
    EXPECT_EQ(status.inputSampleRate, 0);
    EXPECT_EQ(status.outputSampleRate, 0);
    EXPECT_DOUBLE_EQ(status.gpuLoad, 0.0);
    EXPECT_DOUBLE_EQ(status.bufferLevel, 0.0);
    EXPECT_FALSE(status.eqApplied);
    EXPECT_TRUE(status.currentRateFamily.empty());
    EXPECT_EQ(status.framesProcessed, 0u);
}

// ============================================================
// CommandResult Structure Tests
// ============================================================

TEST_F(ZMQInterfaceTest, CommandResultDefaults) {
    CommandResult result;
    EXPECT_EQ(result.status, ResponseStatus::OK);
    EXPECT_TRUE(result.message.empty());
    EXPECT_TRUE(result.data.empty());
}

// ============================================================
// Server/Client Integration Tests (requires actual ZMQ)
// ============================================================

TEST_F(ZMQInterfaceTest, ServerClientRoundTrip) {
    const std::string endpoint = makeUniqueEndpoint("roundtrip");

    // Create server
    ZMQServer server;
    ASSERT_TRUE(server.initialize(endpoint));

    // Register handler
    server.registerHandler(CommandType::GET_STATUS, [](const std::string&) {
        CommandResult result;
        result.status = ResponseStatus::OK;
        result.message = "Status OK";
        result.data = R"({"active":true})";
        return result;
    });

    // Start server (non-blocking)
    ASSERT_TRUE(server.start(false));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Create client
    ZMQClient client;
    ASSERT_TRUE(client.connect(endpoint));

    // Send command
    auto result = client.sendCommand(CommandType::GET_STATUS, "", 2000);
    EXPECT_EQ(result.status, ResponseStatus::OK);
    EXPECT_EQ(result.message, "Status OK");
    EXPECT_NE(result.data.find("active"), std::string::npos);

    // Cleanup
    client.disconnect();
    server.stop();
}

TEST_F(ZMQInterfaceTest, ServerLoadIRHandler) {
    const std::string endpoint = makeUniqueEndpoint("load_ir");

    ZMQServer server;
    ASSERT_TRUE(server.initialize(endpoint));

    // Register LOAD_IR handler
    server.registerHandler(CommandType::LOAD_IR, [](const std::string& params) {
        CommandResult result;
        if (params.find("path") != std::string::npos) {
            result.status = ResponseStatus::OK;
            result.message = "IR loaded";
            result.data = R"({"tap_count":2000000})";
        } else {
            result.status = ResponseStatus::INVALID_PARAMS;
            result.message = "Missing path parameter";
        }
        return result;
    });

    server.start(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ZMQClient client;
    ASSERT_TRUE(client.connect(endpoint));

    // Test with valid params
    auto result = client.loadIR("/data/filter.bin", "44k");
    EXPECT_EQ(result.status, ResponseStatus::OK);
    EXPECT_NE(result.data.find("2000000"), std::string::npos);

    client.disconnect();
    server.stop();
}

TEST_F(ZMQInterfaceTest, ClientNotConnected) {
    ZMQClient client;
    EXPECT_FALSE(client.isConnected());

    auto result = client.sendCommand(CommandType::GET_STATUS);
    EXPECT_EQ(result.status, ResponseStatus::ERROR);
    EXPECT_NE(result.message.find("Not connected"), std::string::npos);
}

TEST_F(ZMQInterfaceTest, ServerNoHandlerRegistered) {
    const std::string endpoint = makeUniqueEndpoint("no_handler");

    ZMQServer server;
    ASSERT_TRUE(server.initialize(endpoint));
    server.start(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ZMQClient client;
    ASSERT_TRUE(client.connect(endpoint));

    // No handler registered for GET_STATUS
    auto result = client.sendCommand(CommandType::GET_STATUS, "", 2000);
    EXPECT_EQ(result.status, ResponseStatus::INVALID_COMMAND);

    client.disconnect();
    server.stop();
}
