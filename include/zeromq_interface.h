#ifndef ZEROMQ_INTERFACE_H
#define ZEROMQ_INTERFACE_H

#include "error_codes.h"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>

namespace ZMQComm {

// Command types from Control Plane to Data Plane
enum class CommandType {
    LOAD_IR,      // Load IR coefficients
    SET_GAIN,     // Set output gain
    SOFT_RESET,   // Soft reset (clear buffers)
    GET_STATUS,   // Get current status
    SWITCH_RATE,  // Switch rate family (44k/48k)
    APPLY_EQ,     // Apply EQ magnitude
    RESTORE_EQ,   // Restore original filter (remove EQ)
    SHUTDOWN,     // Shutdown daemon

    // Crossfeed commands (#150)
    CROSSFEED_ENABLE,            // Enable crossfeed processing
    CROSSFEED_DISABLE,           // Disable crossfeed processing
    CROSSFEED_SET_COMBINED,      // Set combined filter (4ch x 2 rate families, Base64 encoded)
    CROSSFEED_SET_SIZE,          // Set head size (xs/s/m/l/xl)
    CROSSFEED_GET_STATUS,        // Get crossfeed status (enabled, head_size, headphone)
    CROSSFEED_GENERATE_WOODWORTH // Procedurally generate Woodworth HRTF
};

// Response status
enum class ResponseStatus { OK, ERROR, INVALID_COMMAND, INVALID_PARAMS };

// Status data structure
struct EngineStatus {
    int inputSampleRate = 0;
    int outputSampleRate = 0;
    double gpuLoad = 0.0;
    double bufferLevel = 0.0;
    bool eqApplied = false;
    std::string currentRateFamily;
    size_t framesProcessed = 0;
};

// Crossfeed status data structure (#150)
struct CrossfeedStatus {
    bool enabled = false;
    std::string headSize;   // "xs", "s", "m", "l"
    std::string headphone;  // e.g., "HD650", "Sundara"
};

// Command result structure
struct CommandResult {
    ResponseStatus status = ResponseStatus::OK;
    std::string message;
    std::string data;  // JSON data if any
};

// Callback types
using CommandHandler = std::function<CommandResult(const std::string& params)>;
using StatusCallback = std::function<void(const EngineStatus& status)>;

// Convert enums to/from string
const char* commandTypeToString(CommandType type);
CommandType stringToCommandType(const std::string& str);
const char* responseStatusToString(ResponseStatus status);

// ZeroMQ Server (runs in Data Plane / C++ Engine)
class ZMQServer {
   public:
    ZMQServer();
    ~ZMQServer();

    // Initialize server with IPC endpoint
    // endpoint format: "ipc:///tmp/gpu_upsampler.sock" or "tcp://*:5555"
    bool initialize(const std::string& endpoint);

    // Register command handlers
    void registerHandler(CommandType type, CommandHandler handler);

    // Start listening for commands (blocking or non-blocking)
    bool start(bool blocking = false);

    // Stop server
    void stop();

    // Check if server is running
    bool isRunning() const {
        return running_.load();
    }

    // Publish status update (for PUB/SUB pattern)
    bool publishStatus(const EngineStatus& status);

   private:
    void serverLoop();
    std::string processMessage(const std::string& message);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::thread serverThread_;
    std::atomic<bool> running_{false};
    std::map<CommandType, CommandHandler> handlers_;
};

// ZeroMQ Client (runs in Control Plane / Python or C++ orchestrator)
class ZMQClient {
   public:
    ZMQClient();
    ~ZMQClient();

    // Connect to server
    // endpoint format: "ipc:///tmp/gpu_upsampler.sock" or "tcp://localhost:5555"
    bool connect(const std::string& endpoint);

    // Disconnect
    void disconnect();

    // Check if connected
    bool isConnected() const {
        return connected_.load();
    }

    // Send command and receive response (synchronous)
    // timeout_ms: 0 = wait forever
    CommandResult sendCommand(CommandType type, const std::string& params = "",
                              int timeout_ms = 5000);

    // Convenience methods for common commands
    CommandResult loadIR(const std::string& path, const std::string& rateFamily = "44k");
    CommandResult setGain(double gainDb);
    CommandResult softReset();
    CommandResult getStatus();
    CommandResult switchRate(const std::string& rateFamily);
    CommandResult applyEQ(const std::string& eqMagnitudeJson);
    CommandResult restoreEQ();
    CommandResult shutdown();

    // Crossfeed commands (#150)
    CommandResult crossfeedEnable();
    CommandResult crossfeedDisable();
    // Set combined filter for a rate family
    // combinedLL/LR/RL/RR: Base64-encoded cufftComplex arrays
    CommandResult crossfeedSetCombined(const std::string& rateFamily, const std::string& combinedLL,
                                       const std::string& combinedLR, const std::string& combinedRL,
                                       const std::string& combinedRR);
    // Set head size (xs/s/m/l/xl)
    CommandResult crossfeedSetSize(const std::string& headSize);
    CommandResult crossfeedGetStatus();
    CommandResult crossfeedGenerateWoodworth(const std::string& rateFamily, float azimuthDeg,
                                             float headRadiusMeters = 0.09f,
                                             float earSpacingMeters = 0.18f,
                                             float farEarShadowDb = -8.0f,
                                             float diffuseFieldTiltDb = -2.0f);

    // Subscribe to status updates (async)
    bool subscribeStatus(const std::string& pubEndpoint, StatusCallback callback);
    void unsubscribeStatus();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::atomic<bool> connected_{false};
    std::thread subThread_;
    std::atomic<bool> subRunning_{false};
};

// JSON utilities
namespace JSON {
// Build command JSON
std::string buildCommand(CommandType type, const std::string& params);

// Parse command JSON
bool parseCommand(const std::string& json, CommandType& type, std::string& params);

// Build response JSON (legacy format)
std::string buildResponse(ResponseStatus status, const std::string& message = "",
                          const std::string& data = "");

// Build error response JSON with structured error code and inner_error
// Format: { "status": "error", "error_code": "...", "message": "...", "inner_error": {...} }
std::string buildErrorResponse(AudioEngine::ErrorCode code, const std::string& message,
                               const AudioEngine::InnerError& innerError = {});

// Build success response JSON
// Format: { "status": "ok", "message": "...", "data": {...} }
std::string buildOkResponse(const std::string& message = "", const std::string& data = "");

// Parse response JSON
bool parseResponse(const std::string& json, ResponseStatus& status, std::string& message,
                   std::string& data);

// Parse error response JSON (new format with error_code)
bool parseErrorResponse(const std::string& json, std::string& errorCode, std::string& message,
                        std::string& innerErrorJson);

// Build status JSON
std::string buildStatus(const EngineStatus& status);

// Parse status JSON
bool parseStatus(const std::string& json, EngineStatus& status);
}  // namespace JSON

}  // namespace ZMQComm

#endif  // ZEROMQ_INTERFACE_H
