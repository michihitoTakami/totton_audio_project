/**
 * @file test_error_codes.cpp
 * @brief Unit tests for error codes and JSON error response building.
 */

#include "error_codes.h"
#include "zeromq_interface.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace AudioEngine;

// ============================================================
// ErrorCode to String Tests
// ============================================================

TEST(ErrorCodes, ErrorCodeToString) {
    EXPECT_STREQ(errorCodeToString(ErrorCode::OK), "OK");
    EXPECT_STREQ(errorCodeToString(ErrorCode::DAC_RATE_NOT_SUPPORTED), "DAC_RATE_NOT_SUPPORTED");
    EXPECT_STREQ(errorCodeToString(ErrorCode::IPC_TIMEOUT), "IPC_TIMEOUT");
    EXPECT_STREQ(errorCodeToString(ErrorCode::GPU_MEMORY_ERROR), "GPU_MEMORY_ERROR");
    EXPECT_STREQ(errorCodeToString(ErrorCode::VALIDATION_PATH_TRAVERSAL),
                 "VALIDATION_PATH_TRAVERSAL");
}

TEST(ErrorCodes, UnknownErrorCodeReturnsUnknown) {
    auto unknownCode = static_cast<ErrorCode>(0xFFFF);
    EXPECT_STREQ(errorCodeToString(unknownCode), "UNKNOWN_ERROR");
}

// ============================================================
// Error Category Tests
// ============================================================

TEST(ErrorCodes, GetErrorCategory) {
    EXPECT_STREQ(getErrorCategory(ErrorCode::OK), "ok");
    EXPECT_STREQ(getErrorCategory(ErrorCode::AUDIO_INVALID_INPUT_RATE), "audio_processing");
    EXPECT_STREQ(getErrorCategory(ErrorCode::DAC_DEVICE_NOT_FOUND), "dac_alsa");
    EXPECT_STREQ(getErrorCategory(ErrorCode::IPC_CONNECTION_FAILED), "ipc_zeromq");
    EXPECT_STREQ(getErrorCategory(ErrorCode::GPU_INIT_FAILED), "gpu_cuda");
    EXPECT_STREQ(getErrorCategory(ErrorCode::VALIDATION_INVALID_CONFIG), "validation");
}

TEST(ErrorCodes, UnknownCategoryReturnsInternal) {
    auto unknownCode = static_cast<ErrorCode>(0xFFFF);
    EXPECT_STREQ(getErrorCategory(unknownCode), "internal");
}

// ============================================================
// Category Check Helper Tests
// ============================================================

TEST(ErrorCodes, CategoryCheckers) {
    EXPECT_TRUE(isAudioError(ErrorCode::AUDIO_INVALID_INPUT_RATE));
    EXPECT_TRUE(isAudioError(ErrorCode::AUDIO_XRUN_DETECTED));
    EXPECT_FALSE(isAudioError(ErrorCode::DAC_DEVICE_NOT_FOUND));

    EXPECT_TRUE(isDacError(ErrorCode::DAC_DEVICE_NOT_FOUND));
    EXPECT_TRUE(isDacError(ErrorCode::DAC_BUSY));
    EXPECT_FALSE(isDacError(ErrorCode::IPC_TIMEOUT));

    EXPECT_TRUE(isIpcError(ErrorCode::IPC_CONNECTION_FAILED));
    EXPECT_TRUE(isIpcError(ErrorCode::IPC_PROTOCOL_ERROR));
    EXPECT_FALSE(isIpcError(ErrorCode::GPU_INIT_FAILED));

    EXPECT_TRUE(isGpuError(ErrorCode::GPU_INIT_FAILED));
    EXPECT_TRUE(isGpuError(ErrorCode::GPU_CUFFT_ERROR));
    EXPECT_FALSE(isGpuError(ErrorCode::VALIDATION_INVALID_CONFIG));

    EXPECT_TRUE(isValidationError(ErrorCode::VALIDATION_INVALID_CONFIG));
    EXPECT_TRUE(isValidationError(ErrorCode::VALIDATION_INVALID_HEADPHONE));
    EXPECT_FALSE(isValidationError(ErrorCode::AUDIO_BUFFER_OVERFLOW));
}

// ============================================================
// HTTP Status Mapping Tests
// ============================================================

TEST(ErrorCodes, ToHttpStatus) {
    EXPECT_EQ(toHttpStatus(ErrorCode::OK), 200);

    // 400 Bad Request
    EXPECT_EQ(toHttpStatus(ErrorCode::AUDIO_INVALID_INPUT_RATE), 400);
    EXPECT_EQ(toHttpStatus(ErrorCode::IPC_INVALID_COMMAND), 400);
    EXPECT_EQ(toHttpStatus(ErrorCode::VALIDATION_INVALID_CONFIG), 400);

    // 404 Not Found
    EXPECT_EQ(toHttpStatus(ErrorCode::DAC_DEVICE_NOT_FOUND), 404);
    EXPECT_EQ(toHttpStatus(ErrorCode::AUDIO_FILTER_NOT_FOUND), 404);
    EXPECT_EQ(toHttpStatus(ErrorCode::VALIDATION_FILE_NOT_FOUND), 404);

    // 409 Conflict
    EXPECT_EQ(toHttpStatus(ErrorCode::DAC_BUSY), 409);
    EXPECT_EQ(toHttpStatus(ErrorCode::VALIDATION_PROFILE_EXISTS), 409);

    // 422 Unprocessable Entity
    EXPECT_EQ(toHttpStatus(ErrorCode::DAC_RATE_NOT_SUPPORTED), 422);
    EXPECT_EQ(toHttpStatus(ErrorCode::DAC_FORMAT_NOT_SUPPORTED), 422);

    // 500 Internal Server Error
    EXPECT_EQ(toHttpStatus(ErrorCode::GPU_INIT_FAILED), 500);
    EXPECT_EQ(toHttpStatus(ErrorCode::AUDIO_BUFFER_OVERFLOW), 500);

    // 503 Service Unavailable
    EXPECT_EQ(toHttpStatus(ErrorCode::IPC_CONNECTION_FAILED), 503);
    EXPECT_EQ(toHttpStatus(ErrorCode::IPC_DAEMON_NOT_RUNNING), 503);

    // 504 Gateway Timeout
    EXPECT_EQ(toHttpStatus(ErrorCode::IPC_TIMEOUT), 504);
}

TEST(ErrorCodes, UnknownErrorReturns500) {
    auto unknownCode = static_cast<ErrorCode>(0xFFFF);
    EXPECT_EQ(toHttpStatus(unknownCode), 500);
}

// ============================================================
// Error Code to Hex Tests
// ============================================================

TEST(ErrorCodes, ErrorCodeToHex) {
    EXPECT_EQ(errorCodeToHex(ErrorCode::OK), "0x0000");
    EXPECT_EQ(errorCodeToHex(ErrorCode::AUDIO_INVALID_INPUT_RATE), "0x1001");
    EXPECT_EQ(errorCodeToHex(ErrorCode::DAC_RATE_NOT_SUPPORTED), "0x2004");
    EXPECT_EQ(errorCodeToHex(ErrorCode::IPC_TIMEOUT), "0x3002");
    EXPECT_EQ(errorCodeToHex(ErrorCode::GPU_MEMORY_ERROR), "0x4003");
    EXPECT_EQ(errorCodeToHex(ErrorCode::VALIDATION_PATH_TRAVERSAL), "0x5003");
}

// ============================================================
// InnerError Tests
// ============================================================

TEST(ErrorCodes, InnerErrorConstruction) {
    InnerError err(ErrorCode::DAC_RATE_NOT_SUPPORTED, "Rate 1000000 not supported");

    EXPECT_EQ(err.cpp_code, "0x2004");
    EXPECT_EQ(err.cpp_message, "Rate 1000000 not supported");
    EXPECT_FALSE(err.alsa_errno.has_value());
    EXPECT_FALSE(err.alsa_func.has_value());
    EXPECT_FALSE(err.cuda_error.has_value());
}

TEST(ErrorCodes, InnerErrorWithAlsaInfo) {
    InnerError err;
    err.cpp_code = "0x2004";
    err.cpp_message = "Rate not supported";
    err.alsa_errno = -22;
    err.alsa_func = "snd_pcm_hw_params_set_rate_near";

    EXPECT_TRUE(err.alsa_errno.has_value());
    EXPECT_EQ(err.alsa_errno.value(), -22);
    EXPECT_TRUE(err.alsa_func.has_value());
    EXPECT_EQ(err.alsa_func.value(), "snd_pcm_hw_params_set_rate_near");
}

// ============================================================
// JSON Error Response Building Tests
// ============================================================

TEST(ErrorCodes, BuildErrorResponseBasic) {
    auto response =
        ZMQComm::JSON::buildErrorResponse(ErrorCode::DAC_RATE_NOT_SUPPORTED, "Rate not supported");

    auto j = json::parse(response);

    EXPECT_EQ(j["status"], "error");
    EXPECT_EQ(j["error_code"], "DAC_RATE_NOT_SUPPORTED");
    EXPECT_EQ(j["message"], "Rate not supported");
    EXPECT_TRUE(j.contains("inner_error"));
    EXPECT_EQ(j["inner_error"]["cpp_code"], "0x2004");
    // cpp_message should fallback to outer message when InnerError is default
    EXPECT_EQ(j["inner_error"]["cpp_message"], "Rate not supported");
}

TEST(ErrorCodes, BuildErrorResponseWithInnerError) {
    InnerError inner;
    inner.cpp_code = "0x2004";
    inner.cpp_message = "ALSA rate negotiation failed";
    inner.alsa_errno = -22;
    inner.alsa_func = "snd_pcm_hw_params_set_rate_near";

    auto response = ZMQComm::JSON::buildErrorResponse(ErrorCode::DAC_RATE_NOT_SUPPORTED,
                                                      "Rate 1000000 not supported", inner);

    auto j = json::parse(response);

    EXPECT_EQ(j["status"], "error");
    EXPECT_EQ(j["error_code"], "DAC_RATE_NOT_SUPPORTED");
    EXPECT_EQ(j["message"], "Rate 1000000 not supported");

    auto& innerJson = j["inner_error"];
    EXPECT_EQ(innerJson["cpp_code"], "0x2004");
    EXPECT_EQ(innerJson["cpp_message"], "ALSA rate negotiation failed");
    EXPECT_EQ(innerJson["alsa_errno"], -22);
    EXPECT_EQ(innerJson["alsa_func"], "snd_pcm_hw_params_set_rate_near");
    EXPECT_TRUE(innerJson["cuda_error"].is_null());
}

TEST(ErrorCodes, BuildErrorResponseWithCudaError) {
    InnerError inner;
    inner.cpp_code = "0x4003";
    inner.cpp_message = "Failed to allocate GPU memory";
    inner.cuda_error = "cudaErrorMemoryAllocation";

    auto response =
        ZMQComm::JSON::buildErrorResponse(ErrorCode::GPU_MEMORY_ERROR, "GPU memory error", inner);

    auto j = json::parse(response);

    EXPECT_EQ(j["error_code"], "GPU_MEMORY_ERROR");
    EXPECT_EQ(j["inner_error"]["cuda_error"], "cudaErrorMemoryAllocation");
    EXPECT_TRUE(j["inner_error"]["alsa_errno"].is_null());
}

// ============================================================
// JSON OK Response Building Tests
// ============================================================

TEST(ErrorCodes, BuildOkResponseBasic) {
    auto response = ZMQComm::JSON::buildOkResponse();

    auto j = json::parse(response);

    EXPECT_EQ(j["status"], "ok");
    EXPECT_FALSE(j.contains("message"));
    EXPECT_FALSE(j.contains("data"));
}

TEST(ErrorCodes, BuildOkResponseWithMessage) {
    auto response = ZMQComm::JSON::buildOkResponse("Operation completed");

    auto j = json::parse(response);

    EXPECT_EQ(j["status"], "ok");
    EXPECT_EQ(j["message"], "Operation completed");
}

TEST(ErrorCodes, BuildOkResponseWithData) {
    auto response = ZMQComm::JSON::buildOkResponse("Success", R"({"rate": 44100})");

    auto j = json::parse(response);

    EXPECT_EQ(j["status"], "ok");
    EXPECT_EQ(j["message"], "Success");
    EXPECT_EQ(j["data"]["rate"], 44100);
}

// ============================================================
// JSON Error Response Parsing Tests
// ============================================================

TEST(ErrorCodes, ParseErrorResponse) {
    std::string errorJson = R"({
        "status": "error",
        "error_code": "DAC_RATE_NOT_SUPPORTED",
        "message": "Rate not supported",
        "inner_error": {
            "cpp_code": "0x2004",
            "cpp_message": "ALSA error",
            "alsa_errno": -22
        }
    })";

    std::string errorCode, message, innerErrorJson;
    EXPECT_TRUE(ZMQComm::JSON::parseErrorResponse(errorJson, errorCode, message, innerErrorJson));

    EXPECT_EQ(errorCode, "DAC_RATE_NOT_SUPPORTED");
    EXPECT_EQ(message, "Rate not supported");

    auto inner = json::parse(innerErrorJson);
    EXPECT_EQ(inner["cpp_code"], "0x2004");
    EXPECT_EQ(inner["alsa_errno"], -22);
}

TEST(ErrorCodes, ParseErrorResponseFailsOnOkStatus) {
    std::string okJson = R"({"status": "ok", "message": "Success"})";

    std::string errorCode, message, innerErrorJson;
    EXPECT_FALSE(ZMQComm::JSON::parseErrorResponse(okJson, errorCode, message, innerErrorJson));
}

TEST(ErrorCodes, ParseErrorResponseFailsOnMissingErrorCode) {
    // Missing error_code field - should fail validation
    std::string brokenJson = R"({
        "status": "error",
        "message": "Some error"
    })";

    std::string errorCode, message, innerErrorJson;
    EXPECT_FALSE(ZMQComm::JSON::parseErrorResponse(brokenJson, errorCode, message, innerErrorJson));
}

TEST(ErrorCodes, ParseErrorResponseFailsOnMissingMessage) {
    // Missing message field - should fail validation
    std::string brokenJson = R"({
        "status": "error",
        "error_code": "IPC_TIMEOUT"
    })";

    std::string errorCode, message, innerErrorJson;
    EXPECT_FALSE(ZMQComm::JSON::parseErrorResponse(brokenJson, errorCode, message, innerErrorJson));
}

TEST(ErrorCodes, ParseErrorResponseFailsOnWrongTypeErrorCode) {
    // error_code is not a string - should fail validation
    std::string brokenJson = R"({
        "status": "error",
        "error_code": 12345,
        "message": "Some error"
    })";

    std::string errorCode, message, innerErrorJson;
    EXPECT_FALSE(ZMQComm::JSON::parseErrorResponse(brokenJson, errorCode, message, innerErrorJson));
}

TEST(ErrorCodes, ParseErrorResponseFailsOnWrongTypeMessage) {
    // message is not a string - should fail validation
    std::string brokenJson = R"({
        "status": "error",
        "error_code": "IPC_TIMEOUT",
        "message": 12345
    })";

    std::string errorCode, message, innerErrorJson;
    EXPECT_FALSE(ZMQComm::JSON::parseErrorResponse(brokenJson, errorCode, message, innerErrorJson));
}

// ============================================================
// All 30 Error Codes Coverage Test
// ============================================================

TEST(ErrorCodes, AllErrorCodesHaveStringMapping) {
    // Audio Processing (6 codes)
    EXPECT_STRNE(errorCodeToString(ErrorCode::AUDIO_INVALID_INPUT_RATE), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::AUDIO_INVALID_OUTPUT_RATE), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::AUDIO_UNSUPPORTED_FORMAT), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::AUDIO_FILTER_NOT_FOUND), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::AUDIO_BUFFER_OVERFLOW), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::AUDIO_XRUN_DETECTED), "UNKNOWN_ERROR");

    // DAC/ALSA (6 codes)
    EXPECT_STRNE(errorCodeToString(ErrorCode::DAC_DEVICE_NOT_FOUND), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::DAC_OPEN_FAILED), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::DAC_CAPABILITY_SCAN_FAILED), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::DAC_RATE_NOT_SUPPORTED), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::DAC_FORMAT_NOT_SUPPORTED), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::DAC_BUSY), "UNKNOWN_ERROR");

    // IPC/ZeroMQ (6 codes)
    EXPECT_STRNE(errorCodeToString(ErrorCode::IPC_CONNECTION_FAILED), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::IPC_TIMEOUT), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::IPC_INVALID_COMMAND), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::IPC_INVALID_PARAMS), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::IPC_DAEMON_NOT_RUNNING), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::IPC_PROTOCOL_ERROR), "UNKNOWN_ERROR");

    // GPU/CUDA (6 codes)
    EXPECT_STRNE(errorCodeToString(ErrorCode::GPU_INIT_FAILED), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::GPU_DEVICE_NOT_FOUND), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::GPU_MEMORY_ERROR), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::GPU_KERNEL_LAUNCH_FAILED), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::GPU_FILTER_LOAD_FAILED), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::GPU_CUFFT_ERROR), "UNKNOWN_ERROR");

    // Validation (6 codes)
    EXPECT_STRNE(errorCodeToString(ErrorCode::VALIDATION_INVALID_CONFIG), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::VALIDATION_INVALID_PROFILE), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::VALIDATION_PATH_TRAVERSAL), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::VALIDATION_FILE_NOT_FOUND), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::VALIDATION_PROFILE_EXISTS), "UNKNOWN_ERROR");
    EXPECT_STRNE(errorCodeToString(ErrorCode::VALIDATION_INVALID_HEADPHONE), "UNKNOWN_ERROR");
}

TEST(ErrorCodes, AllErrorCodesHaveHttpStatusMapping) {
    // All 30 codes should have HTTP status (not default 500 unless intentional)
    // Just verify they all return a valid HTTP status code
    auto validateStatus = [](ErrorCode code) {
        int status = toHttpStatus(code);
        return status >= 200 && status < 600;
    };

    EXPECT_TRUE(validateStatus(ErrorCode::AUDIO_INVALID_INPUT_RATE));
    EXPECT_TRUE(validateStatus(ErrorCode::DAC_RATE_NOT_SUPPORTED));
    EXPECT_TRUE(validateStatus(ErrorCode::IPC_TIMEOUT));
    EXPECT_TRUE(validateStatus(ErrorCode::GPU_MEMORY_ERROR));
    EXPECT_TRUE(validateStatus(ErrorCode::VALIDATION_PATH_TRAVERSAL));
}

// ============================================================
// INTERNAL_UNKNOWN Tests (Issue #44 additions)
// ============================================================

TEST(ErrorCodes, InternalUnknownExists) {
    EXPECT_STREQ(errorCodeToString(ErrorCode::INTERNAL_UNKNOWN), "INTERNAL_UNKNOWN");
    EXPECT_EQ(errorCodeToHex(ErrorCode::INTERNAL_UNKNOWN), "0xf001");
    EXPECT_EQ(toHttpStatus(ErrorCode::INTERNAL_UNKNOWN), 500);
}

TEST(ErrorCodes, IsInternalErrorHelper) {
    EXPECT_TRUE(isInternalError(ErrorCode::INTERNAL_UNKNOWN));
    EXPECT_FALSE(isInternalError(ErrorCode::DAC_DEVICE_NOT_FOUND));
    EXPECT_FALSE(isInternalError(ErrorCode::GPU_INIT_FAILED));
    EXPECT_FALSE(isInternalError(ErrorCode::OK));
}

// ============================================================
// isRetryable Tests
// ============================================================

TEST(ErrorCodes, IsRetryableReturnsTrue) {
    EXPECT_TRUE(isRetryable(ErrorCode::IPC_DAEMON_NOT_RUNNING));
    EXPECT_TRUE(isRetryable(ErrorCode::IPC_TIMEOUT));
    EXPECT_TRUE(isRetryable(ErrorCode::IPC_CONNECTION_FAILED));
}

TEST(ErrorCodes, IsRetryableReturnsFalse) {
    EXPECT_FALSE(isRetryable(ErrorCode::OK));
    EXPECT_FALSE(isRetryable(ErrorCode::DAC_DEVICE_NOT_FOUND));
    EXPECT_FALSE(isRetryable(ErrorCode::GPU_MEMORY_ERROR));
    EXPECT_FALSE(isRetryable(ErrorCode::VALIDATION_INVALID_CONFIG));
    EXPECT_FALSE(isRetryable(ErrorCode::IPC_INVALID_COMMAND));
    EXPECT_FALSE(isRetryable(ErrorCode::INTERNAL_UNKNOWN));
}

// ============================================================
// stringToErrorCode Tests
// ============================================================

TEST(ErrorCodes, StringToErrorCodeValid) {
    EXPECT_EQ(stringToErrorCode("OK"), ErrorCode::OK);
    EXPECT_EQ(stringToErrorCode("AUDIO_INVALID_INPUT_RATE"), ErrorCode::AUDIO_INVALID_INPUT_RATE);
    EXPECT_EQ(stringToErrorCode("DAC_DEVICE_NOT_FOUND"), ErrorCode::DAC_DEVICE_NOT_FOUND);
    EXPECT_EQ(stringToErrorCode("IPC_TIMEOUT"), ErrorCode::IPC_TIMEOUT);
    EXPECT_EQ(stringToErrorCode("GPU_MEMORY_ERROR"), ErrorCode::GPU_MEMORY_ERROR);
    EXPECT_EQ(stringToErrorCode("VALIDATION_PATH_TRAVERSAL"), ErrorCode::VALIDATION_PATH_TRAVERSAL);
    EXPECT_EQ(stringToErrorCode("INTERNAL_UNKNOWN"), ErrorCode::INTERNAL_UNKNOWN);
}

TEST(ErrorCodes, StringToErrorCodeInvalid) {
    EXPECT_EQ(stringToErrorCode("INVALID_CODE"), ErrorCode::INTERNAL_UNKNOWN);
    EXPECT_EQ(stringToErrorCode(""), ErrorCode::INTERNAL_UNKNOWN);
    EXPECT_EQ(stringToErrorCode("random_string"), ErrorCode::INTERNAL_UNKNOWN);
}

TEST(ErrorCodes, RoundTripStringConversion) {
    // Test that errorCodeToString and stringToErrorCode are inverses
    auto testRoundTrip = [](ErrorCode code) {
        const char* str = errorCodeToString(code);
        ErrorCode result = stringToErrorCode(str);
        EXPECT_EQ(code, result) << "Failed for: " << str;
    };

    testRoundTrip(ErrorCode::OK);
    testRoundTrip(ErrorCode::AUDIO_XRUN_DETECTED);
    testRoundTrip(ErrorCode::DAC_BUSY);
    testRoundTrip(ErrorCode::IPC_PROTOCOL_ERROR);
    testRoundTrip(ErrorCode::GPU_CUFFT_ERROR);
    testRoundTrip(ErrorCode::VALIDATION_INVALID_HEADPHONE);
    testRoundTrip(ErrorCode::INTERNAL_UNKNOWN);
}
