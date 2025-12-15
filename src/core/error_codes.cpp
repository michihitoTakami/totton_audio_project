#include "core/error_codes.h"

#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace AudioEngine {

// Error code to string mapping
static const std::unordered_map<ErrorCode, const char*> kErrorCodeStrings = {
    {ErrorCode::OK, "OK"},

    // Audio Processing
    {ErrorCode::AUDIO_INVALID_INPUT_RATE, "AUDIO_INVALID_INPUT_RATE"},
    {ErrorCode::AUDIO_INVALID_OUTPUT_RATE, "AUDIO_INVALID_OUTPUT_RATE"},
    {ErrorCode::AUDIO_UNSUPPORTED_FORMAT, "AUDIO_UNSUPPORTED_FORMAT"},
    {ErrorCode::AUDIO_FILTER_NOT_FOUND, "AUDIO_FILTER_NOT_FOUND"},
    {ErrorCode::AUDIO_BUFFER_OVERFLOW, "AUDIO_BUFFER_OVERFLOW"},
    {ErrorCode::AUDIO_XRUN_DETECTED, "AUDIO_XRUN_DETECTED"},

    // DAC/ALSA
    {ErrorCode::DAC_DEVICE_NOT_FOUND, "DAC_DEVICE_NOT_FOUND"},
    {ErrorCode::DAC_OPEN_FAILED, "DAC_OPEN_FAILED"},
    {ErrorCode::DAC_CAPABILITY_SCAN_FAILED, "DAC_CAPABILITY_SCAN_FAILED"},
    {ErrorCode::DAC_RATE_NOT_SUPPORTED, "DAC_RATE_NOT_SUPPORTED"},
    {ErrorCode::DAC_FORMAT_NOT_SUPPORTED, "DAC_FORMAT_NOT_SUPPORTED"},
    {ErrorCode::DAC_BUSY, "DAC_BUSY"},

    // IPC/ZeroMQ
    {ErrorCode::IPC_CONNECTION_FAILED, "IPC_CONNECTION_FAILED"},
    {ErrorCode::IPC_TIMEOUT, "IPC_TIMEOUT"},
    {ErrorCode::IPC_INVALID_COMMAND, "IPC_INVALID_COMMAND"},
    {ErrorCode::IPC_INVALID_PARAMS, "IPC_INVALID_PARAMS"},
    {ErrorCode::IPC_DAEMON_NOT_RUNNING, "IPC_DAEMON_NOT_RUNNING"},
    {ErrorCode::IPC_PROTOCOL_ERROR, "IPC_PROTOCOL_ERROR"},

    // GPU/CUDA
    {ErrorCode::GPU_INIT_FAILED, "GPU_INIT_FAILED"},
    {ErrorCode::GPU_DEVICE_NOT_FOUND, "GPU_DEVICE_NOT_FOUND"},
    {ErrorCode::GPU_MEMORY_ERROR, "GPU_MEMORY_ERROR"},
    {ErrorCode::GPU_KERNEL_LAUNCH_FAILED, "GPU_KERNEL_LAUNCH_FAILED"},
    {ErrorCode::GPU_FILTER_LOAD_FAILED, "GPU_FILTER_LOAD_FAILED"},
    {ErrorCode::GPU_CUFFT_ERROR, "GPU_CUFFT_ERROR"},

    // Validation
    {ErrorCode::VALIDATION_INVALID_CONFIG, "VALIDATION_INVALID_CONFIG"},
    {ErrorCode::VALIDATION_INVALID_PROFILE, "VALIDATION_INVALID_PROFILE"},
    {ErrorCode::VALIDATION_PATH_TRAVERSAL, "VALIDATION_PATH_TRAVERSAL"},
    {ErrorCode::VALIDATION_FILE_NOT_FOUND, "VALIDATION_FILE_NOT_FOUND"},
    {ErrorCode::VALIDATION_PROFILE_EXISTS, "VALIDATION_PROFILE_EXISTS"},
    {ErrorCode::VALIDATION_INVALID_HEADPHONE, "VALIDATION_INVALID_HEADPHONE"},
    {ErrorCode::ERR_UNSUPPORTED_MODE, "ERR_UNSUPPORTED_MODE"},

    // Internal
    {ErrorCode::INTERNAL_UNKNOWN, "INTERNAL_UNKNOWN"},
};

// Error code to HTTP status mapping
static const std::unordered_map<ErrorCode, int> kHttpStatusMap = {
    {ErrorCode::OK, 200},

    // Audio Processing
    {ErrorCode::AUDIO_INVALID_INPUT_RATE, 400},
    {ErrorCode::AUDIO_INVALID_OUTPUT_RATE, 400},
    {ErrorCode::AUDIO_UNSUPPORTED_FORMAT, 400},
    {ErrorCode::AUDIO_FILTER_NOT_FOUND, 404},
    {ErrorCode::AUDIO_BUFFER_OVERFLOW, 500},
    {ErrorCode::AUDIO_XRUN_DETECTED, 500},

    // DAC/ALSA
    {ErrorCode::DAC_DEVICE_NOT_FOUND, 404},
    {ErrorCode::DAC_OPEN_FAILED, 500},
    {ErrorCode::DAC_CAPABILITY_SCAN_FAILED, 500},
    {ErrorCode::DAC_RATE_NOT_SUPPORTED, 422},
    {ErrorCode::DAC_FORMAT_NOT_SUPPORTED, 422},
    {ErrorCode::DAC_BUSY, 409},

    // IPC/ZeroMQ
    {ErrorCode::IPC_CONNECTION_FAILED, 503},
    {ErrorCode::IPC_TIMEOUT, 504},
    {ErrorCode::IPC_INVALID_COMMAND, 400},
    {ErrorCode::IPC_INVALID_PARAMS, 400},
    {ErrorCode::IPC_DAEMON_NOT_RUNNING, 503},
    {ErrorCode::IPC_PROTOCOL_ERROR, 500},

    // GPU/CUDA
    {ErrorCode::GPU_INIT_FAILED, 500},
    {ErrorCode::GPU_DEVICE_NOT_FOUND, 500},
    {ErrorCode::GPU_MEMORY_ERROR, 500},
    {ErrorCode::GPU_KERNEL_LAUNCH_FAILED, 500},
    {ErrorCode::GPU_FILTER_LOAD_FAILED, 500},
    {ErrorCode::GPU_CUFFT_ERROR, 500},

    // Validation
    {ErrorCode::VALIDATION_INVALID_CONFIG, 400},
    {ErrorCode::VALIDATION_INVALID_PROFILE, 400},
    {ErrorCode::VALIDATION_PATH_TRAVERSAL, 400},
    {ErrorCode::VALIDATION_FILE_NOT_FOUND, 404},
    {ErrorCode::VALIDATION_PROFILE_EXISTS, 409},
    {ErrorCode::VALIDATION_INVALID_HEADPHONE, 404},
    {ErrorCode::ERR_UNSUPPORTED_MODE, 400},

    // Internal
    {ErrorCode::INTERNAL_UNKNOWN, 500},
};

// String to error code mapping (reverse lookup)
static const std::unordered_map<std::string, ErrorCode> kStringToErrorCode = {
    {"OK", ErrorCode::OK},

    // Audio Processing
    {"AUDIO_INVALID_INPUT_RATE", ErrorCode::AUDIO_INVALID_INPUT_RATE},
    {"AUDIO_INVALID_OUTPUT_RATE", ErrorCode::AUDIO_INVALID_OUTPUT_RATE},
    {"AUDIO_UNSUPPORTED_FORMAT", ErrorCode::AUDIO_UNSUPPORTED_FORMAT},
    {"AUDIO_FILTER_NOT_FOUND", ErrorCode::AUDIO_FILTER_NOT_FOUND},
    {"AUDIO_BUFFER_OVERFLOW", ErrorCode::AUDIO_BUFFER_OVERFLOW},
    {"AUDIO_XRUN_DETECTED", ErrorCode::AUDIO_XRUN_DETECTED},

    // DAC/ALSA
    {"DAC_DEVICE_NOT_FOUND", ErrorCode::DAC_DEVICE_NOT_FOUND},
    {"DAC_OPEN_FAILED", ErrorCode::DAC_OPEN_FAILED},
    {"DAC_CAPABILITY_SCAN_FAILED", ErrorCode::DAC_CAPABILITY_SCAN_FAILED},
    {"DAC_RATE_NOT_SUPPORTED", ErrorCode::DAC_RATE_NOT_SUPPORTED},
    {"DAC_FORMAT_NOT_SUPPORTED", ErrorCode::DAC_FORMAT_NOT_SUPPORTED},
    {"DAC_BUSY", ErrorCode::DAC_BUSY},

    // IPC/ZeroMQ
    {"IPC_CONNECTION_FAILED", ErrorCode::IPC_CONNECTION_FAILED},
    {"IPC_TIMEOUT", ErrorCode::IPC_TIMEOUT},
    {"IPC_INVALID_COMMAND", ErrorCode::IPC_INVALID_COMMAND},
    {"IPC_INVALID_PARAMS", ErrorCode::IPC_INVALID_PARAMS},
    {"IPC_DAEMON_NOT_RUNNING", ErrorCode::IPC_DAEMON_NOT_RUNNING},
    {"IPC_PROTOCOL_ERROR", ErrorCode::IPC_PROTOCOL_ERROR},

    // GPU/CUDA
    {"GPU_INIT_FAILED", ErrorCode::GPU_INIT_FAILED},
    {"GPU_DEVICE_NOT_FOUND", ErrorCode::GPU_DEVICE_NOT_FOUND},
    {"GPU_MEMORY_ERROR", ErrorCode::GPU_MEMORY_ERROR},
    {"GPU_KERNEL_LAUNCH_FAILED", ErrorCode::GPU_KERNEL_LAUNCH_FAILED},
    {"GPU_FILTER_LOAD_FAILED", ErrorCode::GPU_FILTER_LOAD_FAILED},
    {"GPU_CUFFT_ERROR", ErrorCode::GPU_CUFFT_ERROR},

    // Validation
    {"VALIDATION_INVALID_CONFIG", ErrorCode::VALIDATION_INVALID_CONFIG},
    {"VALIDATION_INVALID_PROFILE", ErrorCode::VALIDATION_INVALID_PROFILE},
    {"VALIDATION_PATH_TRAVERSAL", ErrorCode::VALIDATION_PATH_TRAVERSAL},
    {"VALIDATION_FILE_NOT_FOUND", ErrorCode::VALIDATION_FILE_NOT_FOUND},
    {"VALIDATION_PROFILE_EXISTS", ErrorCode::VALIDATION_PROFILE_EXISTS},
    {"VALIDATION_INVALID_HEADPHONE", ErrorCode::VALIDATION_INVALID_HEADPHONE},
    {"ERR_UNSUPPORTED_MODE", ErrorCode::ERR_UNSUPPORTED_MODE},

    // Internal
    {"INTERNAL_UNKNOWN", ErrorCode::INTERNAL_UNKNOWN},
};

InnerError::InnerError(ErrorCode code, const std::string& message)
    : cpp_code(errorCodeToHex(code)), cpp_message(message) {}

const char* errorCodeToString(ErrorCode code) {
    auto it = kErrorCodeStrings.find(code);
    if (it != kErrorCodeStrings.end()) {
        return it->second;
    }
    return "UNKNOWN_ERROR";
}

const char* getErrorCategory(ErrorCode code) {
    if (code == ErrorCode::OK) {
        return "ok";
    }
    if (isAudioError(code)) {
        return "audio_processing";
    }
    if (isDacError(code)) {
        return "dac_alsa";
    }
    if (isIpcError(code)) {
        return "ipc_zeromq";
    }
    if (isGpuError(code)) {
        return "gpu_cuda";
    }
    if (isValidationError(code)) {
        return "validation";
    }
    return "internal";
}

int toHttpStatus(ErrorCode code) {
    auto it = kHttpStatusMap.find(code);
    if (it != kHttpStatusMap.end()) {
        return it->second;
    }
    return 500;  // Default to Internal Server Error
}

std::string errorCodeToHex(ErrorCode code) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setfill('0') << std::setw(4) << static_cast<uint32_t>(code);
    return oss.str();
}

ErrorCode stringToErrorCode(const std::string& str) {
    auto it = kStringToErrorCode.find(str);
    if (it != kStringToErrorCode.end()) {
        return it->second;
    }
    return ErrorCode::INTERNAL_UNKNOWN;
}

}  // namespace AudioEngine
