#ifndef ERROR_CODES_H
#define ERROR_CODES_H

#include <cstdint>
#include <optional>
#include <string>

namespace AudioEngine {

/**
 * @brief Error codes for the Audio Engine.
 *
 * Based on docs/architecture/error-codes.md design document.
 * Categories use upper 12 bits (0xF000 mask):
 * - 0x1xxx: Audio Processing
 * - 0x2xxx: DAC/ALSA
 * - 0x3xxx: IPC/ZeroMQ
 * - 0x4xxx: GPU/CUDA
 * - 0x5xxx: Validation
 * - 0xFxxx: Internal (reserved)
 */
enum class ErrorCode : uint32_t {
    OK = 0,

    // Audio Processing (0x1000)
    AUDIO_INVALID_INPUT_RATE = 0x1001,
    AUDIO_INVALID_OUTPUT_RATE = 0x1002,
    AUDIO_UNSUPPORTED_FORMAT = 0x1003,
    AUDIO_FILTER_NOT_FOUND = 0x1004,
    AUDIO_BUFFER_OVERFLOW = 0x1005,
    AUDIO_XRUN_DETECTED = 0x1006,
    AUDIO_RTP_SOCKET_ERROR = 0x1007,
    AUDIO_RTP_SESSION_NOT_FOUND = 0x1008,

    // DAC/ALSA (0x2000)
    DAC_DEVICE_NOT_FOUND = 0x2001,
    DAC_OPEN_FAILED = 0x2002,
    DAC_CAPABILITY_SCAN_FAILED = 0x2003,
    DAC_RATE_NOT_SUPPORTED = 0x2004,
    DAC_FORMAT_NOT_SUPPORTED = 0x2005,
    DAC_BUSY = 0x2006,

    // IPC/ZeroMQ (0x3000)
    IPC_CONNECTION_FAILED = 0x3001,
    IPC_TIMEOUT = 0x3002,
    IPC_INVALID_COMMAND = 0x3003,
    IPC_INVALID_PARAMS = 0x3004,
    IPC_DAEMON_NOT_RUNNING = 0x3005,
    IPC_PROTOCOL_ERROR = 0x3006,

    // GPU/CUDA (0x4000)
    GPU_INIT_FAILED = 0x4001,
    GPU_DEVICE_NOT_FOUND = 0x4002,
    GPU_MEMORY_ERROR = 0x4003,
    GPU_KERNEL_LAUNCH_FAILED = 0x4004,
    GPU_FILTER_LOAD_FAILED = 0x4005,
    GPU_CUFFT_ERROR = 0x4006,

    // Validation (0x5000)
    VALIDATION_INVALID_CONFIG = 0x5001,
    VALIDATION_INVALID_PROFILE = 0x5002,
    VALIDATION_PATH_TRAVERSAL = 0x5003,
    VALIDATION_FILE_NOT_FOUND = 0x5004,
    VALIDATION_PROFILE_EXISTS = 0x5005,
    VALIDATION_INVALID_HEADPHONE = 0x5006,

    // Internal (0xF000) - Reserved for fallback
    /** @brief Unknown/unmapped error */
    INTERNAL_UNKNOWN = 0xF001,
};

/**
 * @brief Inner error details from lower layers.
 *
 * Used to propagate detailed error information from ALSA, CUDA, etc.
 */
struct InnerError {
    std::string cpp_code;                   // Error code as hex string (e.g., "0x2004")
    std::string cpp_message;                // Detailed C++ error message
    std::optional<int> alsa_errno;          // ALSA error number
    std::optional<std::string> alsa_func;   // Failed ALSA function name
    std::optional<std::string> cuda_error;  // CUDA error name

    // Default constructor
    InnerError() = default;

    // Convenience constructor for simple errors
    InnerError(ErrorCode code, const std::string& message);
};

/**
 * @brief Convert ErrorCode to string representation.
 * @param code The error code
 * @return String name (e.g., "DAC_RATE_NOT_SUPPORTED"), or "UNKNOWN_ERROR" for unknown codes
 */
const char* errorCodeToString(ErrorCode code);

/**
 * @brief Get the category name for an error code.
 * @param code The error code
 * @return Category name (e.g., "dac_alsa"), or "internal" for unknown codes
 */
const char* getErrorCategory(ErrorCode code);

/**
 * @brief Convert ErrorCode to HTTP status code.
 * @param code The error code
 * @return HTTP status code (e.g., 400, 404, 500), or 500 for unknown codes
 */
int toHttpStatus(ErrorCode code);

/**
 * @brief Convert ErrorCode to hex string.
 * @param code The error code
 * @return Hex string (e.g., "0x2004")
 */
std::string errorCodeToHex(ErrorCode code);

/**
 * @brief Convert string to ErrorCode enum.
 * @param str Error code string (e.g., "DAC_DEVICE_NOT_FOUND")
 * @return Corresponding ErrorCode, or INTERNAL_UNKNOWN if not found
 */
ErrorCode stringToErrorCode(const std::string& str);

// Category check helpers
constexpr bool isAudioError(ErrorCode code) {
    return (static_cast<uint32_t>(code) & 0xF000) == 0x1000;
}
constexpr bool isDacError(ErrorCode code) {
    return (static_cast<uint32_t>(code) & 0xF000) == 0x2000;
}
constexpr bool isIpcError(ErrorCode code) {
    return (static_cast<uint32_t>(code) & 0xF000) == 0x3000;
}
constexpr bool isGpuError(ErrorCode code) {
    return (static_cast<uint32_t>(code) & 0xF000) == 0x4000;
}
constexpr bool isValidationError(ErrorCode code) {
    return (static_cast<uint32_t>(code) & 0xF000) == 0x5000;
}
constexpr bool isInternalError(ErrorCode code) {
    return (static_cast<uint32_t>(code) & 0xF000) == 0xF000;
}

/**
 * @brief Check if error is retryable.
 * @param code Error code
 * @return true if operation can be retried
 *
 * Retryable errors (503/504 HTTP status):
 * - IPC_DAEMON_NOT_RUNNING
 * - IPC_TIMEOUT
 * - IPC_CONNECTION_FAILED
 */
constexpr bool isRetryable(ErrorCode code) {
    return code == ErrorCode::IPC_DAEMON_NOT_RUNNING || code == ErrorCode::IPC_TIMEOUT ||
           code == ErrorCode::IPC_CONNECTION_FAILED;
}

}  // namespace AudioEngine

#endif  // ERROR_CODES_H
