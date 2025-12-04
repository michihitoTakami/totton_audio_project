"""Error codes for C++ Audio Engine integration.

This module provides Python equivalents of C++ error codes defined in
docs/architecture/error-codes.md. All codes are synchronized with
include/error_codes.h for consistent error handling across layers.
"""

from dataclasses import dataclass
from enum import Enum


class ErrorCategory(str, Enum):
    """Error category classification.

    Categories match C++ implementation and are used for:
    - Grouping related errors
    - Determining appropriate HTTP status codes
    - Logging and monitoring purposes
    """

    AUDIO_PROCESSING = "audio_processing"
    DAC_ALSA = "dac_alsa"
    IPC_ZEROMQ = "ipc_zeromq"
    GPU_CUDA = "gpu_cuda"
    VALIDATION = "validation"
    CROSSFEED = "crossfeed"  # #150
    INTERNAL = "internal"


class ErrorCode(str, Enum):
    """Error codes synchronized with C++ AudioEngine::ErrorCode.

    Each code maps to a 16-bit hex value in C++ (e.g., AUDIO_INVALID_INPUT_RATE = 0x1001).
    Python uses string names for readability; mapping to HTTP status is done via ERROR_MAPPINGS.
    """

    # Audio Processing (0x1000)
    AUDIO_INVALID_INPUT_RATE = "AUDIO_INVALID_INPUT_RATE"
    AUDIO_INVALID_OUTPUT_RATE = "AUDIO_INVALID_OUTPUT_RATE"
    AUDIO_UNSUPPORTED_FORMAT = "AUDIO_UNSUPPORTED_FORMAT"
    AUDIO_FILTER_NOT_FOUND = "AUDIO_FILTER_NOT_FOUND"
    AUDIO_BUFFER_OVERFLOW = "AUDIO_BUFFER_OVERFLOW"
    AUDIO_XRUN_DETECTED = "AUDIO_XRUN_DETECTED"
    AUDIO_RTP_SOCKET_ERROR = "AUDIO_RTP_SOCKET_ERROR"
    AUDIO_RTP_SESSION_NOT_FOUND = "AUDIO_RTP_SESSION_NOT_FOUND"

    # DAC/ALSA (0x2000)
    DAC_DEVICE_NOT_FOUND = "DAC_DEVICE_NOT_FOUND"
    DAC_OPEN_FAILED = "DAC_OPEN_FAILED"
    DAC_CAPABILITY_SCAN_FAILED = "DAC_CAPABILITY_SCAN_FAILED"
    DAC_RATE_NOT_SUPPORTED = "DAC_RATE_NOT_SUPPORTED"
    DAC_FORMAT_NOT_SUPPORTED = "DAC_FORMAT_NOT_SUPPORTED"
    DAC_BUSY = "DAC_BUSY"

    # IPC/ZeroMQ (0x3000)
    IPC_CONNECTION_FAILED = "IPC_CONNECTION_FAILED"
    IPC_TIMEOUT = "IPC_TIMEOUT"
    IPC_INVALID_COMMAND = "IPC_INVALID_COMMAND"
    IPC_INVALID_PARAMS = "IPC_INVALID_PARAMS"
    IPC_DAEMON_NOT_RUNNING = "IPC_DAEMON_NOT_RUNNING"
    IPC_PROTOCOL_ERROR = "IPC_PROTOCOL_ERROR"

    # GPU/CUDA (0x4000)
    GPU_INIT_FAILED = "GPU_INIT_FAILED"
    GPU_DEVICE_NOT_FOUND = "GPU_DEVICE_NOT_FOUND"
    GPU_MEMORY_ERROR = "GPU_MEMORY_ERROR"
    GPU_KERNEL_LAUNCH_FAILED = "GPU_KERNEL_LAUNCH_FAILED"
    GPU_FILTER_LOAD_FAILED = "GPU_FILTER_LOAD_FAILED"
    GPU_CUFFT_ERROR = "GPU_CUFFT_ERROR"

    # Validation (0x5000)
    VALIDATION_INVALID_CONFIG = "VALIDATION_INVALID_CONFIG"
    VALIDATION_INVALID_PROFILE = "VALIDATION_INVALID_PROFILE"
    VALIDATION_PATH_TRAVERSAL = "VALIDATION_PATH_TRAVERSAL"
    VALIDATION_FILE_NOT_FOUND = "VALIDATION_FILE_NOT_FOUND"
    VALIDATION_PROFILE_EXISTS = "VALIDATION_PROFILE_EXISTS"
    VALIDATION_INVALID_HEADPHONE = "VALIDATION_INVALID_HEADPHONE"
    ERR_UNSUPPORTED_MODE = "ERR_UNSUPPORTED_MODE"

    # Crossfeed/HRTF (0x6000) - #150
    CROSSFEED_NOT_INITIALIZED = "CROSSFEED_NOT_INITIALIZED"
    CROSSFEED_INVALID_RATE_FAMILY = "CROSSFEED_INVALID_RATE_FAMILY"
    CROSSFEED_NOT_IMPLEMENTED = "CROSSFEED_NOT_IMPLEMENTED"
    CROSSFEED_INVALID_FILTER_SIZE = "CROSSFEED_INVALID_FILTER_SIZE"


@dataclass(frozen=True)
class ErrorMapping:
    """Mapping from error code to HTTP response details."""

    http_status: int
    category: ErrorCategory
    title: str


# Complete mapping of all 30 error codes to HTTP status codes
# Based on docs/architecture/error-codes.md §5
ERROR_MAPPINGS: dict[ErrorCode, ErrorMapping] = {
    # Audio Processing (8 codes)
    ErrorCode.AUDIO_INVALID_INPUT_RATE: ErrorMapping(
        400, ErrorCategory.AUDIO_PROCESSING, "Invalid Input Sample Rate"
    ),
    ErrorCode.AUDIO_INVALID_OUTPUT_RATE: ErrorMapping(
        400, ErrorCategory.AUDIO_PROCESSING, "Invalid Output Sample Rate"
    ),
    ErrorCode.AUDIO_UNSUPPORTED_FORMAT: ErrorMapping(
        400, ErrorCategory.AUDIO_PROCESSING, "Unsupported Audio Format"
    ),
    ErrorCode.AUDIO_FILTER_NOT_FOUND: ErrorMapping(
        404, ErrorCategory.AUDIO_PROCESSING, "Filter Not Found"
    ),
    ErrorCode.AUDIO_BUFFER_OVERFLOW: ErrorMapping(
        500, ErrorCategory.AUDIO_PROCESSING, "Buffer Overflow"
    ),
    ErrorCode.AUDIO_XRUN_DETECTED: ErrorMapping(
        500, ErrorCategory.AUDIO_PROCESSING, "Audio XRUN Detected"
    ),
    ErrorCode.AUDIO_RTP_SOCKET_ERROR: ErrorMapping(
        500, ErrorCategory.AUDIO_PROCESSING, "RTP Socket Error"
    ),
    ErrorCode.AUDIO_RTP_SESSION_NOT_FOUND: ErrorMapping(
        404, ErrorCategory.AUDIO_PROCESSING, "RTP Session Not Found"
    ),
    # DAC/ALSA (6 codes)
    ErrorCode.DAC_DEVICE_NOT_FOUND: ErrorMapping(
        404, ErrorCategory.DAC_ALSA, "DAC Device Not Found"
    ),
    ErrorCode.DAC_OPEN_FAILED: ErrorMapping(
        500, ErrorCategory.DAC_ALSA, "DAC Open Failed"
    ),
    ErrorCode.DAC_CAPABILITY_SCAN_FAILED: ErrorMapping(
        500, ErrorCategory.DAC_ALSA, "DAC Capability Scan Failed"
    ),
    ErrorCode.DAC_RATE_NOT_SUPPORTED: ErrorMapping(
        422, ErrorCategory.DAC_ALSA, "DAC Rate Not Supported"
    ),
    ErrorCode.DAC_FORMAT_NOT_SUPPORTED: ErrorMapping(
        422, ErrorCategory.DAC_ALSA, "DAC Format Not Supported"
    ),
    ErrorCode.DAC_BUSY: ErrorMapping(409, ErrorCategory.DAC_ALSA, "DAC Device Busy"),
    # IPC/ZeroMQ (6 codes)
    ErrorCode.IPC_CONNECTION_FAILED: ErrorMapping(
        503, ErrorCategory.IPC_ZEROMQ, "Daemon Connection Failed"
    ),
    ErrorCode.IPC_TIMEOUT: ErrorMapping(
        504, ErrorCategory.IPC_ZEROMQ, "Daemon Timeout"
    ),
    ErrorCode.IPC_INVALID_COMMAND: ErrorMapping(
        400, ErrorCategory.IPC_ZEROMQ, "Invalid Command"
    ),
    ErrorCode.IPC_INVALID_PARAMS: ErrorMapping(
        400, ErrorCategory.IPC_ZEROMQ, "Invalid Parameters"
    ),
    ErrorCode.IPC_DAEMON_NOT_RUNNING: ErrorMapping(
        503, ErrorCategory.IPC_ZEROMQ, "Daemon Not Running"
    ),
    ErrorCode.IPC_PROTOCOL_ERROR: ErrorMapping(
        500, ErrorCategory.IPC_ZEROMQ, "Protocol Error"
    ),
    # GPU/CUDA (6 codes)
    ErrorCode.GPU_INIT_FAILED: ErrorMapping(
        500, ErrorCategory.GPU_CUDA, "GPU Initialization Failed"
    ),
    ErrorCode.GPU_DEVICE_NOT_FOUND: ErrorMapping(
        500, ErrorCategory.GPU_CUDA, "GPU Device Not Found"
    ),
    ErrorCode.GPU_MEMORY_ERROR: ErrorMapping(
        500, ErrorCategory.GPU_CUDA, "GPU Memory Error"
    ),
    ErrorCode.GPU_KERNEL_LAUNCH_FAILED: ErrorMapping(
        500, ErrorCategory.GPU_CUDA, "GPU Kernel Launch Failed"
    ),
    ErrorCode.GPU_FILTER_LOAD_FAILED: ErrorMapping(
        500, ErrorCategory.GPU_CUDA, "GPU Filter Load Failed"
    ),
    ErrorCode.GPU_CUFFT_ERROR: ErrorMapping(500, ErrorCategory.GPU_CUDA, "cuFFT Error"),
    # Validation (6 codes)
    ErrorCode.VALIDATION_INVALID_CONFIG: ErrorMapping(
        400, ErrorCategory.VALIDATION, "Invalid Configuration"
    ),
    ErrorCode.VALIDATION_INVALID_PROFILE: ErrorMapping(
        400, ErrorCategory.VALIDATION, "Invalid EQ Profile"
    ),
    ErrorCode.VALIDATION_PATH_TRAVERSAL: ErrorMapping(
        400, ErrorCategory.VALIDATION, "Path Traversal Detected"
    ),
    ErrorCode.VALIDATION_FILE_NOT_FOUND: ErrorMapping(
        404, ErrorCategory.VALIDATION, "File Not Found"
    ),
    ErrorCode.VALIDATION_PROFILE_EXISTS: ErrorMapping(
        409, ErrorCategory.VALIDATION, "Profile Already Exists"
    ),
    ErrorCode.VALIDATION_INVALID_HEADPHONE: ErrorMapping(
        404, ErrorCategory.VALIDATION, "Headphone Not Found in OPRA DB"
    ),
    ErrorCode.ERR_UNSUPPORTED_MODE: ErrorMapping(
        400, ErrorCategory.VALIDATION, "Unsupported Output Mode"
    ),
    # Crossfeed/HRTF (4 codes) - #150
    ErrorCode.CROSSFEED_NOT_INITIALIZED: ErrorMapping(
        503, ErrorCategory.CROSSFEED, "Crossfeed Not Initialized"
    ),
    ErrorCode.CROSSFEED_INVALID_RATE_FAMILY: ErrorMapping(
        400, ErrorCategory.CROSSFEED, "Invalid Rate Family"
    ),
    ErrorCode.CROSSFEED_NOT_IMPLEMENTED: ErrorMapping(
        501, ErrorCategory.CROSSFEED, "Crossfeed Filter Application Not Implemented"
    ),
    ErrorCode.CROSSFEED_INVALID_FILTER_SIZE: ErrorMapping(
        400, ErrorCategory.CROSSFEED, "Invalid Filter Size"
    ),
}

# Default mapping for unknown error codes
_DEFAULT_MAPPING = ErrorMapping(500, ErrorCategory.INTERNAL, "Internal Error")


def get_error_mapping(error_code: str) -> ErrorMapping:
    """Get error mapping for a given error code string.

    Args:
        error_code: Error code string (e.g., "DAC_RATE_NOT_SUPPORTED")

    Returns:
        ErrorMapping with http_status, category, and title.
        Returns default 500/INTERNAL mapping for unknown codes.
    """
    try:
        code = ErrorCode(error_code)
        return ERROR_MAPPINGS.get(code, _DEFAULT_MAPPING)
    except ValueError:
        # Unknown error code - return internal error
        return _DEFAULT_MAPPING


def get_category_from_code(error_code: str) -> ErrorCategory:
    """Determine category from error code prefix.

    This is useful when the error code is known but not in ERROR_MAPPINGS.
    Falls back to INTERNAL for unknown prefixes.
    """
    if error_code.startswith("AUDIO_"):
        return ErrorCategory.AUDIO_PROCESSING
    elif error_code.startswith("DAC_"):
        return ErrorCategory.DAC_ALSA
    elif error_code.startswith("IPC_"):
        return ErrorCategory.IPC_ZEROMQ
    elif error_code.startswith("GPU_"):
        return ErrorCategory.GPU_CUDA
    elif error_code.startswith("VALIDATION_"):
        return ErrorCategory.VALIDATION
    elif error_code.startswith("CROSSFEED_"):
        return ErrorCategory.CROSSFEED
    else:
        return ErrorCategory.INTERNAL
