"""Pydantic models for the GPU Upsampler Web API."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Core Settings Models
# ============================================================================


class Settings(BaseModel):
    """Application settings model."""

    alsa_device: str = "default"
    upsample_ratio: int = 8
    eq_profile: Optional[str] = None
    input_rate: int = 44100
    output_rate: int = 352800


class SettingsUpdate(BaseModel):
    """Settings update request model."""

    alsa_device: Optional[str] = None
    upsample_ratio: Optional[int] = None
    eq_profile: Optional[str] = None
    input_rate: Optional[int] = None
    output_rate: Optional[int] = None


# ============================================================================
# Status Models
# ============================================================================


class Status(BaseModel):
    """System status response model."""

    settings: Settings
    pipewire_connected: bool = False
    alsa_connected: bool = False
    clip_count: int = 0
    total_samples: int = 0
    clip_rate: float = 0.0
    daemon_running: bool = False
    eq_active: bool = False
    input_rate: int = 0
    output_rate: int = 0


class DaemonStatus(BaseModel):
    """Daemon status response model."""

    running: bool
    pid: Optional[int] = None
    pid_file: str
    binary_path: str
    pipewire_connected: bool = False


class ZmqPingResponse(BaseModel):
    """ZeroMQ ping response model."""

    success: bool
    response: Optional[Any] = None
    daemon_running: bool


PhaseType = Literal["minimum", "linear"]


class PhaseTypeResponse(BaseModel):
    """Phase type response model."""

    phase_type: PhaseType = Field(
        description="Current phase type: 'minimum' or 'linear'"
    )
    latency_warning: Optional[str] = Field(
        default=None,
        description="Warning message for linear phase (high latency ~1 second)",
    )


class PhaseTypeUpdateRequest(BaseModel):
    """Phase type update request model."""

    phase_type: PhaseType = Field(
        description="Target phase type: 'minimum' or 'linear'"
    )


# ============================================================================
# Device Models
# ============================================================================


class AlsaDevice(BaseModel):
    """ALSA device info model."""

    id: str
    name: str
    description: Optional[str] = None


class DevicesResponse(BaseModel):
    """Available ALSA devices response model."""

    devices: list[AlsaDevice]


class DacCapabilityResponse(BaseModel):
    """DAC capability response model."""

    device_name: str
    min_sample_rate: int
    max_sample_rate: int
    supported_rates: list[int]
    max_channels: int
    is_valid: bool
    error_message: Optional[str] = None


class DacCapabilityInfo(BaseModel):
    """DAC capability info for device list."""

    min_sample_rate: int
    max_sample_rate: int
    supported_rates: list[int]
    max_channels: int


class DacDeviceInfo(BaseModel):
    """DAC device with capabilities."""

    id: str
    name: str
    description: Optional[str] = None
    capabilities: Optional[DacCapabilityInfo] = None


class DacDevicesResponse(BaseModel):
    """DAC devices list response model."""

    devices: list[DacDeviceInfo]


class DacSupportedRatesResponse(BaseModel):
    """Supported rates for a rate family response model."""

    device: str
    family: str = Field(description="Rate family: '44k' or '48k'")
    supported_rates: list[int]


class DacMaxRatioResponse(BaseModel):
    """Maximum upsampling ratio response model."""

    device: str
    input_rate: int
    max_ratio: int
    max_output_rate: int


# ============================================================================
# EQ Profile Models
# ============================================================================


class EqProfileInfo(BaseModel):
    """EQ profile info model."""

    name: str
    filename: str
    path: str
    size: int
    modified: float
    type: str = Field(description="Profile type: 'opra' or 'custom'")
    filter_count: int


class EqProfilesResponse(BaseModel):
    """EQ profiles list response model."""

    profiles: list[EqProfileInfo]


class EqValidationResponse(BaseModel):
    """EQ profile validation response model."""

    valid: bool
    errors: list[str] = []
    warnings: list[str] = []
    preamp_db: Optional[float] = None
    filter_count: int = 0
    filename: str
    file_exists: bool
    size_bytes: int


class EqActiveResponse(BaseModel):
    """Active EQ profile response model."""

    active: bool
    name: Optional[str] = None
    error: Optional[str] = None
    source_type: Optional[str] = None
    has_modern_target: bool = False
    opra_info: Optional[dict[str, Any]] = None
    opra_filters: list[str] = []
    original_filters: list[str] = []


# ============================================================================
# OPRA Models
# ============================================================================


class OpraStats(BaseModel):
    """OPRA database statistics response model."""

    vendors: int
    products: int
    eq_profiles: int
    license: str = "CC BY-SA 4.0"
    attribution: str = "OPRA Project (https://github.com/opra-project/OPRA)"


class OpraVendorsResponse(BaseModel):
    """OPRA vendors list response model."""

    vendors: list[str]
    count: int


class OpraSearchResponse(BaseModel):
    """OPRA search results response model."""

    results: list[dict[str, Any]]
    count: int
    query: str


class OpraEqAttribution(BaseModel):
    """OPRA EQ attribution model."""

    license: str = "CC BY-SA 4.0"
    source: str = "OPRA Project"
    author: str


class OpraEqResponse(BaseModel):
    """OPRA EQ profile response model."""

    id: str
    name: str
    author: str
    details: str
    parameters: dict[str, Any] = {}
    apo_format: str
    modern_target_applied: bool
    attribution: OpraEqAttribution


# ============================================================================
# Request Models
# ============================================================================


class RewireRequest(BaseModel):
    """Request model for rewiring PipeWire connections."""

    source_node: str
    target_node: str


# ============================================================================
# Standard Response Models
# ============================================================================


class ApiResponse(BaseModel):
    """Standard API response model for mutations."""

    success: bool
    message: str
    data: Optional[dict[str, Any]] = None
    restart_required: bool = False


# ============================================================================
# Error Response Models (RFC 9457 Problem Details)
# ============================================================================


class InnerError(BaseModel):
    """Inner error details from lower layers (C++/ALSA/CUDA).

    This model captures error details from the C++ Audio Engine,
    preserving diagnostic information for debugging.
    """

    cpp_code: Optional[str] = Field(
        default=None, description="C++ error code in hex (e.g., '0x2004')"
    )
    cpp_message: Optional[str] = Field(
        default=None, description="Original C++ error message"
    )
    alsa_errno: Optional[int] = Field(
        default=None, description="ALSA error number (negative values)"
    )
    alsa_func: Optional[str] = Field(
        default=None, description="ALSA function that failed"
    )
    cuda_error: Optional[str] = Field(
        default=None, description="CUDA error name (e.g., 'cudaErrorMemoryAllocation')"
    )


class ErrorResponse(BaseModel):
    """RFC 9457 Problem Details compliant error response.

    Content-Type: application/problem+json

    Example:
        {
            "type": "/errors/dac-rate-not-supported",
            "title": "DAC Rate Not Supported",
            "status": 422,
            "detail": "Sample rate 1000000 is not supported by DAC",
            "error_code": "DAC_RATE_NOT_SUPPORTED",
            "category": "dac_alsa",
            "inner_error": {"cpp_code": "0x2004", "alsa_errno": -22}
        }
    """

    type: Optional[str] = Field(
        default=None,
        description="URI reference identifying the problem type (e.g., '/errors/dac-rate-not-supported')",
    )
    title: Optional[str] = Field(
        default=None, description="Short human-readable summary of the problem"
    )
    status: Optional[int] = Field(
        default=None, description="HTTP status code for this error"
    )
    detail: str | dict[str, Any] = Field(description="Human-readable error description")
    error_code: Optional[str] = Field(
        default=None,
        description="Application-specific error code (e.g., 'DAC_RATE_NOT_SUPPORTED')",
    )
    category: Optional[str] = Field(
        default=None, description="Error category (e.g., 'dac_alsa', 'ipc_zeromq')"
    )
    inner_error: Optional[InnerError] = Field(
        default=None, description="Nested error details from lower layers"
    )
