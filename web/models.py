"""Pydantic models for the GPU Upsampler Web API."""

from typing import Any, Optional

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


class ErrorResponse(BaseModel):
    """Standard error response model.

    Supports both string and structured error details for flexibility.
    """

    detail: str | dict[str, Any]
    error_code: Optional[str] = None
