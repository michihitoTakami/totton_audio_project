"""Pydantic models for the GPU Upsampler Web API."""

import base64
import binascii
import ipaddress
import re
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    StringConstraints,
    conint,
    constr,
    field_validator,
    model_validator,
)


# ============================================================================
# Core Settings Models
# ============================================================================

# HeadSize type for crossfeed settings (validated by Pydantic)
HeadSize = Literal["s", "m", "l", "xl"]

# Common constrained types
SessionId = constr(
    strip_whitespace=True,
    min_length=1,
    max_length=64,
    pattern=r"^[A-Za-z0-9._-]+$",
)
Port = conint(ge=1024, le=65535)


class CrossfeedSettings(BaseModel):
    """Crossfeed settings model."""

    enabled: bool = False
    head_size: HeadSize = "m"
    hrtf_path: str = "data/crossfeed/hrtf/"


class CrossfeedSettingsUpdate(BaseModel):
    """Crossfeed settings update model (all fields optional)."""

    enabled: Optional[bool] = None
    head_size: Optional[HeadSize] = None
    hrtf_path: Optional[str] = None


InputMode = Literal["pipewire", "rtp"]


class Settings(BaseModel):
    """Application settings model."""

    alsa_device: str = "default"
    upsample_ratio: int = 8
    eq_enabled: bool = False
    eq_profile: Optional[str] = None
    eq_profile_path: Optional[str] = None
    input_rate: int = 44100
    output_rate: int = 352800
    crossfeed: CrossfeedSettings = Field(default_factory=CrossfeedSettings)
    rtp_enabled: bool = False


class SettingsUpdate(BaseModel):
    """Settings update request model."""

    alsa_device: Optional[str] = None
    upsample_ratio: Optional[int] = None
    eq_enabled: Optional[bool] = None
    eq_profile: Optional[str] = None
    eq_profile_path: Optional[str] = None
    input_rate: Optional[int] = None
    output_rate: Optional[int] = None
    crossfeed: Optional[CrossfeedSettingsUpdate] = None


class PartitionedConvolutionSettings(BaseModel):
    """Partitioned convolution (low latency) configuration."""

    enabled: bool = False
    fast_partition_taps: int = Field(default=32768, ge=1024, le=262_144)
    min_partition_taps: int = Field(default=32768, ge=1024, le=262_144)
    max_partitions: int = Field(default=4, ge=1, le=32)
    tail_fft_multiple: int = Field(default=2, ge=2, le=16)

    @model_validator(mode="after")
    def validate_relationships(self) -> "PartitionedConvolutionSettings":
        """Ensure partition parameters are internally consistent."""
        if self.min_partition_taps > self.fast_partition_taps:
            raise ValueError("min_partition_taps cannot exceed fast_partition_taps")
        return self


# ============================================================================
# Status Models
# ============================================================================


class PeakStage(BaseModel):
    """Single stage peak measurement."""

    linear: float = 0.0
    dbfs: float = -200.0


class PeakLevels(BaseModel):
    """Peak levels across pipeline stages."""

    input: PeakStage = Field(default_factory=PeakStage)
    upsampler: PeakStage = Field(default_factory=PeakStage)
    post_mix: PeakStage = Field(default_factory=PeakStage)
    post_gain: PeakStage = Field(default_factory=PeakStage)


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
    peaks: PeakLevels = Field(default_factory=PeakLevels)
    input_mode: InputMode = Field(
        default="pipewire", description="Current input mode reported by config.json"
    )


class DaemonStatus(BaseModel):
    """Daemon status response model."""

    running: bool
    pid: Optional[int] = None
    pid_file: str
    binary_path: str
    pipewire_connected: bool = False
    input_mode: InputMode = Field(
        default="pipewire", description="Current input mode reported by config.json"
    )


class ZmqPingResponse(BaseModel):
    """ZeroMQ ping response model."""

    success: bool
    response: Optional[Any] = None
    daemon_running: bool


PhaseType = Literal["minimum", "hybrid"]


class PhaseTypeResponse(BaseModel):
    """Phase type response model."""

    phase_type: PhaseType = Field(
        description="Current phase type: 'minimum' or 'hybrid'"
    )
    latency_warning: Optional[str] = Field(
        default=None,
        description="Info/warning message for hybrid phase (10ms alignment above 150 Hz)",
    )


class PhaseTypeUpdateRequest(BaseModel):
    """Phase type update request model."""

    phase_type: PhaseType = Field(
        description="Target phase type: 'minimum' or 'hybrid'"
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


class DacDaemonDevice(BaseModel):
    """Runtime DAC device info reported by daemon."""

    id: str
    card: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    is_requested: bool = False
    is_selected: bool = False
    is_active: bool = False


class DacCapabilitySnapshot(BaseModel):
    """Runtime DAC capability snapshot."""

    device: Optional[str] = None
    is_valid: bool = False
    min_rate: int = 0
    max_rate: int = 0
    max_channels: int = 0
    supported_rates: list[int] = Field(default_factory=list)
    error_message: Optional[str] = None
    alsa_errno: Optional[int] = None


class DacDaemonState(BaseModel):
    """Aggregated DAC state as reported by daemon."""

    requested_device: Optional[str] = None
    selected_device: Optional[str] = None
    active_device: Optional[str] = None
    change_pending: bool = False
    device_count: int = 0
    devices: list[DacDaemonDevice] = Field(default_factory=list)
    capability: Optional[DacCapabilitySnapshot] = None
    output_rate: Optional[int] = None
    last_event: Optional[dict[str, Any]] = None


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
    recommended_preamp_db: float = 0.0


class EqTextImportRequest(BaseModel):
    """Request body for text-based EQ profile import."""

    name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
    ]
    content: str = Field(description="Raw EQ profile text content")


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


class DacSelectRequest(BaseModel):
    """Request model for runtime DAC selection."""

    device: str
    persist: bool = False


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
# Input Mode Models
# ============================================================================


class InputModeSwitchRequest(BaseModel):
    """Request body for switching between PipeWire and RTP modes."""

    mode: InputMode = Field(
        description="Target mode: 'pipewire' to use local PipeWire input or 'rtp' for network input"
    )


class InputModeSwitchResponse(BaseModel):
    """Response returned after attempting to switch input modes."""

    success: bool
    current_mode: InputMode = Field(
        description="Mode currently active after the switch"
    )
    restart_required: bool = Field(
        default=False, description="True when the daemon was restarted to apply changes"
    )
    message: str


# ============================================================================
# Error Response Models (RFC 9457 Problem Details)
# ============================================================================


class InnerError(BaseModel):
    """Inner error details from lower layers (C++/ALSA/CUDA/ZeroMQ).

    This model captures error details from the C++ Audio Engine and IPC layer,
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
    zmq_errno: Optional[int] = Field(
        default=None, description="ZeroMQ error number (Python-side IPC errors)"
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


# ============================================================================
# RTP Models
# ============================================================================


def _validate_ipv4_literal(value: Optional[str]) -> Optional[str]:
    """Validate IPv4 literal strings (or return None)."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        ipaddress.IPv4Address(stripped)
    except ipaddress.AddressValueError as exc:  # pragma: no cover - defensive
        raise ValueError("Must be an IPv4 literal") from exc
    return stripped


class RtpEndpointSettings(BaseModel):
    """Networking options for an RTP session."""

    model_config = ConfigDict(extra="forbid")

    bind_address: str = Field(
        default="0.0.0.0",
        description="IPv4 address to bind for RTP reception",
    )
    port: int = Field(
        default=6000, description="UDP port for RTP payloads", ge=1024, le=65535
    )
    source_host: Optional[str] = Field(
        default=None,
        description="Optional IPv4 address filter for incoming packets",
    )
    multicast: bool = Field(default=False, description="Join multicast group")
    multicast_group: Optional[str] = Field(
        default=None, description="Multicast group address when multicast is enabled"
    )
    interface: Optional[str] = Field(
        default=None,
        description="Interface name or IPv4 address used for multicast subscription",
    )
    ttl: int = Field(
        default=32, description="TTL for multicast/QoS traffic", ge=1, le=255
    )
    dscp: int = Field(
        default=-1, description="DiffServ code point (-1 to disable)", ge=-1, le=63
    )

    @field_validator("bind_address", "source_host", "multicast_group")
    @classmethod
    def _validate_ipv4(cls, value: Optional[str]) -> Optional[str]:
        return _validate_ipv4_literal(value)


class RtpFormatSettings(BaseModel):
    """PCM payload description for RTP session."""

    model_config = ConfigDict(extra="forbid")

    sample_rate: int = Field(default=48000, description="Samples per second", gt=0)
    channels: int = Field(default=2, description="Channel count", ge=1, le=8)
    bits_per_sample: Literal[16, 24, 32] = Field(
        default=24, description="PCM word size"
    )
    payload_type: int = Field(
        default=97, description="Dynamic RTP payload type", ge=0, le=127
    )
    big_endian: bool = Field(default=True, description="Endianness of PCM payload")
    signed_samples: bool = Field(default=True, description="True if PCM is signed")


class RtpSyncSettings(BaseModel):
    """Latency and synchronization settings."""

    model_config = ConfigDict(extra="forbid")

    target_latency_ms: int = Field(
        default=5, description="Target buffering latency in milliseconds", ge=1, le=5000
    )
    watchdog_timeout_ms: int = Field(
        default=500,
        description="Watchdog timeout used for gap detection",
        ge=100,
        le=60000,
    )
    telemetry_interval_ms: int = Field(
        default=1000, description="Interval for telemetry emission", ge=100, le=60000
    )
    enable_ptp: bool = Field(default=False, description="Enable PTP clock tracking")
    ptp_interface: Optional[str] = Field(
        default=None, description="PTP network interface name"
    )
    ptp_domain: int = Field(
        default=0,
        description="PTP domain number (defaults to 0)",
    )


class RtpRtcpSettings(BaseModel):
    """RTCP monitoring options."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(default=True, description="Enable RTCP listener")
    port: Optional[int] = Field(
        default=None,
        description="Explicit RTCP port (defaults to RTP port + 1)",
        ge=1024,
        le=65535,
    )


class RtpAdvancedSettings(BaseModel):
    """Advanced buffer tuning for RTP reception."""

    model_config = ConfigDict(extra="forbid")

    socket_buffer_bytes: int = Field(
        default=1 << 20, description="SO_RCVBUF size", ge=65536, le=4 * 1024 * 1024
    )
    mtu_bytes: int = Field(
        default=1500, description="Expected MTU for incoming packets", ge=256, le=9000
    )


class RtpSecurityConfig(BaseModel):
    """Optional SRTP parameters for SDP generation."""

    model_config = ConfigDict(extra="forbid")

    crypto_suite: Literal["AES_CM_128_HMAC_SHA1_80", "AES_CM_128_HMAC_SHA1_32"] = Field(
        default="AES_CM_128_HMAC_SHA1_80", description="SRTP crypto suite"
    )
    key_base64: str = Field(
        min_length=40,
        description="Base64-encoded master key (per RFC 4568)",
    )

    @field_validator("key_base64")
    @classmethod
    def _validate_key(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9+/=]+", value):
            raise ValueError("Key must be base64 characters only")
        try:
            decoded = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:  # pragma: no cover - defensive
            raise ValueError("Invalid base64 SRTP key") from exc
        if len(decoded) < 16:
            raise ValueError("SRTP key must be at least 128 bits")
        return value


class RtpSdpConfig(BaseModel):
    """Optional SDP hints for RTP session creation."""

    model_config = ConfigDict(extra="forbid")

    body: Optional[str] = Field(
        default=None,
        description="Raw SDP string (overrides auto-generation when provided)",
    )
    session_name: str = Field(
        default="MagicBox RTP Session",
        description="Session name for auto-generated SDP",
    )
    connection_address: Optional[str] = Field(
        default=None,
        description="Override connection address for auto-generated SDP",
    )
    media_format: Optional[str] = Field(
        default=None,
        description="Override rtpmap payload descriptor (e.g., 'L24/48000/2')",
    )
    media_clock: Optional[str] = Field(
        default=None,
        description="Optional mediaclk attribute for deterministic streams",
    )

    @field_validator("connection_address")
    @classmethod
    def _validate_conn_addr(cls, value: Optional[str]) -> Optional[str]:
        return _validate_ipv4_literal(value)


class RtpSessionCreateRequest(BaseModel):
    """Request body for creating a new RTP session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        description="Unique identifier for the RTP session",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    endpoint: RtpEndpointSettings = Field(
        default_factory=RtpEndpointSettings,
        description="Endpoint and multicast configuration",
    )
    format: RtpFormatSettings = Field(
        default_factory=RtpFormatSettings,
        description="PCM payload description",
    )
    sync: RtpSyncSettings = Field(
        default_factory=RtpSyncSettings, description="Latency / sync parameters"
    )
    rtcp: RtpRtcpSettings = Field(
        default_factory=RtpRtcpSettings, description="RTCP monitoring configuration"
    )
    advanced: RtpAdvancedSettings = Field(
        default_factory=RtpAdvancedSettings,
        description="Advanced socket / MTU settings",
    )
    sdp: Optional[RtpSdpConfig] = Field(
        default=None, description="Optional SDP override"
    )
    security: Optional[RtpSecurityConfig] = Field(
        default=None, description="Optional SRTP settings"
    )


class RtpSessionConfigSnapshot(BaseModel):
    """Session configuration echoed by the daemon."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    bind_address: str
    port: int
    source_host: Optional[str] = None
    multicast: Optional[bool] = None
    multicast_group: Optional[str] = None
    interface: Optional[str] = None
    ttl: Optional[int] = None
    dscp: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bits_per_sample: Optional[int] = None
    big_endian: Optional[bool] = None
    signed: Optional[bool] = None
    payload_type: Optional[int] = None
    socket_buffer_bytes: Optional[int] = None
    mtu_bytes: Optional[int] = None
    target_latency_ms: Optional[int] = None
    watchdog_timeout_ms: Optional[int] = None
    telemetry_interval_ms: Optional[int] = None
    enable_rtcp: Optional[bool] = None
    rtcp_port: Optional[int] = None
    enable_ptp: Optional[bool] = None
    ptp_interface: Optional[str] = None
    ptp_domain: Optional[int] = None
    sdp: Optional[str] = None

    @classmethod
    def from_daemon(cls, payload: dict[str, Any]) -> "RtpSessionConfigSnapshot":
        return cls(**payload)


class RtpSessionMetrics(BaseModel):
    """Live RTP telemetry reported by the daemon."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    bind_address: Optional[str] = None
    port: Optional[int] = None
    source_host: Optional[str] = None
    multicast: Optional[bool] = None
    multicast_group: Optional[str] = None
    interface: Optional[str] = None
    payload_type: Optional[int] = None
    channels: Optional[int] = None
    bits_per_sample: Optional[int] = None
    big_endian: Optional[bool] = None
    signed: Optional[bool] = None
    enable_rtcp: Optional[bool] = None
    rtcp_port: Optional[int] = None
    enable_ptp: Optional[bool] = None
    target_latency_ms: Optional[int] = None
    watchdog_timeout_ms: Optional[int] = None
    telemetry_interval_ms: Optional[int] = None
    auto_start: bool = False
    ssrc: Optional[int] = None
    ssrc_locked: bool = False
    packets_received: int = 0
    packets_dropped: int = 0
    sequence_resets: int = 0
    bytes_received: int = 0
    rtcp_packets: int = 0
    late_packets: int = 0
    avg_transit_usec: float = 0.0
    network_jitter_usec: float = 0.0
    ptp_offset_ns: float = 0.0
    ptp_mean_path_ns: float = 0.0
    ptp_locked: bool = False
    sample_rate: Optional[int] = None
    last_rtp_timestamp: Optional[int] = None
    last_packet_unix_ms: Optional[int] = None
    updated_at_unix_ms: Optional[int] = Field(
        default=None, description="Timestamp (ms) when telemetry was sampled"
    )

    @classmethod
    def from_daemon(
        cls, payload: dict[str, Any], *, polled_at_unix_ms: Optional[int] = None
    ) -> "RtpSessionMetrics":
        data = payload.copy()
        if polled_at_unix_ms is not None:
            data.setdefault("updated_at_unix_ms", polled_at_unix_ms)
        return cls(**data)


class RtpSessionCreateResponse(BaseModel):
    """Response returned when an RTP session is created."""

    success: bool = Field(default=True, description="True when the session is created")
    message: str = Field(default="RTP session started")
    session: RtpSessionConfigSnapshot = Field(
        description="Normalized SessionConfig echoed by the daemon"
    )


class RtpSessionDetailResponse(BaseModel):
    """Response for GET /api/rtp/sessions/{id}."""

    session: RtpSessionMetrics
    polled_at_unix_ms: Optional[int] = None


class RtpSessionListResponse(BaseModel):
    """Response for GET /api/rtp/sessions."""

    sessions: list[RtpSessionMetrics]
    polled_at_unix_ms: Optional[int] = None


class RtpDiscoveryStream(BaseModel):
    """Single RTP stream candidate discovered by the daemon."""

    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(description="Suggested session identifier")
    display_name: str = Field(description="Human-friendly label for the stream")
    source_host: Optional[str] = Field(
        default=None, description="Source IPv4 host of the RTP sender"
    )
    port: int = Field(
        description="RTP payload port reported by the scanner", ge=1024, le=65535
    )
    status: str = Field(
        default="unknown",
        description="Scanner-reported status such as 'active', 'idle', or diagnostic text",
    )
    existing_session: bool = Field(
        default=False,
        description="True when this candidate already has an active RTP session",
    )
    sample_rate: Optional[int] = Field(
        default=None, description="Sample rate hint to pre-fill new RTP sessions"
    )
    channels: Optional[int] = Field(
        default=None, description="Channel count hint to pre-fill new RTP sessions"
    )
    payload_type: Optional[int] = Field(
        default=None, description="Dynamic payload type hint from scanner"
    )
    multicast: bool = Field(
        default=False,
        description="True if the stream is multicast and requires group subscription",
    )
    multicast_group: Optional[str] = Field(
        default=None, description="Multicast group address when multicast is enabled"
    )
    bind_address: Optional[str] = Field(
        default=None,
        description="Suggested local bind address for receiving the stream",
    )
    last_seen_unix_ms: Optional[int] = Field(
        default=None,
        description="Unix timestamp (ms) when the stream was last observed",
    )
    latency_ms: Optional[int] = Field(
        default=None, description="Estimated transport latency reported by scanner"
    )

    @field_validator("source_host", "multicast_group", "bind_address")
    @classmethod
    def _validate_ipv4(cls, value: Optional[str]) -> Optional[str]:
        return _validate_ipv4_literal(value)


class RtpDiscoveryResponse(BaseModel):
    """Response payload for RTP discovery scans."""

    streams: list[RtpDiscoveryStream] = Field(
        default_factory=list, description="List of discovered RTP senders"
    )
    scanned_at_unix_ms: Optional[int] = Field(
        default=None, description="Unix timestamp (ms) when the scan completed"
    )
    duration_ms: Optional[int] = Field(
        default=None, description="Approximate scan duration in milliseconds"
    )


# ============================================================================
# Crossfeed Models
# ============================================================================


class CrossfeedStatus(BaseModel):
    """Crossfeed status response model."""

    enabled: bool = Field(description="Whether crossfeed is currently enabled")
    headSize: Optional[str] = Field(
        default=None, description="Current head size: 'xs', 's', 'm', 'l', or 'xl'"
    )
    availableSizes: list[str] = Field(
        default_factory=lambda: ["xs", "s", "m", "l", "xl"],
        description="Available head size options",
    )


class CrossfeedEnableResponse(BaseModel):
    """Crossfeed enable response model."""

    success: bool
    message: str


class CrossfeedDisableResponse(BaseModel):
    """Crossfeed disable response model."""

    success: bool
    message: str


class CrossfeedSizeResponse(BaseModel):
    """Crossfeed size change response model."""

    success: bool
    headSize: str
