"""Pydantic models for the GPU Upsampler Web API."""

from typing import Annotated, Any, Literal, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    ConfigDict,
    StringConstraints,
    constr,
    model_validator,
)


# ============================================================================
# Core Settings Models
# ============================================================================

# HeadSize type for crossfeed settings (validated by Pydantic)
# Note: C++ side supports 5 sizes (xs/s/m/l/xl) and the UI exposes them.
HeadSize = Literal["xs", "s", "m", "l", "xl"]

# Common constrained types
SessionId = constr(
    strip_whitespace=True,
    min_length=1,
    max_length=64,
    pattern=r"^[A-Za-z0-9._-]+$",
)
Port = Annotated[int, Field(ge=1024, le=65535)]


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


OutputModeName = Literal["usb"]


class Settings(BaseModel):
    """Application settings model."""

    alsa_device: str = "default"
    upsample_ratio: int = 8
    buffer_size: int = 0
    period_size: int = 0
    gain: float = 1.0
    headroom_target: float = 0.0
    eq_enabled: bool = False
    eq_profile: Optional[str] = None
    eq_profile_path: Optional[str] = None
    input_rate: int = 44100
    output_rate: int = 352800
    crossfeed: CrossfeedSettings = Field(default_factory=CrossfeedSettings)


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


# ============================================================================
# Output Mode Models
# ============================================================================


class OutputModeUsbOptions(BaseModel):
    """USB-specific output mode options."""

    model_config = ConfigDict(extra="ignore")

    preferred_device: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("preferred_device", "preferredDevice"),
        serialization_alias="preferred_device",
        description="Preferred ALSA device identifier",
    )


class OutputModeOptions(BaseModel):
    """Container for per-mode option structures."""

    model_config = ConfigDict(extra="ignore")

    usb: OutputModeUsbOptions = Field(default_factory=OutputModeUsbOptions)


class OutputModeResponse(BaseModel):
    """Response payload for GET /api/output/mode."""

    mode: OutputModeName
    available_modes: list[OutputModeName]
    options: OutputModeOptions


class OutputModeUpdateRequest(BaseModel):
    """Request payload for POST /api/output/mode."""

    mode: OutputModeName = Field(default="usb")
    options: OutputModeOptions = Field(default_factory=OutputModeOptions)


class PartitionedConvolutionSettings(BaseModel):
    """Partitioned convolution (low latency) configuration."""

    enabled: bool = False
    fast_partition_taps: int = Field(default=10240, ge=1024, le=262_144)
    min_partition_taps: int = Field(default=10240, ge=1024, le=262_144)
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


class DelimiterStatus(BaseModel):
    """De-limiter runtime status."""

    enabled: bool = False
    backend_available: bool = False
    backend_valid: bool = False
    mode: str = "unknown"
    target_mode: str = "unknown"
    fallback_reason: str = "unknown"
    bypass_locked: bool = False
    warmup: bool = False
    queue_seconds: float = 0.0
    queue_samples: int = 0
    last_inference_ms: float = 0.0
    detail: Optional[str] = None


class XrunBreakdown(BaseModel):
    """XRUN breakdown across pipeline stages."""

    total: int = 0
    capture: int = 0
    processing: int = 0
    output: int = 0


class Status(BaseModel):
    """System status response model."""

    settings: Settings
    alsa_connected: bool = False
    clip_count: int = 0
    total_samples: int = 0
    clip_rate: float = 0.0
    daemon_running: bool = False
    eq_active: bool = False
    input_rate: int = 0
    output_rate: int = 0
    peaks: PeakLevels = Field(default_factory=PeakLevels)
    # Debug / diagnostics (from /tmp/gpu_upsampler_stats.json)
    xrun: XrunBreakdown = Field(default_factory=XrunBreakdown)
    xrun_count: int = 0
    buffer_underflows: int = 0
    buffer_overflows: int = 0
    buffer_capacity_frames: int = 0
    dropped_frames: int = 0
    rendered_silence_blocks: int = 0
    rendered_silence_frames: int = 0
    alsa_buffer_size_config: int = 0
    alsa_period_size_config: int = 0
    delimiter: Optional[DelimiterStatus] = None


class DaemonStatus(BaseModel):
    """Daemon status response model."""

    running: bool
    # NOTE:
    # - テストでは `DaemonStatus(running=True/False)` の最小構成を許容している。
    # - 実運用ではレスポンス生成側で値を埋める想定だが、モデルとしては optional にする。
    pid_file: Optional[str] = None
    binary_path: Optional[str] = None


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
        description="Warning message for linear phase (high latency ~0.45s @ 705.6kHz)",
    )


class PhaseTypeUpdateRequest(BaseModel):
    """Phase type update request model."""

    phase_type: PhaseType = Field(
        description="Target phase type: 'minimum' or 'linear'"
    )


# ============================================================================
# I2S Peer Status Models (#950)
# ============================================================================


class I2sPeerStatusUpdate(BaseModel):
    """Pi(usb-i2s-bridge) から Jetson Web へ送るステータス（更新用）."""

    running: bool = False
    mode: str = "none"  # capture / silence / none
    sample_rate: int = 0
    format: str = ""
    channels: int = 0
    generation: int = 0
    updated_at_unix_ms: int = 0
    note: Optional[str] = None


class I2sPeerStatus(I2sPeerStatusUpdate):
    """保存済みの peer status（参照用）."""

    received_at_unix_ms: int = 0


# ============================================================================
# Raspberry Pi Control API Models
# ============================================================================


class PiStatus(BaseModel):
    """Pi(USB-I2S bridge) のステータス."""

    running: bool
    mode: str
    sample_rate: int
    format: str
    channels: int
    xruns: int
    last_error: Optional[str] = None
    last_error_at_unix_ms: Optional[int] = None
    uptime_sec: float
    updated_at_unix_ms: Optional[int] = None


class PiUsbI2sConfig(BaseModel):
    """Pi 側 USB-I2S ブリッジの設定."""

    capture_device: str
    playback_device: str
    channels: int = Field(ge=1)
    fallback_rate: int = Field(ge=1)
    preferred_format: str
    alsa_buffer_time_us: int = Field(ge=1)
    alsa_latency_time_us: int = Field(ge=1)
    queue_time_ns: int = Field(ge=1)
    fade_ms: int = Field(ge=0)
    poll_interval_sec: float = Field(gt=0)
    restart_backoff_sec: float = Field(ge=0)
    keep_silence_when_no_capture: bool
    status_report_url: Optional[str] = None
    status_report_timeout_ms: int = Field(ge=1)
    status_report_min_interval_sec: float = Field(ge=0)
    control_endpoint: Optional[str] = None
    control_peer: Optional[str] = None
    control_require_peer: bool
    control_poll_interval_sec: float = Field(gt=0)
    control_timeout_ms: int = Field(ge=1)


class PiUsbI2sConfigUpdate(BaseModel):
    """Pi 側 USB-I2S ブリッジの設定更新."""

    capture_device: Optional[str] = None
    playback_device: Optional[str] = None
    channels: Optional[int] = Field(default=None, ge=1)
    fallback_rate: Optional[int] = Field(default=None, ge=1)
    preferred_format: Optional[str] = None
    alsa_buffer_time_us: Optional[int] = Field(default=None, ge=1)
    alsa_latency_time_us: Optional[int] = Field(default=None, ge=1)
    queue_time_ns: Optional[int] = Field(default=None, ge=1)
    fade_ms: Optional[int] = Field(default=None, ge=0)
    poll_interval_sec: Optional[float] = Field(default=None, gt=0)
    restart_backoff_sec: Optional[float] = Field(default=None, ge=0)
    keep_silence_when_no_capture: Optional[bool] = None
    status_report_url: Optional[str] = None
    status_report_timeout_ms: Optional[int] = Field(default=None, ge=1)
    status_report_min_interval_sec: Optional[float] = Field(default=None, ge=0)
    control_endpoint: Optional[str] = None
    control_peer: Optional[str] = None
    control_require_peer: Optional[bool] = None
    control_poll_interval_sec: Optional[float] = Field(default=None, gt=0)
    control_timeout_ms: Optional[int] = Field(default=None, ge=1)


# ============================================================================
# RTP Input Models
# ============================================================================


RtpEncoding = Literal["L16", "L24", "L32"]


class RtpInputSettings(BaseModel):
    """RTP入力の受信設定."""

    port: Port = 46000
    sample_rate: int = Field(default=44100, ge=8000, le=768000)
    channels: int = Field(default=2, ge=1, le=8)
    latency_ms: int = Field(default=100, ge=10, le=5000)
    encoding: RtpEncoding = "L24"
    device: str = "hw:Loopback,0,0"
    resample_quality: int = Field(default=8, ge=0, le=10)
    rtcp_port: Port = 46001
    rtcp_send_port: Port = 46002
    # USB直結(192.168.55.0/24)をデフォルト想定にする（mDNSが見えない構成が多い）
    sender_host: str = "192.168.55.100"


class RtpInputConfigUpdate(BaseModel):
    """RTP入力設定の更新リクエスト."""

    port: Port | None = None
    sample_rate: int | None = Field(default=None, ge=8000, le=768000)
    channels: int | None = Field(default=None, ge=1, le=8)
    latency_ms: int | None = Field(default=None, ge=10, le=5000)
    encoding: RtpEncoding | None = None
    device: Optional[str] = None
    resample_quality: int | None = Field(default=None, ge=0, le=10)
    rtcp_port: Port | None = None
    rtcp_send_port: Port | None = None
    sender_host: Optional[str] = None


class RtpInputStatus(BaseModel):
    """RTP入力のステータス."""

    running: bool
    pid: Optional[int] = None
    last_error: Optional[str] = None
    settings: RtpInputSettings


class RtpBridgeStatus(BaseModel):
    """ZeroMQ ブリッジ経由で返すRTP統計."""

    running: bool = False
    latency_ms: int = Field(default=100, ge=10, le=500)
    sample_rate: int = Field(default=44100, ge=0)
    packets_received: int = Field(default=0, ge=0)
    packets_lost: int = Field(default=0, ge=0)
    jitter_ms: float = Field(default=0.0, ge=0.0)
    clock_drift_ppm: float = 0.0


class RtpLatencyRequest(BaseModel):
    """レイテンシ変更リクエスト."""

    model_config = ConfigDict(extra="forbid")

    latency_ms: int = Field(ge=10, le=500)


class RtpLatencyResponse(BaseModel):
    """レイテンシ変更レスポンス."""

    status: str = "ok"
    latency_ms: int = Field(ge=10, le=500)


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


class OpraVendor(BaseModel):
    """OPRA vendor information model."""

    id: str
    name: str


class OpraEqProfileInfo(BaseModel):
    """OPRA EQ profile information model.

    Note: name field doesn't exist in OPRA data, use details or id as display name
    """

    id: str
    author: str = ""
    details: str = ""

    # Allow extra fields from OPRA data (type, parameters, product_id, etc.)
    model_config = {"extra": "allow"}


class OpraSearchResult(BaseModel):
    """OPRA search result item model."""

    id: str
    name: str
    type: str
    vendor: OpraVendor
    eq_profiles: list[OpraEqProfileInfo]


class OpraSearchResponse(BaseModel):
    """OPRA search results response model."""

    results: list[OpraSearchResult]
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


class OpraSyncMetadata(BaseModel):
    """Metadata for a synced OPRA database version."""

    commit_sha: str
    source: str
    source_url: str
    downloaded_at: str
    sha256: str
    size_bytes: int
    stats: dict[str, Any] = {}


class OpraSyncStatusResponse(BaseModel):
    """OPRA sync status response model."""

    status: str
    job_id: Optional[str] = None
    current_commit: Optional[str] = None
    previous_commit: Optional[str] = None
    last_updated_at: Optional[str] = None
    last_error: Optional[str] = None
    versions: list[str] = []
    current_metadata: Optional[OpraSyncMetadata] = None


class OpraSyncAvailableResponse(BaseModel):
    """OPRA sync availability response model."""

    source: str
    latest: str
    source_url: str


class OpraSyncUpdateRequest(BaseModel):
    """OPRA sync update request model."""

    target: str = Field(description="latest or commit SHA")
    source: Literal["github_raw", "cloudflare"]


class OpraSyncJobResponse(BaseModel):
    """OPRA sync job response model."""

    job_id: str
    status: str


# ============================================================================
# Request Models
# ============================================================================


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


class DelimiterActionResponse(BaseModel):
    """Response model for De-limiter enable/disable actions."""

    success: bool
    status: DelimiterStatus
    message: Optional[str] = None


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
