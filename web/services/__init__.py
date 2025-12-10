"""Services for the GPU Upsampler Web API."""

from .alsa import get_alsa_devices
from .config import (
    load_config,
    load_output_mode,
    load_partitioned_convolution_settings,
    save_config,
    save_output_mode,
    save_partitioned_convolution_settings,
    save_phase_type,
)
from .dac import (
    DacCapability,
    get_max_upsample_ratio,
    get_supported_output_rates,
    is_safe_device_name,
    scan_dac_capability,
)
from .daemon import (
    check_daemon_running,
    get_configured_rates,
    get_daemon_pid,
    load_stats,
    start_daemon,
    stop_daemon,
)
from .daemon_client import DaemonClient, get_daemon_client
from .eq import (
    is_safe_profile_name,
    parse_eq_profile_content,
    read_and_validate_upload,
    sanitize_filename,
    validate_eq_profile_content,
)
from .tcp_input import parse_tcp_telemetry, TcpTelemetryPoller, TcpTelemetryStore
from .rtp_bridge_client import (
    get_rtp_bridge_client,
    RtpBridgeClient,
    RtpBridgeClientError,
    RtpBridgeConnectionError,
    RtpBridgeResponseError,
)

__all__ = [
    # alsa
    "get_alsa_devices",
    # config
    "load_config",
    "load_output_mode",
    "load_partitioned_convolution_settings",
    "save_config",
    "save_output_mode",
    "save_partitioned_convolution_settings",
    "save_phase_type",
    # dac
    "DacCapability",
    "get_max_upsample_ratio",
    "get_supported_output_rates",
    "is_safe_device_name",
    "scan_dac_capability",
    # daemon
    "check_daemon_running",
    "get_configured_rates",
    "get_daemon_pid",
    "load_stats",
    "start_daemon",
    "stop_daemon",
    # daemon_client
    "DaemonClient",
    "get_daemon_client",
    # eq
    "is_safe_profile_name",
    "parse_eq_profile_content",
    "read_and_validate_upload",
    "sanitize_filename",
    "validate_eq_profile_content",
    # tcp input
    "parse_tcp_telemetry",
    "TcpTelemetryPoller",
    "TcpTelemetryStore",
    # rtp bridge
    "get_rtp_bridge_client",
    "RtpBridgeClient",
    "RtpBridgeClientError",
    "RtpBridgeConnectionError",
    "RtpBridgeResponseError",
]
