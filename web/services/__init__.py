"""Services for the GPU Upsampler Web API."""

from .alsa import get_alsa_devices
from .config import (
    load_config,
    load_partitioned_convolution_settings,
    save_config,
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
    check_pipewire_sink,
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
from .pipewire import (
    restore_default_sink,
    setup_audio_routing,
    setup_pipewire_links,
    wait_for_daemon_node,
)
from .rtp import (
    build_session_config_payload,
    parse_config_snapshot,
    parse_metrics_payload,
    refresh_sessions_from_daemon,
    telemetry_poller,
    telemetry_store,
)

__all__ = [
    # alsa
    "get_alsa_devices",
    # config
    "load_config",
    "load_partitioned_convolution_settings",
    "save_config",
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
    "check_pipewire_sink",
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
    # pipewire
    "restore_default_sink",
    "setup_audio_routing",
    "setup_pipewire_links",
    "wait_for_daemon_node",
    # rtp
    "build_session_config_payload",
    "parse_config_snapshot",
    "parse_metrics_payload",
    "refresh_sessions_from_daemon",
    "telemetry_poller",
    "telemetry_store",
]
