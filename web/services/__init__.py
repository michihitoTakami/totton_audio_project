"""Services for the GPU Upsampler Web API."""

from .alsa import get_alsa_devices
from .config import load_config, save_config
from .dac import (
    DacCapability,
    get_max_upsample_ratio,
    get_supported_output_rates,
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

__all__ = [
    # alsa
    "get_alsa_devices",
    # config
    "load_config",
    "save_config",
    # dac
    "DacCapability",
    "get_max_upsample_ratio",
    "get_supported_output_rates",
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
]
