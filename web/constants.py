"""Constants for the GPU Upsampler Web API."""

import re
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

WEB_DIR = Path(__file__).parent
PROJECT_ROOT = WEB_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"
EQ_PROFILES_DIR = PROJECT_ROOT / "data" / "EQ"
DAEMON_BINARY = PROJECT_ROOT / "build" / "gpu_upsampler_alsa"
PID_FILE_PATH = Path("/tmp/gpu_upsampler_alsa.pid")
STATS_FILE_PATH = Path("/tmp/gpu_upsampler_stats.json")

# ============================================================================
# ZeroMQ
# ============================================================================

ZEROMQ_IPC_PATH = "ipc:///tmp/gpu_os.sock"

# ============================================================================
# Daemon
# ============================================================================

DAEMON_SERVICE = "gpu_upsampler_alsa"  # systemd service name (if using systemd)

# ============================================================================
# EQ Profile Upload Security Limits
# ============================================================================

MAX_EQ_FILE_SIZE = 1 * 1024 * 1024  # 1MB
MAX_EQ_FILTERS = 100
PREAMP_MIN_DB = -100.0
PREAMP_MAX_DB = 20.0
FREQ_MIN_HZ = 20.0
FREQ_MAX_HZ = 24000.0
GAIN_MIN_DB = -30.0
GAIN_MAX_DB = 30.0
Q_MIN = 0.1
Q_MAX = 100.0

# Allowed filename pattern: alphanumeric, underscore, hyphen, dot (with .txt extension)
SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+\.txt$")

# Allowed profile name pattern: alphanumeric, underscore, hyphen, dot (no extension)
# Used for EQ profile names in config and URL parameters
SAFE_PROFILE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

# ============================================================================
# DAC Device Name Validation
# ============================================================================

# Allowed ALSA device name patterns:
# - "default"
# - "hw:N" or "hw:N,M" where N, M are card/device numbers (0-99)
# - "plughw:N" or "plughw:N,M"
# - "sysdefault:CARD=name" format
SAFE_ALSA_DEVICE_PATTERN = re.compile(
    r"^(default|"  # default device
    r"(plug)?hw:\d{1,2}(,\d{1,2})?|"  # hw:0, hw:0,0, plughw:1,0
    r"sysdefault(:CARD=[a-zA-Z0-9_]+)?"  # sysdefault:CARD=PCH
    r")$"
)
