"""Daemon control and monitoring functions."""

import json
import signal
import subprocess
from typing import Optional

from ..constants import DAEMON_BINARY, PID_FILE_PATH, STATS_FILE_PATH
from .config import load_config


def check_daemon_running() -> bool:
    """Check if the daemon process is running."""
    pid = get_daemon_pid()
    if pid is None:
        return False
    try:
        # Check if process exists
        import os

        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_daemon_pid() -> Optional[int]:
    """Get the daemon PID from pid file."""
    if not PID_FILE_PATH.exists():
        return None
    try:
        with open(PID_FILE_PATH) as f:
            return int(f.read().strip())
    except (ValueError, IOError):
        return None


def start_daemon() -> tuple[bool, str]:
    """Start the daemon process."""
    if check_daemon_running():
        return False, "Daemon is already running"

    if not DAEMON_BINARY.exists():
        return False, f"Daemon binary not found: {DAEMON_BINARY}"

    config = load_config()
    try:
        subprocess.Popen(
            [
                str(DAEMON_BINARY),
                "-d",
                config.alsa_device,
                "-r",
                str(config.input_rate),
            ],
            start_new_session=True,
        )
        return True, "Daemon started"
    except subprocess.SubprocessError as e:
        return False, f"Failed to start daemon: {e}"


def stop_daemon() -> tuple[bool, str]:
    """Stop the daemon process."""
    pid = get_daemon_pid()
    if pid is None:
        return False, "Daemon is not running (no PID file)"

    try:
        import os

        os.kill(pid, signal.SIGTERM)
        # Wait a bit for graceful shutdown
        import time

        for _ in range(10):
            time.sleep(0.1)
            if not check_daemon_running():
                break
        return True, "Daemon stopped"
    except (OSError, ProcessLookupError):
        return False, "Daemon process not found"


def check_pipewire_sink() -> bool:
    """Check if PipeWire sink is available."""
    try:
        result = subprocess.run(
            ["pw-cli", "list-objects"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return "gpu_upsampler" in result.stdout.lower()
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def get_configured_rates() -> tuple[int, int]:
    """Get configured input and output rates."""
    config = load_config()
    return config.input_rate, config.output_rate


def load_stats() -> dict:
    """Load statistics from stats file."""
    default_stats = {
        "clip_rate": 0.0,
        "clip_count": 0,
        "total_samples": 0,
        "input_rate": 0,
        "output_rate": 0,
    }
    if not STATS_FILE_PATH.exists():
        return default_stats
    try:
        with open(STATS_FILE_PATH) as f:
            data = json.load(f)

        # Compute clip_rate as ratio
        total = data.get("total_samples", 0)
        clip = data.get("clip_count", 0)
        clip_rate = (clip / total) if total > 0 else 0.0

        return {
            "clip_rate": clip_rate,
            "clip_count": clip,
            "total_samples": total,
            "input_rate": data.get("input_rate", 0),
            "output_rate": data.get("output_rate", 0),
        }
    except (json.JSONDecodeError, IOError):
        return default_stats
