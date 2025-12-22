"""Daemon control and monitoring functions."""

import json
import logging
import os
import shutil
import signal
import subprocess
import time
from typing import Optional

from ..constants import (
    DAEMON_BINARY,
    DAEMON_SERVICE_NAMES,
    PID_FILE_PATH,
    STATS_FILE_PATH,
)
from .config import load_config

logger = logging.getLogger(__name__)


def _systemctl_available() -> bool:
    """Return True if systemctl is available in PATH."""
    return shutil.which("systemctl") is not None


def _get_systemd_service_name() -> Optional[str]:
    """Return the first known systemd service name that is installed."""
    if not _systemctl_available():
        return None

    for name in DAEMON_SERVICE_NAMES:
        try:
            result = subprocess.run(
                ["systemctl", "show", "-p", "LoadState", "--value", name],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip() == "loaded":
                return name
        except subprocess.SubprocessError:
            continue
    return None


def _is_service_active(service: str) -> bool:
    """Check if the systemd service is active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", service],
            timeout=2,
        )
        return result.returncode == 0
    except subprocess.SubprocessError:
        return False


def _get_service_pid(service: str) -> Optional[int]:
    """Get MainPID from systemd."""
    try:
        result = subprocess.run(
            ["systemctl", "show", "-p", "MainPID", "--value", service],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            return None
        pid = int(result.stdout.strip() or "0")
        return pid or None
    except (ValueError, subprocess.SubprocessError):
        return None


def _pid_looks_like_daemon(pid: int) -> bool:
    """Verify PID belongs to gpu_upsampler (avoid PID 1 or unrelated processes)."""
    if pid < 1:
        return False
    comm_path = f"/proc/{pid}/comm"
    try:
        with open(comm_path) as f:
            comm = f.read().strip()
        # Name is truncated to 15 chars; daemon names start with gpu_upsampler
        return comm.startswith("gpu_upsampler")
    except OSError:
        return False


def check_daemon_running() -> bool:
    """Check if the daemon process is running."""
    service = _get_systemd_service_name()
    if service:
        return _is_service_active(service)

    pid = get_daemon_pid()
    if pid is None:
        # Fallback: try ZeroMQ ping (works across container namespaces)
        try:
            from .daemon_client import get_daemon_client

            with get_daemon_client(timeout_ms=500) as client:
                ok, _ = client.send_command("PING")
                return ok
        except Exception:
            return False
    try:
        # Check if process exists
        os.kill(pid, 0)
        return _pid_looks_like_daemon(pid)
    except (OSError, ProcessLookupError):
        return False


def get_daemon_pid() -> Optional[int]:
    """Get the daemon PID from pid file or systemd."""
    service = _get_systemd_service_name()
    if service:
        pid = _get_service_pid(service)
        if pid and _pid_looks_like_daemon(pid):
            return pid

    if not PID_FILE_PATH.exists():
        return None
    try:
        with open(PID_FILE_PATH) as f:
            pid = int(f.read().strip())
            if _pid_looks_like_daemon(pid):
                return pid
    except (ValueError, IOError):
        pass

    # Last-resort: try ZMQ ping and surface None PID if running (for containerized setups)
    try:
        from .daemon_client import get_daemon_client

        with get_daemon_client(timeout_ms=500) as client:
            ok, _ = client.send_command("PING")
            if ok:
                return None
    except Exception:
        pass
    return None


def start_daemon() -> tuple[bool, str]:
    """Start the daemon process.

    This function performs the startup sequence:
    1. Check if daemon is already running
    2. Start daemon process
    """
    logger.info("Starting daemon...")

    config = load_config()

    # Prefer systemd control when service is installed (Jetson)
    service = _get_systemd_service_name()
    if service:
        if _is_service_active(service):
            logger.warning("Daemon is already running via systemd")
            return False, "Daemon is already running (systemd)"

        try:
            result = subprocess.run(
                ["systemctl", "start", service],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info("Daemon started via systemd: %s", service)
                return True, f"Daemon started via systemd ({service})"
            error_msg = (
                result.stderr.strip() or result.stdout.strip() or "unknown error"
            )
            logger.error("Failed to start daemon via systemd: %s", error_msg)
            return False, f"systemctl start {service} failed: {error_msg}"
        except subprocess.SubprocessError as e:
            logger.error("Failed to start daemon via systemd: %s", e)
            return False, f"systemctl start failed: {e}"

    if check_daemon_running():
        logger.warning("Daemon is already running")
        return False, "Daemon is already running"

    if not DAEMON_BINARY.exists():
        logger.error("Daemon binary not found: %s", DAEMON_BINARY)
        return False, f"Daemon binary not found: {DAEMON_BINARY}"

    try:
        # Start daemon process
        logger.info(
            "Launching daemon binary: %s -d %s", DAEMON_BINARY, config.alsa_device
        )
        subprocess.Popen(
            [
                str(DAEMON_BINARY),
                "-d",
                config.alsa_device,
            ],
            start_new_session=True,
        )

        logger.info("Daemon started successfully")
        return True, "Daemon started"
    except subprocess.SubprocessError as e:
        logger.error("Failed to start daemon: %s", e)
        return False, f"Failed to start daemon: {e}"


def _force_stop_daemon() -> None:
    """Force stop daemon process (internal cleanup helper)."""
    pid = get_daemon_pid()
    if pid is None:
        return
    try:
        logger.debug("Sending SIGTERM to daemon (PID: %d)", pid)
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            time.sleep(0.1)
            if not check_daemon_running():
                logger.debug("Daemon stopped gracefully")
                return
        # Force kill if still running
        logger.warning("Daemon did not stop gracefully, sending SIGKILL")
        os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def stop_daemon() -> tuple[bool, str]:
    """Stop the daemon process."""
    logger.info("Stopping daemon...")

    service = _get_systemd_service_name()
    if service:
        if not _is_service_active(service):
            logger.warning("Daemon is not running (systemd)")
            return False, "Daemon is not running (systemd)"
        try:
            result = subprocess.run(
                ["systemctl", "stop", service],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                error_msg = (
                    result.stderr.strip() or result.stdout.strip() or "unknown error"
                )
                logger.error("Failed to stop daemon via systemd: %s", error_msg)
                return False, f"systemctl stop {service} failed: {error_msg}"
            # Confirm stop
            if _is_service_active(service):
                logger.error("Daemon still running after systemctl stop")
                return False, "Daemon did not stop via systemd"
            logger.info("Daemon stopped via systemd: %s", service)
            return True, "Daemon stopped (systemd)"
        except subprocess.SubprocessError as e:
            logger.error("Failed to stop daemon via systemd: %s", e)
            return False, f"systemctl stop failed: {e}"

    pid = get_daemon_pid()
    if pid is None:
        logger.warning("Daemon is not running (no PID file)")
        return False, "Daemon is not running (no PID file)"

    try:
        logger.info("Sending SIGTERM to daemon (PID: %d)", pid)
        os.kill(pid, signal.SIGTERM)
        # Wait a bit for graceful shutdown
        for _ in range(10):
            time.sleep(0.1)
            if not check_daemon_running():
                break

        if check_daemon_running():
            logger.warning("Daemon is still running after SIGTERM, forcing stop")
            _force_stop_daemon()
            if check_daemon_running():
                logger.error("Daemon is still running after SIGKILL fallback")
                return False, "Daemon did not stop"
            logger.info("Daemon stopped via SIGKILL fallback")
            return True, "Daemon stopped (forced)"

        logger.info("Daemon stopped successfully")
        return True, "Daemon stopped"
    except (OSError, ProcessLookupError):
        logger.error("Daemon process not found")
        return False, "Daemon process not found"


def get_configured_rates() -> tuple[int, int]:
    """Get configured input and output rates from runtime stats.

    Note: Input/output rates are auto-negotiated at runtime, not from config.
    This function reads from the daemon's stats file instead.
    """
    stats = load_stats()
    return stats.get("input_rate", 0), stats.get("output_rate", 0)


def load_stats() -> dict:
    """Load statistics from stats file."""
    default_stats = {
        "clip_rate": 0.0,
        "clip_count": 0,
        "total_samples": 0,
        "input_rate": 0,
        "output_rate": 0,
        "peaks": {
            "input": {"linear": 0.0, "dbfs": -200.0},
            "upsampler": {"linear": 0.0, "dbfs": -200.0},
            "post_mix": {"linear": 0.0, "dbfs": -200.0},
            "post_gain": {"linear": 0.0, "dbfs": -200.0},
        },
        # Debug extras (may be absent depending on daemon build/version)
        "xrun_count": 0,
        "buffer_underflows": 0,
        "buffer_overflows": 0,
        "buffer_capacity_frames": 0,
        "dropped_frames": 0,
        "rendered_silence_blocks": 0,
        "rendered_silence_frames": 0,
        "alsa_buffer_size_config": 0,
        "alsa_period_size_config": 0,
        "delimiter": {
            "enabled": False,
            "backend_available": False,
            "backend_valid": False,
            "mode": "unknown",
            "target_mode": "unknown",
            "fallback_reason": "unknown",
            "bypass_locked": False,
            "warmup": False,
            "queue_seconds": 0.0,
            "queue_samples": 0,
            "last_inference_ms": 0.0,
            "detail": None,
        },
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

        peaks = data.get("peaks", {})

        def _stage(name: str) -> dict:
            stage = peaks.get(name, {})
            return {
                "linear": stage.get("linear", 0.0),
                "dbfs": stage.get("dbfs", -200.0),
            }

        audio = data.get("audio", {}) if isinstance(data, dict) else {}
        if not isinstance(audio, dict):
            audio = {}

        buffer = data.get("buffer", {}) if isinstance(data, dict) else {}
        if not isinstance(buffer, dict):
            buffer = {}

        alsa = data.get("alsa", {}) if isinstance(data, dict) else {}
        if not isinstance(alsa, dict):
            alsa = {}

        delimiter = data.get("delimiter", {}) if isinstance(data, dict) else {}
        if not isinstance(delimiter, dict):
            delimiter = {}

        return {
            "clip_rate": clip_rate,
            "clip_count": clip,
            "total_samples": total,
            "input_rate": data.get("input_rate", 0),
            "output_rate": data.get("output_rate", 0),
            "peaks": {
                "input": _stage("input"),
                "upsampler": _stage("upsampler"),
                "post_mix": _stage("post_mix"),
                "post_gain": _stage("post_gain"),
            },
            "xrun_count": int(audio.get("xrun_count", 0) or 0),
            "buffer_underflows": int(audio.get("buffer_underflows", 0) or 0),
            "buffer_overflows": int(audio.get("buffer_overflows", 0) or 0),
            "buffer_capacity_frames": int(buffer.get("capacity_frames", 0) or 0),
            "dropped_frames": int(buffer.get("dropped_frames", 0) or 0),
            "rendered_silence_blocks": int(
                buffer.get("rendered_silence_blocks", 0) or 0
            ),
            "rendered_silence_frames": int(
                buffer.get("rendered_silence_frames", 0) or 0
            ),
            "alsa_buffer_size_config": int(alsa.get("buffer_size_config", 0) or 0),
            "alsa_period_size_config": int(alsa.get("period_size_config", 0) or 0),
            "delimiter": {
                "enabled": bool(delimiter.get("enabled", False)),
                "backend_available": bool(delimiter.get("backend_available", False)),
                "backend_valid": bool(delimiter.get("backend_valid", False)),
                "mode": delimiter.get("mode", "unknown") or "unknown",
                "target_mode": delimiter.get("target_mode", "unknown") or "unknown",
                "fallback_reason": delimiter.get("fallback_reason", "unknown")
                or "unknown",
                "bypass_locked": bool(delimiter.get("bypass_locked", False)),
                "warmup": bool(delimiter.get("warmup", False)),
                "queue_seconds": float(delimiter.get("queue_seconds", 0.0) or 0.0),
                "queue_samples": int(delimiter.get("queue_samples", 0) or 0),
                "last_inference_ms": float(
                    delimiter.get("last_inference_ms", 0.0) or 0.0
                ),
                "detail": delimiter.get("detail"),
            },
        }
    except (json.JSONDecodeError, IOError):
        return default_stats
