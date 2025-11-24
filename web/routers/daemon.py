"""Daemon control endpoints."""

import subprocess
import time

from fastapi import APIRouter, HTTPException

from ..constants import DAEMON_BINARY, PID_FILE_PATH
from ..models import (
    ApiResponse,
    DaemonStatus,
    PhaseTypeResponse,
    PhaseTypeUpdateRequest,
    RewireRequest,
    ZmqPingResponse,
)
from ..services import (
    check_daemon_running,
    check_pipewire_sink,
    get_daemon_client,
    get_daemon_pid,
    load_config,
    start_daemon,
    stop_daemon,
)

router = APIRouter(prefix="/daemon", tags=["daemon"])


@router.post("/start", response_model=ApiResponse)
async def daemon_start():
    """Start the daemon."""
    success, message = start_daemon()
    if success:
        # Wait a bit for daemon to initialize
        time.sleep(0.5)
    return ApiResponse(success=success, message=message)


@router.post("/stop", response_model=ApiResponse)
async def daemon_stop():
    """Stop the daemon."""
    success, message = stop_daemon()
    return ApiResponse(success=success, message=message)


@router.post("/restart", response_model=ApiResponse)
async def daemon_restart():
    """Restart the daemon."""
    # Stop if running
    if check_daemon_running():
        stop_success, stop_msg = stop_daemon()
        if not stop_success:
            return ApiResponse(
                success=False, message=f"Failed to stop daemon: {stop_msg}"
            )
        # Wait for graceful shutdown
        time.sleep(1)

    # Start
    start_success, start_msg = start_daemon()
    if start_success:
        time.sleep(0.5)
        return ApiResponse(success=True, message="Daemon restarted")
    else:
        return ApiResponse(
            success=False, message=f"Failed to start daemon: {start_msg}"
        )


@router.get("/status", response_model=DaemonStatus)
async def daemon_status():
    """Get detailed daemon status."""
    running = check_daemon_running()
    pid = get_daemon_pid()
    pipewire_connected = check_pipewire_sink() if running else False

    return DaemonStatus(
        running=running,
        pid=pid,
        pid_file=str(PID_FILE_PATH),
        binary_path=str(DAEMON_BINARY),
        pipewire_connected=pipewire_connected,
    )


# ============================================================================
# ZeroMQ Communication
# ============================================================================


@router.get("/zmq/ping", response_model=ZmqPingResponse)
async def zmq_ping():
    """Ping the daemon via ZeroMQ."""
    with get_daemon_client(timeout_ms=1000) as client:
        success, response = client.send_command("PING")
        return ZmqPingResponse(
            success=success,
            response=response,
            daemon_running=check_daemon_running(),
        )


@router.post("/zmq/command/{cmd}", response_model=ApiResponse)
async def zmq_command(cmd: str):
    """Send arbitrary command to daemon via ZeroMQ."""
    allowed_commands = {"PING", "STATS", "RELOAD", "SOFT_RESET"}
    cmd_upper = cmd.upper()

    if cmd_upper not in allowed_commands:
        raise HTTPException(
            status_code=400,
            detail=f"Command not allowed. Allowed: {', '.join(allowed_commands)}",
        )

    with get_daemon_client() as client:
        success, response = client.send_command(cmd_upper)
        return ApiResponse(
            success=success,
            message=response if isinstance(response, str) else "Command executed",
            data={"response": response} if success else None,
        )


# ============================================================================
# Phase Type Control
# ============================================================================


@router.get("/phase-type", response_model=PhaseTypeResponse)
async def get_phase_type():
    """
    Get current phase type from daemon.

    Returns the current filter phase type (minimum or linear).
    Linear phase has high latency (~1 second) but no phase distortion.
    Minimum phase has no pre-ringing and low latency.
    """
    with get_daemon_client() as client:
        success, result = client.get_phase_type()
        if not success:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to get phase type: {result}",
            )

        phase_type = result.get("phase_type")
        if phase_type not in ("minimum", "linear"):
            raise HTTPException(
                status_code=503,
                detail=f"Invalid phase type from daemon: {phase_type}",
            )

        latency_warning = None
        if phase_type == "linear":
            latency_warning = (
                "Linear phase filter has high latency (~1 second). "
                "Use minimum phase for real-time applications."
            )

        return PhaseTypeResponse(
            phase_type=phase_type,
            latency_warning=latency_warning,
        )


@router.put("/phase-type", response_model=ApiResponse)
async def set_phase_type(request: PhaseTypeUpdateRequest):
    """
    Set phase type on daemon.

    Requires quad-phase mode to be enabled (4 filter variants preloaded).
    Changes take effect immediately without daemon restart.

    - minimum: Minimum phase filter (recommended, no pre-ringing)
    - linear: Linear phase filter (high latency ~1 second)
    """
    # Validation is handled by Pydantic Literal type (returns 422 for invalid values)
    with get_daemon_client() as client:
        success, message = client.set_phase_type(request.phase_type)
        if not success:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to set phase type: {message}",
            )

        return ApiResponse(
            success=True,
            message=f"Phase type set to {request.phase_type}",
            data={"phase_type": request.phase_type},
        )


# ============================================================================
# Restart / Rewire
# ============================================================================


@router.post("/restart-full", response_model=ApiResponse)
async def reload_daemon():
    """
    Restart daemon with updated configuration.

    This is called after settings changes that require a daemon restart
    (e.g., sample rate changes, EQ profile changes).
    """
    config = load_config()

    # First try soft reset via ZeroMQ (if daemon is running)
    if check_daemon_running():
        with get_daemon_client() as client:
            success, msg = client.reload_config()
            if success:
                return ApiResponse(
                    success=True,
                    message="Configuration reloaded via ZeroMQ",
                    data={"method": "zmq_reload"},
                )

    # Fall back to hard restart if ZeroMQ fails or daemon not running
    was_running = check_daemon_running()

    if was_running:
        stop_success, stop_msg = stop_daemon()
        if not stop_success:
            return ApiResponse(
                success=False, message=f"Failed to stop daemon: {stop_msg}"
            )
        time.sleep(1)

    start_success, start_msg = start_daemon()
    if start_success:
        time.sleep(0.5)
        return ApiResponse(
            success=True,
            message="Daemon restarted with new configuration",
            data={"method": "hard_restart", "config": config.model_dump()},
        )
    else:
        return ApiResponse(
            success=False, message=f"Failed to start daemon: {start_msg}"
        )


@router.post("/rewire", response_model=ApiResponse)
async def rewire_pipewire(request: RewireRequest):
    """Rewire PipeWire connections."""
    try:
        # Use pw-link to create connection
        result = subprocess.run(
            ["pw-link", request.source_node, request.target_node],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return ApiResponse(
                success=True,
                message=f"Connected {request.source_node} -> {request.target_node}",
            )
        else:
            return ApiResponse(
                success=False, message=f"pw-link failed: {result.stderr}"
            )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        return ApiResponse(success=False, message=f"Failed to rewire: {e}")
