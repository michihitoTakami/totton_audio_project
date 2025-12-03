"""Daemon control endpoints."""

import subprocess
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from ..constants import (
    DAEMON_BINARY,
    DAEMON_PHASE_LINEAR,
    HYBRID_PHASE_WARNING,
    PHASE_TYPE_HYBRID,
    PHASE_TYPE_MINIMUM,
    PID_FILE_PATH,
)
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
    load_partitioned_convolution_settings,
    save_partitioned_convolution_settings,
    save_phase_type,
    start_daemon,
    stop_daemon,
)
from ..services.daemon_client import DaemonError

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
    settings = load_config()
    input_mode = "rtp" if settings.rtp_enabled else "pipewire"

    return DaemonStatus(
        running=running,
        pid=pid,
        pid_file=str(PID_FILE_PATH),
        binary_path=str(DAEMON_BINARY),
        pipewire_connected=pipewire_connected,
        input_mode=input_mode,
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

    Returns the current filter phase type (minimum or hybrid).
    Hybrid combines minimum phase below 150 Hz with linear phase above 150 Hz
    (~6.7 ms alignment, 1 period of crossover frequency) to retain imaging while keeping bass tight.
    """
    with get_daemon_client() as client:
        response = client.send_command_v2("PHASE_TYPE_GET")
        if not response.success:
            # Raise DaemonError to trigger RFC 9457 error handler
            raise response.error

        # Parse phase type from response data
        data = response.data
        if isinstance(data, str):
            import json

            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise DaemonError(
                    error_code="IPC_PROTOCOL_ERROR",
                    message=f"Invalid JSON in phase type response: {data}",
                )

        if not isinstance(data, dict):
            raise DaemonError(
                error_code="IPC_PROTOCOL_ERROR",
                message=f"Expected dict for phase type, got {type(data).__name__}",
            )

        phase_type_raw = data.get("phase_type")
        if phase_type_raw not in (
            PHASE_TYPE_MINIMUM,
            PHASE_TYPE_HYBRID,
            DAEMON_PHASE_LINEAR,
        ):
            raise DaemonError(
                error_code="IPC_PROTOCOL_ERROR",
                message=f"Invalid phase type from daemon: {phase_type_raw}",
            )

        ui_phase_type = (
            PHASE_TYPE_HYBRID
            if phase_type_raw in (DAEMON_PHASE_LINEAR, PHASE_TYPE_HYBRID)
            else PHASE_TYPE_MINIMUM
        )
        latency_warning = (
            HYBRID_PHASE_WARNING if ui_phase_type == PHASE_TYPE_HYBRID else None
        )

        return PhaseTypeResponse(
            phase_type=ui_phase_type,
            latency_warning=latency_warning,
        )


@router.put("/phase-type", response_model=ApiResponse)
async def set_phase_type(request: PhaseTypeUpdateRequest):
    """
    Set phase type on daemon.

    All filter variants (min/linear × rate families) are preloaded by default, so changes take effect immediately without daemon restart.

    - minimum: Minimum phase filter (recommended, no pre-ringing)
    - hybrid: Minimum phase below 150 Hz / linear phase above 150 Hz (~6.7 ms alignment, 1 period of crossover frequency)
    """
    # Validation is handled by Pydantic Literal type (returns 422 for invalid values)
    with get_daemon_client() as client:
        daemon_phase = (
            DAEMON_PHASE_LINEAR
            if request.phase_type == PHASE_TYPE_HYBRID
            else PHASE_TYPE_MINIMUM
        )
        response = client.send_command_v2(f"PHASE_TYPE_SET:{daemon_phase}")
        if not response.success:
            # Raise DaemonError to trigger RFC 9457 error handler
            raise response.error

        save_phase_type(request.phase_type)
        partition_disabled = False
        if request.phase_type == PHASE_TYPE_HYBRID:
            current_partition = load_partitioned_convolution_settings()
            if current_partition.enabled:
                current_partition.enabled = False
                save_partitioned_convolution_settings(current_partition)
                partition_disabled = True

        response_data: dict[str, Any] = {"phase_type": request.phase_type}
        if partition_disabled:
            response_data["partition_disabled"] = True

        return ApiResponse(
            success=True,
            message=f"Phase type set to {request.phase_type}",
            data=response_data,
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
