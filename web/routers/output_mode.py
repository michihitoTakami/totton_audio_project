"""Output mode configuration endpoints (Issue #515)."""

from fastapi import APIRouter, HTTPException

from ..error_codes import ErrorCode
from ..models import ApiResponse, OutputModeResponse, OutputModeUpdateRequest
from ..services import (
    get_daemon_client,
    is_safe_device_name,
    load_output_mode,
    save_output_mode,
)

router = APIRouter(prefix="/api/output", tags=["output"])


def _fetch_runtime_output_mode() -> dict | None:
    """Query daemon over ZeroMQ for current output mode."""
    try:
        with get_daemon_client(timeout_ms=1500) as client:
            response = client.send_json_command_v2("OUTPUT_MODE_GET")
            if response.success and isinstance(response.data, dict):
                return response.data
    except Exception:
        # Daemon not reachable or JSON parse error - fall back to config
        return None
    return None


def _apply_runtime_output_mode(payload: dict) -> tuple[dict | None, bool]:
    """Send OUTPUT_MODE_SET to the daemon."""
    restart_required = False
    runtime_state = None
    try:
        with get_daemon_client(timeout_ms=2000) as client:
            response = client.send_json_command_v2("OUTPUT_MODE_SET", payload)
            if response.success and isinstance(response.data, dict):
                runtime_state = response.data
            elif response.error:
                # Gracefully degrade when daemon is offline
                if response.error.error_code in {
                    ErrorCode.IPC_DAEMON_NOT_RUNNING.value,
                    ErrorCode.IPC_CONNECTION_FAILED.value,
                    ErrorCode.IPC_TIMEOUT.value,
                }:
                    restart_required = True
                else:
                    raise HTTPException(
                        status_code=response.error.http_status,
                        detail={
                            "error_code": response.error.error_code,
                            "message": response.error.message,
                        },
                    )
            else:
                restart_required = True
    except HTTPException:
        raise
    except Exception:
        restart_required = True
    return runtime_state, restart_required


@router.get("/mode", response_model=OutputModeResponse, summary="Get current output mode")
async def get_output_mode():
    """Return current output mode and available modes."""
    runtime_state = _fetch_runtime_output_mode()
    if runtime_state:
        return runtime_state
    return load_output_mode()


@router.post("/mode", response_model=ApiResponse, summary="Update output mode and options")
async def update_output_mode(request: OutputModeUpdateRequest):
    """Update output mode and USB preferred device."""
    current_state = load_output_mode()
    usb_options = request.options.usb
    preferred_device = (usb_options.preferred_device or "").strip()
    fallback_device = current_state["options"]["usb"]["preferred_device"]
    device_to_use = preferred_device or fallback_device or "default"

    if device_to_use and not is_safe_device_name(device_to_use):
        raise HTTPException(
            status_code=400,
            detail="Invalid ALSA device name. Use hw:, plughw:, sysdefault:, or 'default'.",
        )

    if not save_output_mode(request.mode, device_to_use):
        raise HTTPException(status_code=500, detail="Failed to persist output mode")

    runtime_state, restart_required = _apply_runtime_output_mode(
        {
            "mode": request.mode,
            "options": {
                "usb": {
                    "preferred_device": device_to_use,
                }
            },
        }
    )

    response_state = runtime_state or load_output_mode()

    return ApiResponse(
        success=True,
        message="Output mode updated",
        data={"output_mode": response_state},
        restart_required=restart_required,
    )

