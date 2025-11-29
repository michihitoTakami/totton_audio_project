"""Input mode switching endpoints."""

import time

from fastapi import APIRouter, HTTPException

from ..models import InputModeSwitchRequest, InputModeSwitchResponse
from ..services import check_daemon_running, start_daemon, stop_daemon
from ..services.config import get_input_mode, save_input_mode

router = APIRouter(prefix="/api/input-mode", tags=["input-mode"])


@router.post("/switch", response_model=InputModeSwitchResponse)
async def switch_input_mode(request: InputModeSwitchRequest):
    """Toggle PipeWire ⇄ RTP input modes."""
    target_mode = request.mode
    current_mode = get_input_mode()

    if target_mode == current_mode:
        return InputModeSwitchResponse(
            success=True,
            current_mode=current_mode,
            restart_required=False,
            message=f"Input mode already set to {current_mode}",
        )

    if not save_input_mode(target_mode):
        raise HTTPException(
            status_code=500, detail="Failed to update input mode in config.json"
        )

    restart_required = False
    if check_daemon_running():
        stop_success, stop_msg = stop_daemon()
        if not stop_success:
            raise HTTPException(
                status_code=500, detail=f"Failed to stop daemon: {stop_msg}"
            )
        time.sleep(1.0)
        start_success, start_msg = start_daemon()
        if not start_success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to restart daemon after switching: {start_msg}",
            )
        time.sleep(0.5)
        restart_required = True

    return InputModeSwitchResponse(
        success=True,
        current_mode=target_mode,
        restart_required=restart_required,
        message=(
            "Daemon restarted to apply new input mode"
            if restart_required
            else "Input mode updated (daemon not running)"
        ),
    )

