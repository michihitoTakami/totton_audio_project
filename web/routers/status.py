"""Status and settings endpoints."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..models import ApiResponse, SettingsUpdate, Status
from ..services import (
    check_daemon_running,
    check_pipewire_sink,
    get_alsa_devices,
    load_config,
    load_stats,
    save_config,
)
from ..templates import get_embedded_html

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def root():
    """Serve the embedded HTML page."""
    return get_embedded_html()


@router.get("/status", response_model=Status)
async def get_status():
    """Get current system status."""
    settings = load_config()
    daemon_running = check_daemon_running()
    pipewire_connected = check_pipewire_sink() if daemon_running else False
    stats = load_stats()

    return Status(
        settings=settings,
        daemon_running=daemon_running,
        pipewire_connected=pipewire_connected,
        eq_active=settings.eq_profile is not None,
        clip_rate=stats["clip_rate"],
        clip_count=stats["clip_count"],
        total_samples=stats["total_samples"],
        input_rate=stats["input_rate"],
        output_rate=stats["output_rate"],
    )


@router.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """WebSocket endpoint for real-time stats updates."""
    await websocket.accept()
    try:
        while True:
            # Get current stats
            running = check_daemon_running()
            stats = load_stats()

            # Send to client
            await websocket.send_json(
                {
                    "daemon_running": running,
                    "clip_rate": stats["clip_rate"],
                    "clip_count": stats["clip_count"],
                    "total_samples": stats["total_samples"],
                    "input_rate": stats["input_rate"],
                    "output_rate": stats["output_rate"],
                }
            )

            # Wait before next update
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


@router.post("/settings", response_model=ApiResponse)
async def update_settings(update: SettingsUpdate):
    """Update application settings."""
    current = load_config()

    # Update only provided fields
    if update.alsa_device is not None:
        current.alsa_device = update.alsa_device
    if update.upsample_ratio is not None:
        current.upsample_ratio = update.upsample_ratio
    if update.eq_profile is not None:
        current.eq_profile = update.eq_profile
    if update.input_rate is not None:
        current.input_rate = update.input_rate
    if update.output_rate is not None:
        current.output_rate = update.output_rate

    if save_config(current):
        # Check if daemon needs restart
        restart_required = check_daemon_running()
        return ApiResponse(
            success=True,
            message="Settings updated",
            data=current.model_dump(),
            restart_required=restart_required,
        )
    else:
        return ApiResponse(success=False, message="Failed to save settings")


@router.get("/devices")
async def list_devices():
    """List available ALSA devices."""
    return {"devices": get_alsa_devices()}
