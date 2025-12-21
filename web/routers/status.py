"""Status and settings endpoints."""

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..constants import EQ_PROFILES_DIR
from ..models import ApiResponse, DevicesResponse, SettingsUpdate, Status
from ..services import (
    check_daemon_running,
    get_alsa_devices,
    is_safe_profile_name,
    load_config,
    load_stats,
    save_config,
)

router = APIRouter(tags=["status"])


@router.get("/status", response_model=Status)
async def get_status():
    """Get current system status."""
    settings = load_config()
    daemon_running = check_daemon_running()
    stats = load_stats()

    return Status(
        settings=settings,
        daemon_running=daemon_running,
        eq_active=bool(settings.eq_enabled and settings.eq_profile_path),
        clip_rate=stats["clip_rate"],
        clip_count=stats["clip_count"],
        total_samples=stats["total_samples"],
        input_rate=stats["input_rate"],
        output_rate=stats["output_rate"],
        peaks=stats["peaks"],
        xrun_count=stats.get("xrun_count", 0),
        buffer_underflows=stats.get("buffer_underflows", 0),
        buffer_overflows=stats.get("buffer_overflows", 0),
        buffer_capacity_frames=stats.get("buffer_capacity_frames", 0),
        dropped_frames=stats.get("dropped_frames", 0),
        rendered_silence_blocks=stats.get("rendered_silence_blocks", 0),
        rendered_silence_frames=stats.get("rendered_silence_frames", 0),
        alsa_buffer_size_config=stats.get("alsa_buffer_size_config", 0),
        alsa_period_size_config=stats.get("alsa_period_size_config", 0),
        delimiter=stats.get("delimiter"),
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
                    "peaks": stats["peaks"],
                    "xrun_count": stats.get("xrun_count", 0),
                    "buffer_underflows": stats.get("buffer_underflows", 0),
                    "buffer_overflows": stats.get("buffer_overflows", 0),
                    "buffer_capacity_frames": stats.get("buffer_capacity_frames", 0),
                    "dropped_frames": stats.get("dropped_frames", 0),
                    "alsa_buffer_size_config": stats.get("alsa_buffer_size_config", 0),
                    "alsa_period_size_config": stats.get("alsa_period_size_config", 0),
                }
            )

            # Wait before next update
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


@router.post("/settings", response_model=ApiResponse)
async def update_settings(update: SettingsUpdate):
    """Update application settings."""
    # Validate eq_profile if provided (prevent path traversal)
    if update.eq_profile is not None and not is_safe_profile_name(update.eq_profile):
        raise HTTPException(
            status_code=400,
            detail="Invalid EQ profile name. Use only letters, numbers, underscores, hyphens, and dots.",
        )

    current = load_config()

    # Update only provided fields
    if update.alsa_device is not None:
        current.alsa_device = update.alsa_device
    if update.upsample_ratio is not None:
        current.upsample_ratio = update.upsample_ratio
    if update.eq_profile is not None:
        current.eq_profile = update.eq_profile
    if update.eq_profile_path is not None:
        current.eq_profile_path = update.eq_profile_path
    # eq_enabled is applied after we derive path so that enabling without a path does not stick
    eq_enabled_requested = update.eq_enabled

    # Keep EQ fields consistent with the daemon expectations
    if current.eq_profile_path is None and current.eq_profile:
        current.eq_profile_path = str(EQ_PROFILES_DIR / f"{current.eq_profile}.txt")

    # Apply eq_enabled after path derivation so that enable without path becomes False
    if eq_enabled_requested is None:
        current.eq_enabled = bool(current.eq_profile_path) and current.eq_enabled
    else:
        current.eq_enabled = bool(eq_enabled_requested and current.eq_profile_path)

    if not current.eq_enabled:
        current.eq_profile_path = None

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


@router.get("/devices", response_model=DevicesResponse)
async def list_devices():
    """List available ALSA devices."""
    return DevicesResponse(devices=get_alsa_devices())
