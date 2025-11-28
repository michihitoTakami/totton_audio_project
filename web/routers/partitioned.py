"""Low-latency partitioned convolution configuration endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException

from ..models import ApiResponse, PartitionedConvolutionSettings
from ..services import (
    check_daemon_running,
    get_daemon_client,
    load_partitioned_convolution_settings,
    save_partitioned_convolution_settings,
    save_phase_type,
)

router = APIRouter(
    prefix="/partitioned-convolution",
    tags=["partitioned"],
    responses={404: {"description": "Not found"}},
)


@router.get(
    "",
    response_model=PartitionedConvolutionSettings,
    summary="Get partitioned convolution settings",
    description="Return the current low-latency partitioned convolution configuration "
    "stored in config.json.",
)
async def get_partitioned_convolution_settings():
    """Return current partitioned convolution settings."""
    return load_partitioned_convolution_settings()


@router.put(
    "",
    response_model=ApiResponse,
    summary="Update partitioned convolution settings",
    description="Validate and persist low-latency partitioned convolution settings. "
    "Returns whether the daemon restart is required for changes to take effect.",
)
async def update_partitioned_convolution_settings(
    settings: PartitionedConvolutionSettings,
):
    """Persist partitioned convolution settings and report restart requirement."""
    if not save_partitioned_convolution_settings(settings):
        raise HTTPException(
            status_code=500,
            detail="Failed to save partitioned convolution settings",
        )

    daemon_running = check_daemon_running()
    phase_adjusted = False
    runtime_phase_updated = False

    if settings.enabled:
        phase_adjusted = save_phase_type("minimum")
        if daemon_running:
            try:
                with get_daemon_client() as client:
                    phase_response = client.send_command_v2("PHASE_TYPE_SET:minimum")
                    runtime_phase_updated = phase_response.success
                    if not phase_response.success and phase_response.error:
                        # Only log client-side; API still succeeds because config was saved
                        message = str(phase_response.error)
                        print(
                            f"[Partition] Warning: Failed to force minimum phase: {message}"
                        )
            except Exception as exc:  # pragma: no cover - defensive logging
                print(
                    f"[Partition] Warning: Unable to contact daemon for phase sync: {exc}"
                )

    response_data: dict[str, Any] = {"partitioned_convolution": settings.model_dump()}
    if phase_adjusted:
        response_data["phase_type"] = "minimum"
        response_data["phase_adjusted"] = True
        response_data["phase_runtime_updated"] = runtime_phase_updated

    return ApiResponse(
        success=True,
        message="Partitioned convolution settings updated",
        data=response_data,
        restart_required=daemon_running,
    )
