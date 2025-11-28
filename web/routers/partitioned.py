"""Low-latency partitioned convolution configuration endpoints."""

from fastapi import APIRouter, HTTPException

from ..models import ApiResponse, PartitionedConvolutionSettings
from ..services import (
    check_daemon_running,
    load_partitioned_convolution_settings,
    save_partitioned_convolution_settings,
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

    restart_required = check_daemon_running()
    return ApiResponse(
        success=True,
        message="Partitioned convolution settings updated",
        data={"partitioned_convolution": settings.model_dump()},
        restart_required=restart_required,
    )

