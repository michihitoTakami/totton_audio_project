"""Crossfeed (HRTF) control endpoints."""

from fastapi import APIRouter, HTTPException

from ..models import (
    CrossfeedDisableResponse,
    CrossfeedEnableResponse,
    CrossfeedSizeResponse,
    CrossfeedStatus,
)
from ..services.daemon_client import get_daemon_client

router = APIRouter(prefix="/crossfeed", tags=["crossfeed"])


# Available head sizes (matching C++ HeadSize enum + xs)
VALID_HEAD_SIZES = ["xs", "s", "m", "l", "xl"]


@router.get("/status", response_model=CrossfeedStatus)
async def get_crossfeed_status():
    """
    Get current crossfeed status.

    Returns:
        CrossfeedStatus with enabled state, current head size, and available sizes.
    """
    with get_daemon_client() as client:
        result = client.crossfeed_get_status()

        if not result.success:
            raise result.error

        # Parse response data
        data = result.data
        if isinstance(data, dict):
            enabled = data.get("enabled", False)
            head_size = data.get("head_size") or data.get("headSize")
            # Normalize head_size to lowercase if present
            if head_size:
                head_size = head_size.lower()
        else:
            enabled = False
            head_size = None

        return CrossfeedStatus(
            enabled=enabled,
            headSize=head_size,
            availableSizes=VALID_HEAD_SIZES,
        )


@router.post("/enable", response_model=CrossfeedEnableResponse)
async def enable_crossfeed():
    """
    Enable crossfeed (HRTF) processing.

    Returns:
        Success response with confirmation message.
    """
    with get_daemon_client() as client:
        result = client.crossfeed_enable()

        if not result.success:
            raise result.error

        return CrossfeedEnableResponse(success=True, message="Crossfeed enabled")


@router.post("/disable", response_model=CrossfeedDisableResponse)
async def disable_crossfeed():
    """
    Disable crossfeed processing.

    Returns:
        Success response with confirmation message.
    """
    with get_daemon_client() as client:
        result = client.crossfeed_disable()

        if not result.success:
            raise result.error

        return CrossfeedDisableResponse(success=True, message="Crossfeed disabled")


@router.post("/size/{size}", response_model=CrossfeedSizeResponse)
async def set_crossfeed_size(size: str):
    """
    Change crossfeed head size.

    Args:
        size: Head size to set ('xs', 's', 'm', 'l', or 'xl')

    Returns:
        Success response with new head size.
    """
    # Validate head size
    size_lower = size.lower()
    if size_lower not in VALID_HEAD_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid head size: {size}. Must be one of {VALID_HEAD_SIZES}",
        )

    with get_daemon_client() as client:
        result = client.crossfeed_set_size(size_lower)

        if not result.success:
            raise result.error

        # Parse response data
        data = result.data
        if isinstance(data, dict):
            new_head_size = data.get("head_size", size_lower)
        else:
            new_head_size = size_lower

        return CrossfeedSizeResponse(success=True, headSize=new_head_size)
