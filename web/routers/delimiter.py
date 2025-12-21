"""De-limiter control endpoints."""

from typing import Any

from fastapi import APIRouter

from ..models import DelimiterActionResponse, DelimiterStatus
from ..services.daemon_client import get_daemon_client

router = APIRouter(prefix="/delimiter", tags=["delimiter"])


def _to_float(value: Any, default: float = 0.0) -> float:
    """Safely cast to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    """Safely cast to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_status(data: Any) -> DelimiterStatus:
    """Normalize daemon response into DelimiterStatus."""
    if not isinstance(data, dict):
        return DelimiterStatus()

    target_mode = data.get("target_mode", data.get("targetMode", "unknown"))
    detail_raw = data.get("detail")

    return DelimiterStatus(
        enabled=bool(data.get("enabled", False)),
        backend_available=bool(data.get("backend_available", False)),
        backend_valid=bool(data.get("backend_valid", False)),
        mode=str(data.get("mode", "unknown") or "unknown"),
        target_mode=str(target_mode or "unknown"),
        fallback_reason=str(data.get("fallback_reason", "unknown") or "unknown"),
        bypass_locked=bool(data.get("bypass_locked", False)),
        warmup=bool(data.get("warmup", False)),
        queue_seconds=_to_float(data.get("queue_seconds", 0.0)),
        queue_samples=_to_int(data.get("queue_samples", 0)),
        last_inference_ms=_to_float(data.get("last_inference_ms", 0.0)),
        detail=str(detail_raw) if detail_raw is not None else None,
    )


def _fetch_status(client) -> DelimiterStatus:
    """Fetch status or propagate daemon errors."""
    result = client.delimiter_status()
    if not result.success:
        raise result.error
    return _parse_status(result.data)


@router.get("/status", response_model=DelimiterStatus)
async def get_delimiter_status() -> DelimiterStatus:
    """Get current De-limiter status."""
    with get_daemon_client() as client:
        result = client.delimiter_status()
        if not result.success:
            raise result.error
        return _parse_status(result.data)


@router.post("/enable", response_model=DelimiterActionResponse)
async def enable_delimiter() -> DelimiterActionResponse:
    """Enable De-limiter processing."""
    with get_daemon_client() as client:
        result = client.delimiter_enable()
        if not result.success:
            raise result.error

        status = _parse_status(result.data)
        if not status.backend_available or not status.backend_valid:
            status = _fetch_status(client)

        return DelimiterActionResponse(
            success=True, message="De-limiter enabled", status=status
        )


@router.post("/disable", response_model=DelimiterActionResponse)
async def disable_delimiter() -> DelimiterActionResponse:
    """Disable De-limiter processing."""
    with get_daemon_client() as client:
        result = client.delimiter_disable()
        if not result.success:
            raise result.error

        status = _parse_status(result.data)
        if not status.backend_available or not status.backend_valid:
            status = _fetch_status(client)

        return DelimiterActionResponse(
            success=True, message="De-limiter disabled", status=status
        )
