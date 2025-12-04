"""System information and logs API endpoints."""

from fastapi import APIRouter, HTTPException, Query

from ..models import SystemLogsResponse
from ..services.system import read_system_logs

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/logs", response_model=SystemLogsResponse)
async def get_system_logs(
    lines: int = Query(
        default=100, ge=1, le=1000, description="Number of lines to return"
    ),
    offset: int = Query(default=0, ge=0, description="Number of lines to skip"),
    level: str | None = Query(
        default=None, description="Filter by log level (debug, info, warning, error)"
    ),
) -> SystemLogsResponse:
    """
    Retrieve system logs from the daemon log file.

    Args:
        lines: Number of lines to return (1-1000)
        offset: Number of lines to skip from the beginning
        level: Optional log level filter

    Returns:
        SystemLogsResponse containing logs and metadata

    Raises:
        HTTPException: If log file is too large or cannot be read
    """
    try:
        result = read_system_logs(lines=lines, offset=offset, level_filter=level)
        return SystemLogsResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Log file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")
