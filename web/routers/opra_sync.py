"""Admin OPRA sync endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..models import (
    OpraSyncAvailableResponse,
    OpraSyncJobResponse,
    OpraSyncStatusResponse,
    OpraSyncUpdateRequest,
)
from ..services.admin_auth import require_admin_basic
from ..services.opra_sync import (
    get_opra_cache_manager,
    load_current_metadata,
    resolve_available_source,
    run_rollback_job,
    run_update_job,
    start_rollback_job,
    start_update_job,
)

router = APIRouter(
    prefix="/api/opra/sync",
    tags=["opra"],
    dependencies=[Depends(require_admin_basic)],
)


@router.get("/status", response_model=OpraSyncStatusResponse)
async def opra_sync_status():
    """Return current OPRA sync status."""
    manager = get_opra_cache_manager()
    state = manager.load_state()
    current_commit = manager.get_current_commit() or state.current_commit
    metadata = load_current_metadata(manager)
    return OpraSyncStatusResponse(
        status=state.status,
        job_id=state.job_id,
        current_commit=current_commit,
        previous_commit=state.previous_commit,
        last_updated_at=state.last_updated_at,
        last_error=state.last_error,
        versions=state.versions,
        current_metadata=metadata,
    )


@router.get("/available", response_model=OpraSyncAvailableResponse)
async def opra_sync_available(source: str = "github_raw"):
    """Check available OPRA updates for the given source."""
    normalized = source.strip().lower()
    if normalized not in {"github_raw", "cloudflare"}:
        raise HTTPException(status_code=400, detail="Unsupported OPRA source")
    latest, source_url = resolve_available_source(normalized)
    return OpraSyncAvailableResponse(
        source=normalized,
        latest=latest,
        source_url=source_url,
    )


@router.post("/update", response_model=OpraSyncJobResponse, status_code=202)
async def opra_sync_update(
    request: OpraSyncUpdateRequest, background_tasks: BackgroundTasks
):
    """Kick off OPRA sync update in background."""
    try:
        job_id = start_update_job(target=request.target, source=request.source)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    background_tasks.add_task(
        run_update_job,
        job_id=job_id,
        target=request.target,
        source=request.source,
    )
    return OpraSyncJobResponse(job_id=job_id, status="running")


@router.post("/rollback", response_model=OpraSyncJobResponse, status_code=202)
async def opra_sync_rollback(background_tasks: BackgroundTasks):
    """Rollback to the previous OPRA version."""
    try:
        job_id = start_rollback_job()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    background_tasks.add_task(run_rollback_job, job_id=job_id)
    return OpraSyncJobResponse(job_id=job_id, status="running")
