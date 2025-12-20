"""OPRA sync orchestration for the admin API."""

from __future__ import annotations

import json
import tempfile
import threading
import uuid
from pathlib import Path

from scripts.integration.opra_cache import (
    METADATA_FILENAME,
    OpraCacheManager,
)
from scripts.integration.opra_downloader import (
    OPRA_CLOUDFLARE_URL,
    OPRA_RAW_BASE,
    OpraDownloadError,
    download_opra_database,
    resolve_latest_commit_sha,
)

_job_lock = threading.Lock()


def get_opra_cache_manager() -> OpraCacheManager:
    """Return an OpraCacheManager instance."""
    return OpraCacheManager()


def load_current_metadata(manager: OpraCacheManager) -> dict | None:
    """Load metadata.json for the active OPRA version."""
    current = manager.get_current_commit()
    if not current:
        return None
    metadata_path = manager.versions_dir / current / METADATA_FILENAME
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_available_source(source: str) -> tuple[str, str]:
    """Resolve the latest commit and URL for the given source."""
    normalized = source.strip().lower()
    if normalized == "github_raw":
        latest_sha = resolve_latest_commit_sha()
        source_url = f"{OPRA_RAW_BASE}/{latest_sha}/dist/database_v1.jsonl"
        return latest_sha, source_url
    return "unknown", OPRA_CLOUDFLARE_URL


def start_update_job(*, target: str, source: str) -> str:
    """Mark sync as running and return a job id."""
    manager = get_opra_cache_manager()
    with _job_lock:
        state = manager.load_state()
        if state.status == "running":
            raise RuntimeError("OPRA sync is already running")
        job_id = uuid.uuid4().hex
        manager.set_status("running", job_id=job_id)
        return job_id


def run_update_job(*, job_id: str, target: str, source: str) -> None:
    """Perform OPRA sync work (download -> validate -> activate)."""
    manager = get_opra_cache_manager()
    try:
        with tempfile.TemporaryDirectory(prefix="opra_sync_") as temp_dir:
            result = download_opra_database(
                target=target,
                source=source,
                temp_dir=Path(temp_dir),
            )
            metadata = manager.build_metadata(
                commit_sha=result.commit_sha,
                source=result.source,
                source_url=result.source_url,
                database_path=result.database_path,
                sha256=result.sha256,
                stats=result.stats,
            )
            manager.install_version(
                result.commit_sha,
                result.database_path,
                metadata,
            )
            manager.activate_version(result.commit_sha)
    except (OpraDownloadError, OSError, RuntimeError, ValueError) as exc:
        manager.set_status("failed", error=str(exc), job_id=None)
        return
    manager.set_status("ready", job_id=None)


def start_rollback_job() -> str:
    """Mark rollback as running and return a job id."""
    manager = get_opra_cache_manager()
    with _job_lock:
        state = manager.load_state()
        if state.status == "running":
            raise RuntimeError("OPRA sync is already running")
        if not state.previous_commit:
            raise RuntimeError("No previous OPRA version to rollback to")
        job_id = uuid.uuid4().hex
        manager.set_status("running", job_id=job_id)
        return job_id


def run_rollback_job(*, job_id: str) -> None:
    """Activate the previous OPRA version."""
    manager = get_opra_cache_manager()
    try:
        manager.rollback()
    except (OSError, RuntimeError, ValueError) as exc:
        manager.set_status("failed", error=str(exc), job_id=None)
        return
    manager.set_status("ready", job_id=None)
