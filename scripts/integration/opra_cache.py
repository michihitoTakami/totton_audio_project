#!/usr/bin/env python3
"""
OPRA cache manager (versions, metadata, atomic swap, rollback).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATABASE_FILENAME = "database_v1.jsonl"
METADATA_FILENAME = "metadata.json"
STATE_FILENAME = "opra_sync_state.json"
CURRENT_SYMLINK = "current"
VERSIONS_DIRNAME = "versions"
LOCK_DIRNAME = "lock"
LOCK_FILENAME = "opra_update.lock"

_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_data_dir() -> Path:
    return Path(
        os.environ.get("GPU_OS_DATA_DIR")
        or os.environ.get("DATA_DIR")
        or "/var/lib/gpu_upsampler"
    )


def normalize_commit_sha(commit_sha: str | None) -> str:
    if not commit_sha:
        return "unknown"
    if commit_sha == "unknown":
        return commit_sha
    if not _SHA_PATTERN.match(commit_sha):
        raise ValueError(f"Invalid commit SHA: {commit_sha}")
    return commit_sha.lower()


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class OpraSyncState:
    status: str = "idle"
    job_id: str | None = None
    current_commit: str | None = None
    previous_commit: str | None = None
    last_updated_at: str | None = None
    last_error: str | None = None
    versions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpraSyncState":
        return cls(
            status=data.get("status", "idle"),
            job_id=data.get("job_id"),
            current_commit=data.get("current_commit"),
            previous_commit=data.get("previous_commit"),
            last_updated_at=data.get("last_updated_at"),
            last_error=data.get("last_error"),
            versions=list(data.get("versions", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "job_id": self.job_id,
            "current_commit": self.current_commit,
            "previous_commit": self.previous_commit,
            "last_updated_at": self.last_updated_at,
            "last_error": self.last_error,
            "versions": list(self.versions),
        }


@dataclass
class OpraCacheManager:
    opra_dir: Path = field(default_factory=lambda: _default_data_dir() / "opra")
    max_versions: int = 3

    @property
    def versions_dir(self) -> Path:
        return self.opra_dir / VERSIONS_DIRNAME

    @property
    def lock_dir(self) -> Path:
        return self.opra_dir / LOCK_DIRNAME

    @property
    def lock_path(self) -> Path:
        return self.lock_dir / LOCK_FILENAME

    @property
    def current_path(self) -> Path:
        return self.opra_dir / CURRENT_SYMLINK

    @property
    def state_path(self) -> Path:
        return self.opra_dir / STATE_FILENAME

    def ensure_layout(self) -> None:
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    def build_metadata(
        self,
        *,
        commit_sha: str,
        source: str,
        source_url: str,
        database_path: Path,
        downloaded_at: str | None = None,
        sha256: str | None = None,
        stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_sha = normalize_commit_sha(commit_sha)
        return {
            "commit_sha": safe_sha,
            "source": source,
            "source_url": source_url,
            "downloaded_at": downloaded_at or _now_iso(),
            "sha256": sha256 or compute_sha256(database_path),
            "size_bytes": database_path.stat().st_size,
            "stats": stats or {},
        }

    def install_version(
        self,
        commit_sha: str,
        database_path: Path,
        metadata: dict[str, Any],
    ) -> Path:
        self.ensure_layout()
        safe_sha = normalize_commit_sha(commit_sha)
        version_dir = self.versions_dir / safe_sha
        if version_dir.exists():
            return version_dir

        temp_dir = self.versions_dir / f".{safe_sha}.tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=False)

        shutil.copy2(database_path, temp_dir / DATABASE_FILENAME)
        metadata_payload = dict(metadata)
        metadata_payload["commit_sha"] = safe_sha
        with (temp_dir / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(metadata_payload, handle, ensure_ascii=True, indent=2)

        os.replace(temp_dir, version_dir)
        return version_dir

    def get_current_commit(self) -> str | None:
        if not self.current_path.exists():
            return None
        if self.current_path.is_symlink():
            target = self.current_path.resolve()
            return target.name
        return None

    def activate_version(self, commit_sha: str) -> None:
        safe_sha = normalize_commit_sha(commit_sha)
        target_dir = self.versions_dir / safe_sha
        if not target_dir.exists():
            raise FileNotFoundError(f"OPRA version not found: {safe_sha}")

        if self.current_path.exists() and not self.current_path.is_symlink():
            raise RuntimeError("OPRA current path is not a symlink")

        temp_link = self.opra_dir / f"{CURRENT_SYMLINK}.tmp"
        if temp_link.exists() or temp_link.is_symlink():
            temp_link.unlink()
        temp_link.symlink_to(target_dir)
        os.replace(temp_link, self.current_path)

        self._record_activation(safe_sha)
        self.prune_versions()

    def rollback(self) -> str:
        state = self.load_state()
        if not state.previous_commit:
            raise RuntimeError("No previous OPRA version to rollback to")
        self.activate_version(state.previous_commit)
        return state.previous_commit

    def load_state(self) -> OpraSyncState:
        if not self.state_path.exists():
            return OpraSyncState()
        with self.state_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return OpraSyncState.from_dict(data)

    def save_state(self, state: OpraSyncState) -> None:
        self.opra_dir.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, ensure_ascii=True, indent=2)

    def set_status(
        self, status: str, error: str | None = None, job_id: str | None = None
    ) -> None:
        state = self.load_state()
        state.status = status
        state.job_id = job_id
        state.last_error = error
        state.last_updated_at = _now_iso()
        state.versions = self._list_version_names()
        self.save_state(state)

    def prune_versions(self) -> None:
        state = self.load_state()
        protected = {state.current_commit, state.previous_commit}
        protected.discard(None)

        versions = self._list_versions_by_mtime()
        keep_names = set(protected)
        for version in versions:
            if version.name in keep_names:
                continue
            if len(keep_names) < self.max_versions:
                keep_names.add(version.name)
                continue
            shutil.rmtree(version)

        state.versions = [
            version.name for version in versions if version.name in keep_names
        ]
        self.save_state(state)

    def _record_activation(self, safe_sha: str) -> None:
        state = self.load_state()
        current_commit = state.current_commit
        if current_commit and current_commit != safe_sha:
            state.previous_commit = current_commit
        state.current_commit = safe_sha
        state.status = "ready"
        state.job_id = None
        state.last_error = None
        state.last_updated_at = _now_iso()
        state.versions = self._list_version_names()
        self.save_state(state)

    def _list_versions_by_mtime(self) -> list[Path]:
        if not self.versions_dir.exists():
            return []
        return sorted(
            [path for path in self.versions_dir.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def _list_version_names(self) -> list[str]:
        return [path.name for path in self._list_versions_by_mtime()]
