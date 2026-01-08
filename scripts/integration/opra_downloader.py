#!/usr/bin/env python3
"""
OPRA downloader (GitHub Raw / Cloudflare) with basic validation.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib import error, request

from scripts.integration.opra_cache import compute_sha256, normalize_commit_sha

GITHUB_API_BASE = "https://api.github.com"
OPRA_REPO = "opra-project/OPRA"
OPRA_RAW_BASE = "https://raw.githubusercontent.com/opra-project/OPRA"
OPRA_CLOUDFLARE_URL = "http://opra.roonlabs.net/database_v1.jsonl"

DEFAULT_TIMEOUT_MS = int(os.getenv("OPRA_DOWNLOAD_TIMEOUT_MS", "10000"))
DEFAULT_SAMPLE_LINES = int(os.getenv("OPRA_VALIDATE_SAMPLE_LINES", "200"))


class OpraDownloadError(Exception):
    """Base class for OPRA downloader errors."""


class OpraNetworkError(OpraDownloadError):
    """Raised when network is unreachable or times out."""


class OpraResponseError(OpraDownloadError):
    """Raised when OPRA API returns invalid or unexpected data."""


FetchFunc = Callable[[str, float], bytes]


@dataclass
class OpraDownloadResult:
    commit_sha: str
    source: str
    source_url: str
    database_path: Path
    sha256: str
    size_bytes: int
    stats: dict[str, int]


def _timeout_sec(timeout_ms: int | None) -> float:
    resolved = timeout_ms or DEFAULT_TIMEOUT_MS
    return max(0.1, resolved / 1000.0)


def _read_url(url: str, timeout_sec: float) -> bytes:
    req = request.Request(
        url,
        headers={
            "Accept": "*/*",
            "User-Agent": "Totton Audio Project-opra-sync",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            return response.read()
    except error.HTTPError as exc:
        message = exc.reason or "HTTP error"
        raise OpraResponseError(f"HTTP {exc.code} for {url}: {message}") from exc
    except error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise OpraNetworkError(f"Network error for {url}: {reason}") from exc


def resolve_latest_commit_sha(
    *, fetcher: FetchFunc = _read_url, timeout_ms: int | None = None
) -> str:
    url = f"{GITHUB_API_BASE}/repos/{OPRA_REPO}/commits/main"
    body = fetcher(url, _timeout_sec(timeout_ms))
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise OpraResponseError("Invalid JSON from GitHub API") from exc
    commit_sha = payload.get("sha")
    if not commit_sha or not isinstance(commit_sha, str):
        raise OpraResponseError("GitHub API response missing sha")
    return normalize_commit_sha(commit_sha)


def validate_database(
    database_path: Path, *, sample_lines: int | None = None
) -> dict[str, int]:
    resolved_lines = sample_lines or DEFAULT_SAMPLE_LINES
    if resolved_lines <= 0:
        raise OpraDownloadError("Sample line count must be positive")

    size_bytes = database_path.stat().st_size
    if size_bytes <= 0:
        raise OpraDownloadError("Downloaded OPRA database is empty")

    counts = {"vendor": 0, "product": 0, "eq": 0}
    with database_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if index > resolved_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OpraDownloadError(
                    f"Invalid JSONL at line {index}: {exc}"
                ) from exc
            entry_type = entry.get("type")
            if not entry_type:
                raise OpraDownloadError(f"Missing type field at line {index}")
            if entry_type in counts:
                counts[entry_type] += 1

    if sum(counts.values()) == 0:
        raise OpraDownloadError("OPRA database validation failed (no known types)")

    return counts


def _write_downloaded_file(payload: bytes, *, temp_dir: Path | None = None) -> Path:
    temp_root = temp_dir or Path(tempfile.mkdtemp(prefix="opra_download_"))
    temp_root.mkdir(parents=True, exist_ok=True)
    database_path = temp_root / "database_v1.jsonl"
    database_path.write_bytes(payload)
    return database_path


def download_opra_database(
    *,
    target: str,
    source: str,
    fetcher: FetchFunc = _read_url,
    timeout_ms: int | None = None,
    temp_dir: Path | None = None,
    sample_lines: int | None = None,
) -> OpraDownloadResult:
    source_key = source.strip().lower()
    if source_key not in {"github_raw", "cloudflare"}:
        raise OpraDownloadError(f"Unsupported OPRA source: {source}")

    commit_sha = "unknown"
    source_url = OPRA_CLOUDFLARE_URL
    if source_key == "github_raw":
        if target == "latest":
            commit_sha = resolve_latest_commit_sha(
                fetcher=fetcher, timeout_ms=timeout_ms
            )
        else:
            commit_sha = normalize_commit_sha(target)
        source_url = f"{OPRA_RAW_BASE}/{commit_sha}/dist/database_v1.jsonl"
    else:
        commit_sha = normalize_commit_sha("unknown")

    payload = fetcher(source_url, _timeout_sec(timeout_ms))
    database_path = _write_downloaded_file(payload, temp_dir=temp_dir)
    stats = validate_database(database_path, sample_lines=sample_lines)
    sha256 = compute_sha256(database_path)
    size_bytes = database_path.stat().st_size

    return OpraDownloadResult(
        commit_sha=commit_sha,
        source=source_key,
        source_url=source_url,
        database_path=database_path,
        sha256=sha256,
        size_bytes=size_bytes,
        stats=stats,
    )


__all__ = [
    "DEFAULT_SAMPLE_LINES",
    "DEFAULT_TIMEOUT_MS",
    "OPRA_CLOUDFLARE_URL",
    "OPRA_RAW_BASE",
    "OpraDownloadError",
    "OpraNetworkError",
    "OpraResponseError",
    "OpraDownloadResult",
    "download_opra_database",
    "resolve_latest_commit_sha",
    "validate_database",
]
