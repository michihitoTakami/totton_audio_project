"""I2S peer status endpoints (Pi -> Jetson) (#950)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..models import ApiResponse, I2sPeerStatus, I2sPeerStatusUpdate

router = APIRouter(prefix="/i2s", tags=["status"])


def _status_path() -> Path:
    # NOTE: テストで monkeypatch.setenv を使うため、env は都度参照する
    return Path(
        os.getenv("MAGICBOX_I2S_PEER_STATUS_PATH", "/tmp/magicbox-i2s-peer-status.json")
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _load_raw() -> dict[str, Any]:
    path = _status_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_raw(data: dict[str, Any]) -> None:
    path = _status_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except Exception:
        # 監視用途なので書けなくても致命にしない
        return


@router.get("/peer-status", response_model=I2sPeerStatus)
async def get_peer_status():
    """Get latest reported peer status (best-effort)."""
    raw = _load_raw()
    try:
        return I2sPeerStatus.model_validate(raw)
    except Exception:
        # 破損時は空にフォールバック
        return I2sPeerStatus()


@router.post("/peer-status", response_model=ApiResponse)
async def post_peer_status(update: I2sPeerStatusUpdate):
    """Accept peer status report.

    IMPORTANT:
    - 受け取っただけでは daemon 再起動等は行わない（ループ防止）
    - generation / updated_at_unix_ms が古い更新は無視する（stale対策）
    """
    current = _load_raw()
    now = _now_ms()

    incoming = update.model_dump()
    incoming["received_at_unix_ms"] = now

    # Stale guard: generation 優先、同世代なら updated_at_unix_ms を比較
    cur_gen = int(current.get("generation") or 0)
    cur_ts = int(current.get("updated_at_unix_ms") or 0)
    in_gen = int(incoming.get("generation") or 0)
    in_ts = int(incoming.get("updated_at_unix_ms") or 0)

    accept = False
    if in_gen > cur_gen:
        accept = True
    elif in_gen == cur_gen and in_ts >= cur_ts:
        accept = True

    if accept:
        _save_raw(incoming)
        return ApiResponse(success=True, message="peer status updated", data=incoming)
    return ApiResponse(
        success=True,
        message="peer status ignored (stale)",
        data={"current_generation": cur_gen, "current_updated_at_unix_ms": cur_ts},
    )
