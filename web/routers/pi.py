"""Raspberry Pi Control API proxy endpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, TypeVar

from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError

from ..models import ApiResponse, PiStatus, PiUsbI2sConfig, PiUsbI2sConfigUpdate
from ..services.pi_client import (
    PiClient,
    PiClientError,
    PiConnectionError,
    PiResponseError,
    get_pi_client,
)

router = APIRouter(prefix="/pi", tags=["pi"])
logger = logging.getLogger(__name__)

T = TypeVar("T")


def _handle_error(exc: PiClientError) -> HTTPException:
    logger.warning("Pi API proxy failed: %s", exc)
    if isinstance(exc, PiConnectionError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, PiResponseError):
        return HTTPException(status_code=exc.status_code, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


async def _run_blocking(func: Callable[[PiClient], T]) -> T:
    """BlockingなPi API呼び出しをワーカースレッドで実行."""

    def _call() -> T:
        client = get_pi_client()
        return func(client)

    return await asyncio.to_thread(_call)


@router.get("/status", response_model=PiStatus, summary="Piステータス取得")
async def pi_status():
    """Pi control API からステータスを取得する."""
    try:
        data = await _run_blocking(lambda client: client.status())
        return PiStatus.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502, detail="Invalid Pi status payload"
        ) from exc
    except PiClientError as exc:
        raise _handle_error(exc) from exc


@router.get("/config", response_model=PiUsbI2sConfig, summary="Pi設定取得")
async def pi_config():
    """Pi control API から設定を取得する."""
    try:
        data = await _run_blocking(lambda client: client.get_config())
        return PiUsbI2sConfig.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502, detail="Invalid Pi config payload"
        ) from exc
    except PiClientError as exc:
        raise _handle_error(exc) from exc


@router.put("/config", response_model=PiUsbI2sConfig, summary="Pi設定更新")
async def pi_config_update(
    request: PiUsbI2sConfigUpdate,
    apply: bool = Query(
        default=True, description="Apply changes by restarting the bridge"
    ),
):
    """Pi control API へ設定更新を転送する."""
    payload = request.model_dump(exclude_none=True)
    try:
        data = await _run_blocking(
            lambda client: client.update_config(payload, apply=apply)
        )
        return PiUsbI2sConfig.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502, detail="Invalid Pi config payload"
        ) from exc
    except PiClientError as exc:
        raise _handle_error(exc) from exc


@router.post("/actions/restart", response_model=ApiResponse, summary="Piブリッジ再起動")
async def pi_restart_bridge():
    """Pi 側 USB-I2S ブリッジを再起動する."""
    try:
        data = await _run_blocking(lambda client: client.restart_bridge())
    except PiClientError as exc:
        raise _handle_error(exc) from exc
    return ApiResponse(success=True, message="pi bridge restarted", data=data)


__all__ = ["router"]
