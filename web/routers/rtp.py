"""RTP ZeroMQ bridge endpoints."""

import asyncio
from typing import Callable, TypeVar

from fastapi import APIRouter, HTTPException

from ..models import RtpBridgeStatus, RtpLatencyRequest, RtpLatencyResponse
from ..services.rtp_bridge_client import (
    RtpBridgeClient,
    RtpBridgeClientError,
    RtpBridgeConnectionError,
    RtpBridgeResponseError,
    get_rtp_bridge_client,
)

router = APIRouter(prefix="/api/rtp", tags=["rtp"])

T = TypeVar("T")


def _handle_error(exc: RtpBridgeClientError) -> HTTPException:
    if isinstance(exc, RtpBridgeConnectionError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, RtpBridgeResponseError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


async def _run_blocking(func: Callable[[RtpBridgeClient], T]) -> T:
    """BlockingsなZeroMQクライアント呼び出しをワーカースレッドで実行."""

    def _call() -> T:
        with get_rtp_bridge_client() as client:
            return func(client)

    return await asyncio.to_thread(_call)


@router.get(
    "/status",
    response_model=RtpBridgeStatus,
    summary="RTPステータス取得 (ZeroMQ)",
)
async def rtp_status():
    """ZeroMQブリッジからRTP統計を取得する."""
    try:
        return await _run_blocking(lambda client: client.status())
    except RtpBridgeClientError as exc:
        raise _handle_error(exc) from exc


@router.post(
    "/latency",
    response_model=RtpLatencyResponse,
    summary="RTPレイテンシ変更 (ZeroMQ)",
)
async def rtp_set_latency(request: RtpLatencyRequest):
    """ZeroMQブリッジ経由でレイテンシを変更する."""
    try:
        return await _run_blocking(
            lambda client: client.set_latency(request.latency_ms)
        )
    except RtpBridgeClientError as exc:
        raise _handle_error(exc) from exc


__all__ = ["router"]
