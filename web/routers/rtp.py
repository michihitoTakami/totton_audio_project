"""RTP session management API."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Path, status

from ..models import (
    ApiResponse,
    RtpDiscoveryResponse,
    RtpSessionCreateRequest,
    RtpSessionCreateResponse,
    RtpSessionDetailResponse,
    RtpSessionListResponse,
)
from ..services import (
    build_session_config_payload,
    flag_existing_sessions,
    get_daemon_client,
    parse_config_snapshot,
    parse_discovery_streams,
    parse_metrics_payload,
    refresh_sessions_from_daemon,
    telemetry_store,
)

router = APIRouter(prefix="/api/rtp", tags=["rtp"])

SESSION_ID_PARAM = Path(
    ...,
    pattern=r"^[A-Za-z0-9._-]{1,64}$",
    description="RTP session identifier",
)


async def _execute_daemon(callable_fn: Callable[[], Any]):
    """Run blocking ZeroMQ calls on a worker thread."""
    return await asyncio.to_thread(callable_fn)


@router.post(
    "/sessions",
    response_model=RtpSessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create RTP session",
)
async def create_session(request: RtpSessionCreateRequest) -> RtpSessionCreateResponse:
    """Create a new RTP session and start receiving packets."""
    params = build_session_config_payload(request)

    def _command():
        with get_daemon_client() as client:
            return client.rtp_start_session(params)

    response = await _execute_daemon(_command)
    if not response.success:
        if response.error:
            raise response.error
        raise HTTPException(status_code=502, detail="Failed to start RTP session")

    data = response.data if isinstance(response.data, dict) else {}
    if "session_id" not in data:
        data["session_id"] = request.session_id
    snapshot = parse_config_snapshot(data)
    return RtpSessionCreateResponse(session=snapshot)


@router.get(
    "/sessions",
    response_model=RtpSessionListResponse,
    summary="List RTP sessions",
)
async def list_sessions() -> RtpSessionListResponse:
    """Return cached RTP session telemetry."""
    sessions, polled_at = telemetry_store.snapshot()
    if not sessions:
        await refresh_sessions_from_daemon()
        sessions, polled_at = telemetry_store.snapshot()
    return RtpSessionListResponse(sessions=sessions, polled_at_unix_ms=polled_at)


@router.get(
    "/discover",
    response_model=RtpDiscoveryResponse,
    summary="Discover available RTP senders",
)
async def discover_streams() -> RtpDiscoveryResponse:
    """Trigger a short-lived scan for available network RTP inputs."""

    def _command():
        with get_daemon_client() as client:
            return client.rtp_discover_streams()

    response = await _execute_daemon(_command)
    if not response.success:
        if response.error:
            raise response.error
        raise HTTPException(status_code=502, detail="Failed to discover RTP streams")

    data = response.data if isinstance(response.data, dict) else {}
    streams = parse_discovery_streams(data.get("streams", []))
    active_sessions, _ = telemetry_store.snapshot()
    flag_existing_sessions(streams, {session.session_id for session in active_sessions})
    return RtpDiscoveryResponse(
        streams=streams,
        scanned_at_unix_ms=data.get("scanned_at_unix_ms") or data.get("scanned_at"),
        duration_ms=data.get("duration_ms"),
    )


@router.get(
    "/sessions/{session_id}",
    response_model=RtpSessionDetailResponse,
    summary="Get RTP session detail",
)
async def get_session(session_id: str = SESSION_ID_PARAM) -> RtpSessionDetailResponse:
    """Return telemetry for a single RTP session."""
    cached, polled_at = telemetry_store.get(session_id)
    if cached:
        return RtpSessionDetailResponse(session=cached, polled_at_unix_ms=polled_at)

    def _command():
        with get_daemon_client() as client:
            return client.rtp_get_session(session_id)

    response = await _execute_daemon(_command)
    if not response.success:
        if response.error:
            raise response.error
        raise HTTPException(status_code=404, detail="RTP session not found")

    data = response.data if isinstance(response.data, dict) else {}
    telemetry_store.upsert(data)
    session, polled_at = telemetry_store.get(session_id)
    if not session:
        session = parse_metrics_payload(data)
        if session is None:
            raise HTTPException(
                status_code=502, detail="Daemon returned invalid telemetry payload"
            )
    return RtpSessionDetailResponse(session=session, polled_at_unix_ms=polled_at)


@router.delete(
    "/sessions/{session_id}",
    response_model=ApiResponse,
    summary="Delete RTP session",
)
async def delete_session(session_id: str = SESSION_ID_PARAM) -> ApiResponse:
    """Stop and remove an RTP session."""

    def _command():
        with get_daemon_client() as client:
            return client.rtp_stop_session(session_id)

    response = await _execute_daemon(_command)
    if not response.success:
        if response.error:
            raise response.error
        raise HTTPException(status_code=404, detail="RTP session not found")

    telemetry_store.remove(session_id)
    return ApiResponse(
        success=True,
        message="RTP session stopped",
        data={"session_id": session_id},
    )
