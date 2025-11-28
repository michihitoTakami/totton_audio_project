"""Helpers for RTP session control and telemetry polling."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any

from ..models import (
    RtpAdvancedSettings,
    RtpEndpointSettings,
    RtpFormatSettings,
    RtpRtcpSettings,
    RtpSdpConfig,
    RtpSecurityConfig,
    RtpSessionConfigSnapshot,
    RtpSessionCreateRequest,
    RtpSessionMetrics,
    RtpSyncSettings,
)
from .daemon_client import DaemonResponse, get_daemon_client

logger = logging.getLogger(__name__)


def _ms_now() -> int:
    return int(time.time() * 1000)


def _format_media_descriptor(fmt: RtpFormatSettings) -> str:
    return f"L{fmt.bits_per_sample}/{fmt.sample_rate}/{fmt.channels}"


def _render_sdp_body(
    request: RtpSessionCreateRequest,
    endpoint: RtpEndpointSettings,
    fmt: RtpFormatSettings,
    sync: RtpSyncSettings,
    rtcp: RtpRtcpSettings,
    sdp: RtpSdpConfig | None,
    security: RtpSecurityConfig | None,
) -> str | None:
    if not sdp and not security:
        return None
    if sdp and sdp.body:
        return sdp.body.strip()

    connection_address = (
        (sdp.connection_address if sdp else None)
        or endpoint.multicast_group
        or endpoint.bind_address
    )

    media_format = sdp.media_format if sdp and sdp.media_format else _format_media_descriptor(fmt)

    lines = [
        "v=0",
        f"o=- 0 0 IN IP4 {connection_address}",
        f"s={(sdp.session_name if sdp else 'MagicBox RTP Session')}",
        f"c=IN IP4 {connection_address}",
        "t=0 0",
        f"m=audio {endpoint.port} RTP/AVP {fmt.payload_type}",
        f"a=rtpmap:{fmt.payload_type} {media_format}",
    ]

    if rtcp.enable:
        rtcp_port = rtcp.port or (endpoint.port + 1)
        lines.append(f"a=rtcp:{rtcp_port}")

    if sdp and sdp.media_clock:
        lines.append(f"a=mediaclk:{sdp.media_clock}")

    if security:
        lines.append(
            f"a=crypto:1 {security.crypto_suite} inline:{security.key_base64}"
        )

    return "\r\n".join(lines) + "\r\n"


def build_session_config_payload(request: RtpSessionCreateRequest) -> dict[str, Any]:
    """Convert API request into daemon SessionConfig payload."""
    endpoint = request.endpoint
    fmt = request.format
    sync = request.sync
    rtcp = request.rtcp
    advanced = request.advanced

    params: dict[str, Any] = {
        "session_id": request.session_id,
        "bind_address": endpoint.bind_address,
        "port": endpoint.port,
        "multicast": endpoint.multicast,
        "ttl": endpoint.ttl,
        "dscp": endpoint.dscp,
        "sample_rate": fmt.sample_rate,
        "channels": fmt.channels,
        "bits_per_sample": fmt.bits_per_sample,
        "big_endian": fmt.big_endian,
        "signed": fmt.signed_samples,
        "payload_type": fmt.payload_type,
        "socket_buffer_bytes": advanced.socket_buffer_bytes,
        "mtu_bytes": advanced.mtu_bytes,
        "target_latency_ms": sync.target_latency_ms,
        "watchdog_timeout_ms": sync.watchdog_timeout_ms,
        "telemetry_interval_ms": sync.telemetry_interval_ms,
        "enable_rtcp": rtcp.enable,
        "enable_ptp": sync.enable_ptp,
        "ptp_domain": sync.ptp_domain,
    }

    if endpoint.source_host:
        params["source_host"] = endpoint.source_host
    if endpoint.multicast_group:
        params["multicast_group"] = endpoint.multicast_group
    if endpoint.interface:
        params["interface"] = endpoint.interface
    if rtcp.port:
        params["rtcp_port"] = rtcp.port
    if sync.ptp_interface:
        params["ptp_interface"] = sync.ptp_interface

    sdp_body = _render_sdp_body(
        request,
        endpoint,
        fmt,
        sync,
        rtcp,
        request.sdp,
        request.security,
    )
    if sdp_body:
        params["sdp"] = sdp_body

    return params


def parse_config_snapshot(payload: dict[str, Any]) -> RtpSessionConfigSnapshot:
    """Convert daemon payload into typed snapshot."""
    return RtpSessionConfigSnapshot.from_daemon(payload)


def parse_metrics_payload(
    payload: dict[str, Any], polled_at_ms: int | None = None
) -> RtpSessionMetrics | None:
    """Convert daemon metrics JSON into RtpSessionMetrics."""
    session_id = payload.get("session_id")
    if not isinstance(session_id, str):
        return None
    return RtpSessionMetrics.from_daemon(payload, polled_at_unix_ms=polled_at_ms)


class RtpTelemetryStore:
    """In-memory cache of RTP telemetry for UI consumption."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, RtpSessionMetrics] = {}
        self._polled_at_ms: int | None = None

    def update_from_list(self, sessions: list[dict[str, Any]]) -> None:
        polled_at = _ms_now()
        with self._lock:
            updated: dict[str, RtpSessionMetrics] = {}
            for payload in sessions:
                metrics = parse_metrics_payload(payload, polled_at)
                if metrics:
                    updated[metrics.session_id] = metrics
            self._sessions = updated
            self._polled_at_ms = polled_at if updated else self._polled_at_ms

    def upsert(self, payload: dict[str, Any]) -> None:
        metrics = parse_metrics_payload(payload, _ms_now())
        if not metrics:
            return
        with self._lock:
            self._sessions[metrics.session_id] = metrics
            self._polled_at_ms = metrics.updated_at_unix_ms

    def remove(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def snapshot(self) -> tuple[list[RtpSessionMetrics], int | None]:
        with self._lock:
            sessions = [metrics.model_copy() for metrics in self._sessions.values()]
            return sessions, self._polled_at_ms

    def get(self, session_id: str) -> tuple[RtpSessionMetrics | None, int | None]:
        with self._lock:
            metrics = self._sessions.get(session_id)
            if metrics:
                return metrics.model_copy(), self._polled_at_ms
            return None, self._polled_at_ms

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._polled_at_ms = None


class RtpTelemetryPoller:
    """Background task that polls daemon for RTP telemetry."""

    def __init__(
        self,
        store: RtpTelemetryStore,
        *,
        interval_s: float = 1.5,
        enabled: bool = True,
    ) -> None:
        self._store = store
        self._interval_s = max(0.25, interval_s)
        self._enabled = enabled
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def start(self) -> None:
        if not self._enabled:
            return
        if self._task and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._run())

    async def stop(self) -> None:
        if not self._task:
            return
        if self._stop_event:
            self._stop_event.set()
        try:
            await self._task
        finally:
            self._task = None
            self._stop_event = None

    async def _run(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self._poll_once()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("RTP telemetry poller error: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval_s)
            except asyncio.TimeoutError:
                continue

    async def _poll_once(self) -> None:
        def _execute() -> DaemonResponse:
            with get_daemon_client() as client:
                return client.rtp_list_sessions()

        response = await asyncio.to_thread(_execute)
        if response.success and isinstance(response.data, dict):
            sessions = response.data.get("sessions", [])
            if isinstance(sessions, list):
                self._store.update_from_list(sessions)
        elif response.error:
            logger.debug("RTP telemetry poll failed: %s", response.error)


def _is_polling_enabled() -> bool:
    flag = os.getenv("MAGICBOX_DISABLE_RTP_POLLING", "").lower()
    return flag not in {"1", "true", "yes", "on"}


def _poll_interval() -> float:
    value = os.getenv("MAGICBOX_RTP_POLL_INTERVAL_SEC", "").strip()
    try:
        parsed = float(value)
        return parsed if parsed > 0 else 1.5
    except ValueError:
        return 1.5


telemetry_store = RtpTelemetryStore()
telemetry_poller = RtpTelemetryPoller(
    telemetry_store,
    interval_s=_poll_interval(),
    enabled=_is_polling_enabled(),
)


async def refresh_sessions_from_daemon() -> list[RtpSessionMetrics]:
    """Force-refresh telemetry cache by calling RTP_LIST_SESSIONS."""

    def _execute() -> DaemonResponse:
        with get_daemon_client() as client:
            return client.rtp_list_sessions()

    response = await asyncio.to_thread(_execute)
    if not response.success:
        if response.error:
            raise response.error
        return []

    sessions = response.data.get("sessions", []) if isinstance(response.data, dict) else []
    if isinstance(sessions, list):
        telemetry_store.update_from_list(sessions)
    snapshot, _ = telemetry_store.snapshot()
    return snapshot

