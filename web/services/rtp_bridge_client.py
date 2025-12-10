"""Client for the RTP ZeroMQ bridge (STATUS / SET_LATENCY)."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import ValidationError
import zmq

from ..models import RtpBridgeStatus, RtpLatencyRequest, RtpLatencyResponse
from .rtp_input import DEFAULT_LATENCY_MS, DEFAULT_SAMPLE_RATE

DEFAULT_RTP_BRIDGE_ENDPOINT = os.getenv(
    "RTP_BRIDGE_ENDPOINT", "ipc:///tmp/rtp_receiver.sock"
)
DEFAULT_RTP_BRIDGE_TIMEOUT_MS = int(os.getenv("RTP_BRIDGE_TIMEOUT_MS", "1500"))


class RtpBridgeClientError(Exception):
    """Base class for bridge client errors."""


class RtpBridgeConnectionError(RtpBridgeClientError):
    """Raised when ZeroMQ send/recv fails."""


class RtpBridgeResponseError(RtpBridgeClientError):
    """Raised when bridge returns an error or invalid payload."""


class RtpBridgeClient:
    """Thin synchronous client for the RTP ZeroMQ bridge."""

    def __init__(self, endpoint: str | None = None, timeout_ms: int | None = None):
        self.endpoint = endpoint or DEFAULT_RTP_BRIDGE_ENDPOINT
        self.timeout_ms = timeout_ms or DEFAULT_RTP_BRIDGE_TIMEOUT_MS
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    def __enter__(self) -> "RtpBridgeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:  # noqa: ANN001
        self.close()
        return False

    def close(self) -> None:
        """Close socket/context if present."""
        if self._socket is not None:
            try:
                self._socket.close(0)
            finally:
                self._socket = None
        if self._context is not None:
            try:
                self._context.term()
            finally:
                self._context = None

    def _ensure_socket(self) -> zmq.Socket:
        if self._socket is None:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REQ)
            self._socket.rcvtimeo = self.timeout_ms
            self._socket.sndtimeo = self.timeout_ms
            self._socket.linger = 0
            self._socket.connect(self.endpoint)
        return self._socket

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            socket = self._ensure_socket()
            socket.send_json(payload)
            reply = socket.recv_json()
        except zmq.Again as exc:
            raise RtpBridgeConnectionError("ZeroMQ request timed out") from exc
        except zmq.ZMQError as exc:
            raise RtpBridgeConnectionError(f"ZeroMQ error: {exc}") from exc

        if not isinstance(reply, dict):
            raise RtpBridgeResponseError("Invalid response: expected JSON object")
        return reply

    def status(self) -> RtpBridgeStatus:
        """Fetch RTP stats via STATUS."""
        reply = self._request({"cmd": "STATUS"})
        if str(reply.get("status")) != "ok":
            raise RtpBridgeResponseError(str(reply.get("message", "unknown error")))

        data = reply.get("data") or {}
        if not isinstance(data, dict):
            raise RtpBridgeResponseError("STATUS payload must be an object")

        payload = {
            "running": bool(data.get("running", True)),
            "latency_ms": data.get("latency_ms", DEFAULT_LATENCY_MS),
            "sample_rate": data.get("sample_rate", DEFAULT_SAMPLE_RATE),
            "packets_received": data.get("packets_received", 0),
            "packets_lost": data.get("packets_lost", 0),
            "jitter_ms": data.get("jitter_ms", 0.0),
            "clock_drift_ppm": data.get("clock_drift_ppm", 0.0),
        }

        try:
            return RtpBridgeStatus.model_validate(payload)
        except ValidationError as exc:
            raise RtpBridgeResponseError(f"Invalid STATUS payload: {exc}") from exc

    def set_latency(self, latency_ms: int) -> RtpLatencyResponse:
        """Apply latency via SET_LATENCY."""
        try:
            request = RtpLatencyRequest(latency_ms=latency_ms)
        except ValidationError as exc:
            raise RtpBridgeResponseError(f"Invalid latency request: {exc}") from exc
        reply = self._request(
            {"cmd": "SET_LATENCY", "params": {"latency_ms": request.latency_ms}}
        )

        if str(reply.get("status")) != "ok":
            raise RtpBridgeResponseError(
                str(reply.get("message", "failed to set latency"))
            )

        data = reply.get("data") or {}
        try:
            return RtpLatencyResponse(
                status=str(reply.get("status", "ok")),
                latency_ms=int(data.get("latency_ms", request.latency_ms)),
            )
        except (ValueError, TypeError, ValidationError) as exc:
            raise RtpBridgeResponseError(
                f"Invalid SET_LATENCY response: {exc}"
            ) from exc


def get_rtp_bridge_client() -> RtpBridgeClient:
    """Factory for dependency injection."""
    return RtpBridgeClient()
