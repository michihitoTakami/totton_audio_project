"""
Tests for RTP session FastAPI endpoints.
"""

import os
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MAGICBOX_DISABLE_RTP_POLLING", "1")

from web.main import app  # noqa: E402
from web.services import build_discovery_stream, telemetry_store  # noqa: E402
from web.services.daemon_client import DaemonError, DaemonResponse  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_store():
    telemetry_store.clear()
    yield
    telemetry_store.clear()


class DummyDaemonClient:
    """Context-manager friendly fake daemon client."""

    def __init__(
        self,
        *,
        start_response: DaemonResponse | None = None,
        stop_response: DaemonResponse | None = None,
        get_response: DaemonResponse | None = None,
        discover_response: DaemonResponse | None = None,
    ):
        self.start_response = start_response or DaemonResponse(success=True, data={})
        self.stop_response = stop_response or DaemonResponse(success=True, data={})
        self.get_response = get_response or DaemonResponse(success=True, data={})
        self.discover_response = discover_response or DaemonResponse(
            success=True,
            data={"streams": []},
        )
        self.last_params: dict[str, Any] | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def rtp_start_session(self, params: dict[str, Any]) -> DaemonResponse:
        self.last_params = params
        return self.start_response

    def rtp_stop_session(self, session_id: str) -> DaemonResponse:
        self.last_params = {"session_id": session_id}
        return self.stop_response

    def rtp_get_session(self, session_id: str) -> DaemonResponse:
        self.last_params = {"session_id": session_id}
        return self.get_response

    def rtp_discover_streams(self) -> DaemonResponse:
        return self.discover_response


def test_create_session_success(client):
    """POST /api/rtp/sessions should forward payload to daemon."""
    dummy = DummyDaemonClient(
        start_response=DaemonResponse(
            success=True,
            data={
                "session_id": "aes67-main",
                "bind_address": "0.0.0.0",
                "port": 6000,
            },
        )
    )

    with patch("web.routers.rtp.get_daemon_client", return_value=dummy):
        response = client.post("/api/rtp/sessions", json={"session_id": "aes67-main"})

    assert response.status_code == 201
    body = response.json()
    assert body["session"]["session_id"] == "aes67-main"
    assert dummy.last_params is not None
    assert dummy.last_params["session_id"] == "aes67-main"
    assert dummy.last_params["payload_type"] == 97


def test_create_session_invalid_session_id(client):
    """Invalid session IDs should trigger validation errors."""
    response = client.post("/api/rtp/sessions", json={"session_id": "invalid id"})
    assert response.status_code == 422


def test_list_sessions_refreshes_from_daemon(client):
    """GET /api/rtp/sessions should refresh cache when empty."""

    async def fake_refresh():
        telemetry_store.update_from_list(
            [
                {
                    "session_id": "aes67-main",
                    "packets_received": 10,
                    "packets_dropped": 0,
                }
            ]
        )
        sessions, _ = telemetry_store.snapshot()
        return sessions

    with patch("web.routers.rtp.refresh_sessions_from_daemon", new=fake_refresh):
        response = client.get("/api/rtp/sessions")

    assert response.status_code == 200
    data = response.json()
    assert data["sessions"][0]["session_id"] == "aes67-main"
    assert data["sessions"][0]["packets_received"] == 10


def test_get_session_fetches_from_daemon(client):
    """GET /api/rtp/sessions/{id} should fallback to daemon when cache empty."""
    dummy = DummyDaemonClient(
        get_response=DaemonResponse(
            success=True,
            data={
                "session_id": "aes67-main",
                "packets_received": 42,
                "packets_dropped": 1,
            },
        )
    )

    with patch("web.routers.rtp.get_daemon_client", return_value=dummy):
        response = client.get("/api/rtp/sessions/aes67-main")

    assert response.status_code == 200
    data = response.json()
    assert data["session"]["packets_received"] == 42


def test_delete_session_error_propagates(client):
    """Daemon error should map to HTTP status via exception handler."""
    dummy = DummyDaemonClient(
        stop_response=DaemonResponse(
            success=False,
            error=DaemonError(
                error_code="AUDIO_RTP_SESSION_NOT_FOUND",
                message="Session missing",
            ),
        )
    )

    with patch("web.routers.rtp.get_daemon_client", return_value=dummy):
        response = client.delete("/api/rtp/sessions/unknown")

    assert response.status_code == 404


def test_discover_streams_success_marks_existing(client):
    """GET /api/rtp/discover surfaces scanner output and marks active sessions."""
    telemetry_store.update_from_list(
        [
            {
                "session_id": "aes67-main",
                "packets_received": 10,
                "packets_dropped": 0,
            }
        ]
    )
    dummy = DummyDaemonClient(
        discover_response=DaemonResponse(
            success=True,
            data={
                "streams": [
                    {
                        "session_id": "aes67-main",
                        "display_name": "Main AES67",
                        "source_host": "239.0.0.1",
                        "port": 5004,
                        "status": "active",
                        "sample_rate": 48000,
                        "channels": 2,
                    },
                    {
                        "display_name": "Backup Feed",
                        "source_host": "10.0.0.5",
                        "port": 6000,
                        "status": "idle",
                    },
                ],
                "scanned_at_unix_ms": 123456789,
                "duration_ms": 120,
            },
        )
    )

    with patch("web.routers.rtp.get_daemon_client", return_value=dummy):
        response = client.get("/api/rtp/discover")

    assert response.status_code == 200
    data = response.json()
    assert data["scanned_at_unix_ms"] == 123456789
    assert len(data["streams"]) == 2
    assert data["streams"][0]["existing_session"] is True
    # Second stream should have auto-generated session_id from display_name or source_host+port
    assert "session_id" in data["streams"][1]
    assert len(data["streams"][1]["session_id"]) > 0
    assert data["streams"][1]["existing_session"] is False


def test_discover_streams_error_propagates(client):
    """Daemon discovery errors bubble up with mapped HTTP code."""
    dummy = DummyDaemonClient(
        discover_response=DaemonResponse(
            success=False,
            error=DaemonError(
                error_code="IPC_DAEMON_NOT_RUNNING",
                message="Daemon offline",
            ),
        )
    )

    with patch("web.routers.rtp.get_daemon_client", return_value=dummy):
        response = client.get("/api/rtp/discover")

    assert response.status_code == 503


def test_build_discovery_stream_preserves_existing_flag():
    """Daemon discovery payloads should propagate existing_session/multicast hints."""
    payload = {
        "session_id": "aes67-main",
        "display_name": "AES67 Main",
        "source_host": "239.1.1.10",
        "port": 5004,
        "existing_session": True,
        "payload_type": 97,
        "multicast": True,
    }

    stream = build_discovery_stream(payload)

    assert stream.existing_session is True
    assert stream.payload_type == 97
    assert stream.multicast is True
    assert stream.session_id == "aes67-main"
