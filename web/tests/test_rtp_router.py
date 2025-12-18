from fastapi.testclient import TestClient

from web.main import create_app
from web.models import RtpBridgeStatus, RtpLatencyResponse
from web.routers import rtp as rtp_router
from web.services.rtp_bridge_client import (
    RtpBridgeConnectionError,
    RtpBridgeResponseError,
)


class _FakeBridgeClient:
    def __init__(self, status=None, latency_resp=None, error=None):
        self._status = status
        self._latency_resp = latency_resp
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def status(self):
        if self._error:
            raise self._error
        return self._status

    def set_latency(self, latency_ms: int):
        if self._error:
            raise self._error
        if self._latency_resp:
            return self._latency_resp
        return RtpLatencyResponse(status="ok", latency_ms=latency_ms)


def test_rtp_status_success(monkeypatch):
    expected = RtpBridgeStatus(
        running=True,
        latency_ms=120,
        sample_rate=48000,
        packets_received=10,
        packets_lost=1,
        jitter_ms=0.2,
        clock_drift_ppm=0.5,
    )

    monkeypatch.setattr(
        rtp_router, "get_rtp_bridge_client", lambda: _FakeBridgeClient(status=expected)
    )

    app = create_app(enable_rtp=True)
    client = TestClient(app)
    resp = client.get("/api/rtp/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["latency_ms"] == 120
    assert body["packets_received"] == 10


def test_rtp_latency_success(monkeypatch):
    monkeypatch.setattr(
        rtp_router,
        "get_rtp_bridge_client",
        lambda: _FakeBridgeClient(
            latency_resp=RtpLatencyResponse(status="ok", latency_ms=150)
        ),
    )

    app = create_app(enable_rtp=True)
    client = TestClient(app)
    resp = client.post("/api/rtp/latency", json={"latency_ms": 150})
    assert resp.status_code == 200
    assert resp.json()["latency_ms"] == 150


def test_rtp_latency_connection_error(monkeypatch):
    monkeypatch.setattr(
        rtp_router,
        "get_rtp_bridge_client",
        lambda: _FakeBridgeClient(error=RtpBridgeConnectionError("down")),
    )

    app = create_app(enable_rtp=True)
    client = TestClient(app)
    resp = client.post("/api/rtp/latency", json={"latency_ms": 120})
    assert resp.status_code == 502


def test_rtp_latency_bridge_error(monkeypatch):
    monkeypatch.setattr(
        rtp_router,
        "get_rtp_bridge_client",
        lambda: _FakeBridgeClient(error=RtpBridgeResponseError("bad request")),
    )

    app = create_app(enable_rtp=True)
    client = TestClient(app)
    resp = client.post("/api/rtp/latency", json={"latency_ms": 9})
    # request validation runs first, so ensure valid input to trigger bridge error
    resp = client.post("/api/rtp/latency", json={"latency_ms": 120})
    assert resp.status_code == 400
