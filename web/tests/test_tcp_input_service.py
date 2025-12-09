"""Tests for TCP input service utilities."""

import asyncio

import pytest

from web.models import TcpInputTelemetry
from web.services.tcp_input import (
    TcpTelemetryPoller,
    TcpTelemetryStore,
    parse_tcp_telemetry,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    """Force asyncio backend to avoid trio dependency for these tests."""
    return "asyncio"


def test_parse_tcp_telemetry_maps_fields():
    """Daemon payload should be normalized into telemetry model."""
    payload = {
        "listening": True,
        "bound_port": 46001,
        "client_connected": True,
        "streaming": True,
        "client_address": "10.0.0.5",
        "uptime_seconds": 42,
        "xrun_count": 2,
        "ring_buffer_frames": 8192,
        "watermark_frames": 6144,
        "buffered_frames": 4096,
        "max_buffered_frames": 12000,
        "dropped_frames": 32,
        "disconnect_reason": None,
        "connection_mode": "takeover",
        "priority_clients": ["10.0.0.1"],
        "rep_endpoint": "ipc:///tmp/jetson.sock",
        "pub_endpoint": "ipc:///tmp/jetson.pub",
        "last_header": {
            "sample_rate": 48000,
            "channels": 2,
            "format": 1,
            "version": 1,
        },
    }

    telemetry = parse_tcp_telemetry(payload)

    assert telemetry.listening is True
    assert telemetry.bound_port == 46001
    assert telemetry.connection_mode == "takeover"
    assert telemetry.priority_clients == ["10.0.0.1"]
    assert telemetry.client_address == "10.0.0.5"
    assert telemetry.uptime_seconds == 42
    assert telemetry.last_header is not None
    assert telemetry.last_header.format == "S16_LE"
    assert telemetry.last_header.sample_rate == 48000


def test_parse_tcp_telemetry_handles_missing_payload():
    """Missing payload should return default telemetry."""
    telemetry = parse_tcp_telemetry(None)
    assert telemetry.listening is False
    assert telemetry.bound_port is None
    assert telemetry.connection_mode == "single"


async def test_store_snapshot_and_error():
    """Store should keep telemetry and last error separately."""
    store = TcpTelemetryStore()
    telemetry = TcpInputTelemetry(listening=True, bound_port=5000)

    await store.update(telemetry)
    snap, updated, err = await store.snapshot()
    assert snap is telemetry
    assert err is None
    assert updated is not None

    await store.record_error("temporary failure")
    snap2, updated2, err2 = await store.snapshot()
    assert snap2 is telemetry
    assert err2 == "temporary failure"
    assert updated2 is not None
    assert updated2 >= updated


async def test_poller_updates_store(monkeypatch):
    """Poller should invoke fetcher and persist telemetry."""
    calls = {"count": 0}

    async def fetch():
        calls["count"] += 1
        return {"listening": True, "bound_port": 48000}

    store = TcpTelemetryStore()
    poller = TcpTelemetryPoller(fetcher=fetch, store=store, interval_seconds=0.01)

    await poller.start()
    await asyncio.sleep(0.04)
    await poller.stop()

    telemetry = await store.latest()
    assert telemetry.listening is True
    assert telemetry.bound_port == 48000
    assert calls["count"] >= 1


async def test_poller_can_be_disabled(monkeypatch):
    """Environment flag should disable polling."""
    monkeypatch.setenv("MAGICBOX_DISABLE_TCP_POLLING", "1")

    store = TcpTelemetryStore()
    poller = TcpTelemetryPoller(
        fetcher=lambda: TcpInputTelemetry(listening=True),
        store=store,
        interval_seconds=0.01,
    )

    await poller.start()
    await asyncio.sleep(0.02)
    await poller.stop()

    telemetry = await store.latest()
    assert telemetry.listening is False
