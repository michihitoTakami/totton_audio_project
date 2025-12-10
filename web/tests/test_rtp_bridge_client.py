import pytest

from raspberry_pi.rtp_receiver.zmq_bridge import RtpStatsStore, RtpZmqBridge
from web.services.rtp_bridge_client import (
    RtpBridgeClient,
    RtpBridgeResponseError,
)


def test_bridge_client_status(tmp_path):
    endpoint = f"ipc://{tmp_path/'rtp_client_status.sock'}"
    stats = RtpStatsStore()
    stats.update(
        running=True,
        latency_ms=130,
        packets_received=42,
        packets_lost=2,
        jitter_ms=0.5,
        clock_drift_ppm=-1.2,
        sample_rate=48000,
    )

    with RtpZmqBridge(stats, lambda _lat: None, endpoint=endpoint, timeout_ms=1000):
        with RtpBridgeClient(endpoint=endpoint, timeout_ms=500) as client:
            status = client.status()

    assert status.running is True
    assert status.latency_ms == 130
    assert status.sample_rate == 48000
    assert status.packets_received == 42
    assert status.packets_lost == 2
    assert abs(status.jitter_ms - 0.5) < 1e-6
    assert abs(status.clock_drift_ppm + 1.2) < 1e-6


def test_bridge_client_set_latency_and_status(tmp_path):
    endpoint = f"ipc://{tmp_path/'rtp_client_latency.sock'}"
    applied: list[int] = []

    with RtpZmqBridge(
        RtpStatsStore(),
        lambda value: applied.append(value),
        endpoint=endpoint,
        timeout_ms=1000,
    ):
        with RtpBridgeClient(endpoint=endpoint, timeout_ms=500) as client:
            resp = client.set_latency(140)
            status = client.status()

    assert resp.latency_ms == 140
    assert applied == [140]
    assert status.latency_ms == 140
    assert status.running is True


def test_bridge_client_rejects_invalid_latency(tmp_path):
    endpoint = f"ipc://{tmp_path/'rtp_client_latency_err.sock'}"
    with RtpZmqBridge(
        RtpStatsStore(), lambda _lat: None, endpoint=endpoint, timeout_ms=1000
    ):
        with RtpBridgeClient(endpoint=endpoint, timeout_ms=500) as client:
            with pytest.raises(RtpBridgeResponseError):
                client.set_latency(5)
