import zmq

from raspberry_pi.rtp_receiver.zmq_bridge import RtpStatsStore, RtpZmqBridge


def _client(endpoint: str, timeout_ms: int = 1000):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.rcvtimeo = timeout_ms
    sock.sndtimeo = timeout_ms
    sock.connect(endpoint)
    return ctx, sock


def test_status_returns_latest_stats(tmp_path):
    endpoint = f"ipc://{tmp_path/'rtp_bridge_status.sock'}"
    stats = RtpStatsStore()
    stats.update(
        packets_received=123,
        packets_lost=4,
        jitter_ms=1.25,
        clock_drift_ppm=-12.5,
        sample_rate=48000,
    )

    with RtpZmqBridge(stats, lambda _lat: None, endpoint=endpoint, timeout_ms=1000):
        ctx, sock = _client(endpoint)
        try:
            sock.send_json({"cmd": "STATUS"})
            reply = sock.recv_json()
        finally:
            sock.close(0)
            ctx.term()

    assert reply["status"] == "ok"
    data = reply["data"]
    assert data["packets_received"] == 123
    assert data["packets_lost"] == 4
    assert data["sample_rate"] == 48000
    assert abs(data["jitter_ms"] - 1.25) < 1e-6
    assert abs(data["clock_drift_ppm"] + 12.5) < 1e-6


def test_set_latency_validates_and_applies(tmp_path):
    endpoint = f"ipc://{tmp_path/'rtp_bridge_latency.sock'}"
    applied: list[int] = []

    def _setter(value: int):
        applied.append(value)

    with RtpZmqBridge(RtpStatsStore(), _setter, endpoint=endpoint, timeout_ms=1000):
        ctx, sock = _client(endpoint)
        try:
            sock.send_json({"cmd": "SET_LATENCY", "params": {"latency_ms": 120}})
            ok_reply = sock.recv_json()

            sock.send_json({"cmd": "SET_LATENCY", "params": {"latency_ms": 5}})
            ng_reply = sock.recv_json()
        finally:
            sock.close(0)
            ctx.term()

    assert ok_reply["status"] == "ok"
    assert ok_reply["data"]["latency_ms"] == 120
    assert applied == [120]

    assert ng_reply["status"] == "error"
    assert "latency_ms" in ng_reply["message"]
    assert applied == [120]


def test_unknown_command_returns_error(tmp_path):
    endpoint = f"ipc://{tmp_path/'rtp_bridge_unknown.sock'}"
    with RtpZmqBridge(
        RtpStatsStore(), lambda _lat: None, endpoint=endpoint, timeout_ms=1000
    ):
        ctx, sock = _client(endpoint)
        try:
            sock.send_json({"cmd": "PING"})
            reply = sock.recv_json()
        finally:
            sock.close(0)
            ctx.term()

    assert reply["status"] == "error"
    assert "unknown command" in reply["message"]


def test_invalid_json_does_not_break_loop(tmp_path):
    endpoint = f"ipc://{tmp_path/'rtp_bridge_invalid.sock'}"
    stats = RtpStatsStore()
    stats.update(packets_received=1)

    with RtpZmqBridge(stats, lambda _lat: None, endpoint=endpoint, timeout_ms=1000):
        ctx, sock = _client(endpoint)
        try:
            # 1st request: invalid JSON payload
            sock.send(b"not-json")
            reply1 = sock.recv_json()

            # 2nd request: valid STATUS should still work
            sock.send_json({"cmd": "STATUS"})
            reply2 = sock.recv_json()
        finally:
            sock.close(0)
            ctx.term()

    assert reply1["status"] == "error"
    assert "invalid json" in reply1["message"]
    assert reply2["status"] == "ok"
    assert reply2["data"]["packets_received"] == 1
