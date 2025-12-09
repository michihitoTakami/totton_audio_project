"""TCP入力APIルーターの動作検証 (#686)."""

from fastapi.testclient import TestClient
import pytest

from web.error_codes import ErrorCode
from web.main import app
from web.services.daemon_client import DaemonError, DaemonResponse


class _DummyClient:
    """Context managerモックでZeroMQ呼び出しを置き換える."""

    def __init__(self, responses, recorder=None):
        self._responses = responses
        self.recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def tcp_input_status(self):
        return self._responses["status"]

    def tcp_input_start(self):
        return self._responses["start"]

    def tcp_input_stop(self):
        return self._responses["stop"]

    def tcp_input_config_update(self, params):
        if self.recorder is not None:
            self.recorder["params"] = params
        return self._responses["config"]


@pytest.fixture(autouse=True)
def disable_poller(monkeypatch):
    """テスト時はポーラーを起動させない."""

    async def _noop():
        return None

    monkeypatch.setattr("web.main._tcp_telemetry_poller.start", _noop)
    monkeypatch.setattr("web.main._tcp_telemetry_poller.stop", _noop)


def _client_with_responses(monkeypatch, responses, recorder=None):
    """tcp_inputルーターのget_daemon_clientをモック."""

    def _factory(*args, **kwargs):
        return _DummyClient(responses, recorder=recorder)

    monkeypatch.setattr("web.routers.tcp_input.get_daemon_client", _factory)


def test_status_returns_status_response(monkeypatch):
    """settings+telemetryを統合して返却する."""
    responses = {
        "status": DaemonResponse(
            success=True,
            data={
                "settings": {
                    "enabled": True,
                    "bind_address": "0.0.0.0",
                    "port": 47000,
                    "buffer_size_bytes": 262144,
                    "connection_mode": "single",
                    "priority_clients": [],
                },
                "telemetry": {
                    "listening": True,
                    "bound_port": 47000,
                    "priority_clients": [],
                    "rep_endpoint": "ipc:///tmp/tcp.rep",
                },
            },
        ),
        "start": DaemonResponse(success=True, data={}),
        "stop": DaemonResponse(success=True, data={}),
        "config": DaemonResponse(success=True, data={}),
    }
    _client_with_responses(monkeypatch, responses)

    client = TestClient(app)
    response = client.get("/api/tcp-input/status")

    assert response.status_code == 200
    body = response.json()
    assert body["settings"]["port"] == 47000
    assert body["telemetry"]["bound_port"] == 47000
    assert body["telemetry"]["rep_endpoint"] == "ipc:///tmp/tcp.rep"


def test_status_returns_502_on_connection_error(monkeypatch):
    """デーモン接続失敗は502 Bad Gatewayを返す."""
    responses = {
        "status": DaemonResponse(
            success=False,
            error=DaemonError(
                error_code=ErrorCode.IPC_CONNECTION_FAILED.value,
                message="unreachable",
            ),
        ),
        "start": DaemonResponse(success=True, data={}),
        "stop": DaemonResponse(success=True, data={}),
        "config": DaemonResponse(success=True, data={}),
    }
    _client_with_responses(monkeypatch, responses)

    client = TestClient(app)
    response = client.get("/api/tcp-input/status")

    assert response.status_code == 502
    assert (
        response.json()["detail"]["error_code"] == ErrorCode.IPC_CONNECTION_FAILED.value
    )


def test_config_update_propagates_payload(monkeypatch):
    """設定更新リクエストがZeroMQにシリアライズされる."""
    recorder = {}
    responses = {
        "status": DaemonResponse(success=True, data={}),
        "start": DaemonResponse(success=True, data={}),
        "stop": DaemonResponse(success=True, data={}),
        "config": DaemonResponse(success=True, data={"applied": True}),
    }
    _client_with_responses(monkeypatch, responses, recorder=recorder)

    client = TestClient(app)
    payload = {
        "bind_address": "127.0.0.1",
        "port": 48000,
        "connection_mode": "priority",
        "priority_clients": ["10.0.0.1"],
    }
    response = client.put("/api/tcp-input/config", json=payload)

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert recorder["params"].port == 48000
    assert recorder["params"].connection_mode == "priority"


def test_start_and_stop(monkeypatch):
    """開始/停止エンドポイントが成功レスポンスを返す."""
    responses = {
        "status": DaemonResponse(success=True, data={}),
        "start": DaemonResponse(success=True, data={}),
        "stop": DaemonResponse(success=True, data={}),
        "config": DaemonResponse(success=True, data={}),
    }
    _client_with_responses(monkeypatch, responses)

    client = TestClient(app)

    start_resp = client.post("/api/tcp-input/start")
    stop_resp = client.post("/api/tcp-input/stop")

    assert start_resp.status_code == 200
    assert stop_resp.status_code == 200
    assert start_resp.json()["success"] is True
    assert stop_resp.json()["success"] is True
