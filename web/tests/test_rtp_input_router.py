from fastapi.testclient import TestClient

from web.main import create_app
from web.services.rtp_input import (
    get_rtp_receiver_manager,
    load_default_settings,
)


class _FakeManager:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.settings = load_default_settings()

    async def status(self):
        return {
            "running": self.started and not self.stopped,
            "pid": 123,
            "last_error": None,
            "settings": self.settings,
        }

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def apply_config(self, request):
        data = self.settings.model_dump()
        data.update(request.model_dump(exclude_none=True))
        self.settings = type(self.settings).model_validate(data)
        return self.settings


def test_rtp_status_and_start_stop(monkeypatch):
    manager = _FakeManager()
    app = create_app(enable_rtp=True)
    app.dependency_overrides[get_rtp_receiver_manager] = lambda: manager

    client = TestClient(app)

    resp = client.get("/api/rtp-input/status")
    assert resp.status_code == 200
    assert resp.json()["running"] is False

    resp = client.post("/api/rtp-input/start")
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    resp = client.post("/api/rtp-input/stop")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    app.dependency_overrides.clear()


def test_rtp_config_update(monkeypatch):
    manager = _FakeManager()
    app = create_app(enable_rtp=True)
    app.dependency_overrides[get_rtp_receiver_manager] = lambda: manager

    client = TestClient(app)

    resp = client.put(
        "/api/rtp-input/config",
        json={
            "latency_ms": 150,
            "encoding": "L32",
            "rtcp_port": 47001,
            "rtcp_send_port": 47002,
            "sender_host": "192.168.55.1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["latency_ms"] == 150
    assert body["encoding"] == "L32"
    assert body["rtcp_port"] == 47001
    assert body["rtcp_send_port"] == 47002
    assert body["sender_host"] == "192.168.55.1"

    # cleanup override
    app.dependency_overrides.clear()
