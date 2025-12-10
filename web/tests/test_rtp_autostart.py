from fastapi.testclient import TestClient

from web.main import app
from web.services.rtp_input import get_rtp_receiver_manager


class _SpyRtpManager:
    def __init__(self):
        self.start_called = 0
        self.shutdown_called = 0

    async def start(self):
        self.start_called += 1

    async def shutdown(self):
        self.shutdown_called += 1


def _make_client(monkeypatch, autostart_env: str | None):
    manager = _SpyRtpManager()
    if autostart_env is None:
        monkeypatch.delenv("MAGICBOX_RTP_AUTOSTART", raising=False)
    else:
        monkeypatch.setenv("MAGICBOX_RTP_AUTOSTART", autostart_env)
    app.dependency_overrides[get_rtp_receiver_manager] = lambda: manager
    client = TestClient(app)
    return client, manager


def test_autostart_enabled_by_default(monkeypatch):
    client, manager = _make_client(monkeypatch, autostart_env=None)
    try:
        with client:
            assert manager.start_called == 1
        assert manager.shutdown_called == 1
    finally:
        app.dependency_overrides.clear()


def test_autostart_can_be_disabled(monkeypatch):
    client, manager = _make_client(monkeypatch, autostart_env="false")
    try:
        with client:
            assert manager.start_called == 0
        assert manager.shutdown_called == 1
    finally:
        app.dependency_overrides.clear()
