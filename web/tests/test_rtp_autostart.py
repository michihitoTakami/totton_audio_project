from fastapi.testclient import TestClient

from web.main import create_app
from web.services.rtp_input import get_rtp_receiver_manager


class _SpyRtpManager:
    def __init__(self):
        self.start_called = 0
        self.shutdown_called = 0

    async def start(self):
        self.start_called += 1

    async def shutdown(self):
        self.shutdown_called += 1


def _make_client(monkeypatch, autostart_env: str | None, enable_env: str | None):
    manager = _SpyRtpManager()
    if autostart_env is None:
        monkeypatch.delenv("TOTTON_AUDIO_RTP_AUTOSTART", raising=False)
    else:
        monkeypatch.setenv("TOTTON_AUDIO_RTP_AUTOSTART", autostart_env)

    enable_rtp = enable_env is not None and enable_env.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    app = create_app(enable_rtp=enable_rtp)
    app.dependency_overrides[get_rtp_receiver_manager] = lambda: manager
    client = TestClient(app)
    return client, manager, app


def test_autostart_disabled_by_default(monkeypatch):
    client, manager, app = _make_client(
        monkeypatch, autostart_env=None, enable_env=None
    )
    try:
        with client:
            assert manager.start_called == 0
        assert manager.shutdown_called == 0
    finally:
        app.dependency_overrides.clear()


def test_autostart_can_be_enabled(monkeypatch):
    client, manager, app = _make_client(
        monkeypatch, autostart_env="true", enable_env="true"
    )
    try:
        with client:
            assert manager.start_called == 1
        assert manager.shutdown_called == 1
    finally:
        app.dependency_overrides.clear()


def test_autostart_can_be_disabled_even_if_enabled(monkeypatch):
    client, manager, app = _make_client(
        monkeypatch, autostart_env="false", enable_env="true"
    )
    try:
        with client:
            assert manager.start_called == 0
        assert manager.shutdown_called == 1
    finally:
        app.dependency_overrides.clear()
