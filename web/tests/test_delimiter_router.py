from fastapi.testclient import TestClient

from web.main import create_app
from web.routers import delimiter as delimiter_router
from web.services.daemon_client import DaemonError, DaemonResponse


class _FakeDelimiterClient:
    def __init__(self, status: dict | None = None, error: DaemonError | None = None):
        self.status = status or {
            "enabled": False,
            "backend_available": True,
            "backend_valid": True,
            "mode": "active",
            "target_mode": "active",
            "fallback_reason": "none",
            "bypass_locked": False,
            "warmup": False,
            "queue_seconds": 0.0,
            "queue_samples": 0,
            "last_inference_ms": 0.0,
            "detail": "",
        }
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def _error_response(self) -> DaemonResponse:
        return DaemonResponse(success=False, error=self.error)

    def delimiter_status(self) -> DaemonResponse:
        if self.error:
            return self._error_response()
        return DaemonResponse(success=True, data=self.status)

    def delimiter_enable(self) -> DaemonResponse:
        if self.error:
            return self._error_response()
        enabled = dict(self.status)
        enabled["enabled"] = True
        self.status = enabled
        return DaemonResponse(success=True, data=enabled)

    def delimiter_disable(self) -> DaemonResponse:
        if self.error:
            return self._error_response()
        disabled = dict(self.status)
        disabled["enabled"] = False
        self.status = disabled
        return DaemonResponse(success=True, data=disabled)


def test_delimiter_status_success(monkeypatch):
    sample_status = {
        "enabled": False,
        "backend_available": True,
        "backend_valid": True,
        "mode": "active",
        "target_mode": "active",
        "fallback_reason": "none",
        "bypass_locked": False,
        "warmup": True,
        "queue_seconds": 1.23,
        "queue_samples": 512,
        "last_inference_ms": 15.6,
        "detail": "ok",
    }
    fake_client = _FakeDelimiterClient(status=sample_status)
    monkeypatch.setattr(delimiter_router, "get_daemon_client", lambda: fake_client)

    app = create_app()
    client = TestClient(app)

    resp = client.get("/delimiter/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["backend_available"] is True
    assert body["warmup"] is True
    assert body["queue_samples"] == 512
    assert body["detail"] == "ok"


def test_delimiter_enable(monkeypatch):
    fake_client = _FakeDelimiterClient()
    monkeypatch.setattr(delimiter_router, "get_daemon_client", lambda: fake_client)
    monkeypatch.setattr(
        delimiter_router, "save_delimiter_enabled", lambda enabled: True
    )

    app = create_app()
    client = TestClient(app)

    resp = client.post("/delimiter/enable")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["status"]["enabled"] is True


def test_delimiter_disable(monkeypatch):
    fake_client = _FakeDelimiterClient(status={"enabled": True})
    monkeypatch.setattr(delimiter_router, "get_daemon_client", lambda: fake_client)
    monkeypatch.setattr(
        delimiter_router, "save_delimiter_enabled", lambda enabled: True
    )

    app = create_app()
    client = TestClient(app)

    resp = client.post("/delimiter/disable")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["status"]["enabled"] is False


def test_delimiter_daemon_down(monkeypatch):
    error = DaemonError(
        error_code="IPC_DAEMON_NOT_RUNNING",
        message="daemon not running",
        inner_error={"zmq_errno": 61},
    )
    fake_client = _FakeDelimiterClient(error=error)
    monkeypatch.setattr(delimiter_router, "get_daemon_client", lambda: fake_client)

    app = create_app()
    client = TestClient(app)

    resp = client.get("/delimiter/status")
    assert resp.status_code == 503
    body = resp.json()
    assert body["error_code"] == "IPC_DAEMON_NOT_RUNNING"


def test_delimiter_persist_failure(monkeypatch):
    fake_client = _FakeDelimiterClient()
    monkeypatch.setattr(delimiter_router, "get_daemon_client", lambda: fake_client)
    monkeypatch.setattr(
        delimiter_router, "save_delimiter_enabled", lambda enabled: False
    )

    app = create_app()
    client = TestClient(app)

    resp = client.post("/delimiter/enable")
    assert resp.status_code == 500
    body = resp.json()
    assert "persist" in body["detail"].lower()
