"""Tests for PipeWire ⇄ RTP input mode switching (Issue #379)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web import main as web_main
from web.services import config as config_service


def _write_config(path: Path, *, rtp_enabled: bool) -> None:
    """Write a minimal config file with the requested RTP flag."""
    config_data = {
        "alsaDevice": "hw:AUDIO",
        "upsampleRatio": 8,
        "eqEnabled": False,
        "crossfeed": {
            "enabled": False,
            "headSize": "m",
            "hrtfPath": "data/crossfeed/hrtf/",
        },
        "rtp": {
            "enabled": rtp_enabled,
            "autoStart": True,
            "sessionId": "pc_stream",
        },
    }
    path.write_text(json.dumps(config_data))


def _patch_config_paths(monkeypatch: pytest.MonkeyPatch, config_file: Path) -> None:
    """Ensure every module reads the temporary config file."""
    monkeypatch.setattr("web.constants.CONFIG_PATH", config_file)
    monkeypatch.setattr("web.services.config.CONFIG_PATH", config_file)
    monkeypatch.setattr("web.services.daemon.CONFIG_PATH", config_file)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Provide a TestClient with temporary config.json."""
    config_file = tmp_path / "config.json"
    _write_config(config_file, rtp_enabled=False)
    _patch_config_paths(monkeypatch, config_file)
    return TestClient(web_main.app, raise_server_exceptions=False)


def test_switch_to_rtp_updates_config_and_restarts(monkeypatch, tmp_path):
    """POST /api/input-mode/switch should toggle RTP mode and restart daemon."""
    config_file = tmp_path / "config.json"
    _write_config(config_file, rtp_enabled=False)
    _patch_config_paths(monkeypatch, config_file)

    restart_calls = {"stop": 0, "start": 0}

    monkeypatch.setattr("web.routers.input_mode.check_daemon_running", lambda: True)

    def fake_stop():
        restart_calls["stop"] += 1
        return True, "stopped"

    def fake_start():
        restart_calls["start"] += 1
        return True, "started"

    monkeypatch.setattr("web.routers.input_mode.stop_daemon", lambda: fake_stop())
    monkeypatch.setattr("web.routers.input_mode.start_daemon", lambda: fake_start())
    monkeypatch.setattr("web.routers.input_mode.time.sleep", lambda *_: None)

    client = TestClient(web_main.app, raise_server_exceptions=False)
    response = client.post("/api/input-mode/switch", json={"mode": "rtp"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["current_mode"] == "rtp"
    assert data["restart_required"] is True
    assert restart_calls == {"stop": 1, "start": 1}

    updated = json.loads(config_file.read_text())
    assert updated["rtp"]["enabled"] is True


def test_switch_noop_when_mode_unchanged(monkeypatch, tmp_path):
    """Switch endpoint should be a no-op when mode already matches."""
    config_file = tmp_path / "config.json"
    _write_config(config_file, rtp_enabled=True)
    _patch_config_paths(monkeypatch, config_file)

    monkeypatch.setattr("web.routers.input_mode.check_daemon_running", lambda: False)

    client = TestClient(web_main.app, raise_server_exceptions=False)
    response = client.post("/api/input-mode/switch", json={"mode": "rtp"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["restart_required"] is False
    assert data["current_mode"] == "rtp"

    updated = json.loads(config_file.read_text())
    assert updated["rtp"]["enabled"] is True


def test_status_endpoints_report_input_mode(client):
    """Both /status and /daemon/status should expose the input_mode field."""
    # Start with PipeWire mode
    config_file = Path(config_service.CONFIG_PATH)
    _write_config(config_file, rtp_enabled=False)

    status_resp = client.get("/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["input_mode"] == "pipewire"

    # Switch config to RTP and verify /daemon/status reflects it
    _write_config(config_file, rtp_enabled=True)
    daemon_resp = client.get("/daemon/status")
    assert daemon_resp.status_code == 200
    assert daemon_resp.json()["input_mode"] == "rtp"

