"""Tests for output mode REST endpoints."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web import main as web_main


def _write_config(path: Path, device: str = "hw:USB") -> None:
    """Write minimal config with legacy alsaDevice."""
    config_data = {
        "alsaDevice": device,
        "upsampleRatio": 8,
    }
    path.write_text(json.dumps(config_data))


class TestOutputModeApi:
    """Test suite for /api/output/mode endpoints."""

    def setup_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        config_file = tmp_path / "config.json"
        _write_config(config_file)
        monkeypatch.setattr("web.constants.CONFIG_PATH", config_file)
        monkeypatch.setattr("web.services.config.CONFIG_PATH", config_file)
        return config_file

    def test_get_output_mode_falls_back_to_config(self, tmp_path: Path, monkeypatch):
        """GET should return config state when daemon is unavailable."""
        self.setup_paths(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "web.routers.output_mode._fetch_runtime_output_mode", lambda: None
        )
        client = TestClient(web_main.app, raise_server_exceptions=False)

        response = client.get("/api/output/mode")
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "usb"
        assert data["options"]["usb"]["preferred_device"] == "hw:USB"

    def test_post_output_mode_updates_config(self, tmp_path: Path, monkeypatch):
        """POST should persist config and call runtime helper."""
        config_file = self.setup_paths(tmp_path, monkeypatch)

        def fake_apply(payload):
            return (
                {
                    "mode": payload["mode"],
                    "available_modes": ["usb"],
                    "options": {
                        "usb": {
                            "preferred_device": payload["options"]["usb"][
                                "preferred_device"
                            ]
                        }
                    },
                },
                False,
            )

        monkeypatch.setattr(
            "web.routers.output_mode._apply_runtime_output_mode", fake_apply
        )

        client = TestClient(web_main.app, raise_server_exceptions=False)
        response = client.post(
            "/api/output/mode",
            json={
                "mode": "usb",
                "options": {"usb": {"preferred_device": "hw:USB2"}},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert (
            body["data"]["output_mode"]["options"]["usb"]["preferred_device"]
            == "hw:USB2"
        )

        saved_config = json.loads(config_file.read_text())
        assert saved_config["output"]["options"]["usb"]["preferredDevice"] == "hw:USB2"

    def test_post_output_mode_rejects_invalid_device(self, tmp_path: Path, monkeypatch):
        """Invalid ALSA device should return HTTP 400."""
        self.setup_paths(tmp_path, monkeypatch)
        client = TestClient(web_main.app, raise_server_exceptions=False)
        response = client.post(
            "/api/output/mode",
            json={
                "mode": "usb",
                "options": {"usb": {"preferred_device": "../etc/passwd"}},
            },
        )
        assert response.status_code == 400
