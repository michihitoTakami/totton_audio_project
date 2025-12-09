"""Tests for config service TCP helpers."""

import json

import pytest

from web.services import config
from web.models import TcpInputSettings


@pytest.fixture
def temp_config_path(tmp_path, monkeypatch):
    """Use a temporary config.json for each test."""
    path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    return path


def test_get_tcp_config_defaults_when_missing(temp_config_path):
    """Missing section should return TcpInputSettings defaults."""
    temp_config_path.write_text(json.dumps({"other": "value"}))

    settings = config.get_tcp_config()

    assert isinstance(settings, TcpInputSettings)
    assert settings.enabled is True
    assert settings.bind_address == "0.0.0.0"
    assert settings.port == 46001
    assert settings.buffer_size_bytes == 262144


def test_get_tcp_config_parses_values(temp_config_path):
    """Existing tcpInput section should be parsed with aliases."""
    temp_config_path.write_text(
        json.dumps(
            {
                "tcpInput": {
                    "enabled": False,
                    "bindAddress": "127.0.0.1",
                    "port": 12345,
                    "bufferSizeBytes": 8192,
                    "connection_mode": "priority",
                }
            }
        )
    )

    settings = config.get_tcp_config()

    assert settings.enabled is False
    assert settings.bind_address == "127.0.0.1"
    assert settings.port == 12345
    assert settings.buffer_size_bytes == 8192
    assert settings.connection_mode == "priority"


def test_update_tcp_config_merges_and_persists(temp_config_path):
    """Updates should merge with existing tcpInput section and persist."""
    temp_config_path.write_text(
        json.dumps(
            {
                "tcpInput": {
                    "enabled": False,
                    "bindAddress": "0.0.0.0",
                    "port": 46001,
                    "bufferSizeBytes": 128000,
                },
                "keep": {"untouched": True},
            }
        )
    )

    result = config.update_tcp_config({"port": 47000, "bind_address": "10.0.0.1"})

    assert result is True
    saved = json.loads(temp_config_path.read_text())
    tcp_section = saved.get("tcpInput", {})
    assert tcp_section["port"] == 47000
    assert tcp_section["bindAddress"] == "10.0.0.1"
    # Existing values should be preserved when not overwritten
    assert tcp_section["bufferSizeBytes"] == 128000
    # Unrelated keys remain untouched
    assert saved["keep"] == {"untouched": True}


def test_update_tcp_config_rejects_empty_payload(temp_config_path):
    """Empty updates should be rejected for safety."""
    temp_config_path.write_text(json.dumps({}))

    assert config.update_tcp_config({}) is False
