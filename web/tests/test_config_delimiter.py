import json

from web.services import config


def test_save_delimiter_enabled_creates_section(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)

    assert config.save_delimiter_enabled(False) is True

    saved = json.loads(cfg_path.read_text())
    assert saved["delimiter"]["enabled"] is False


def test_save_delimiter_enabled_preserves_existing_fields(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"alsaDevice": "hw:AUDIO"}))
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)

    assert config.save_delimiter_enabled(True) is True

    saved = json.loads(cfg_path.read_text())
    assert saved["alsaDevice"] == "hw:AUDIO"
    assert saved["delimiter"]["enabled"] is True
