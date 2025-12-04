"""Tests for web/services/config.py."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch


from web.services.config import (
    load_config,
    load_output_mode,
    load_raw_config,
    save_config,
    save_output_mode,
)
from web.models import CrossfeedSettings, Settings


class TestLoadRawConfig:
    """Tests for load_raw_config function."""

    def test_load_existing_config(self, tmp_path: Path) -> None:
        """Test loading an existing config file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "alsaDevice": "hw:AUDIO",
            "upsampleRatio": 16,
            "filterPath44kMin": "path/to/filter.bin",
        }
        config_file.write_text(json.dumps(config_data))

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = load_raw_config()

        assert result == config_data

    def test_load_nonexistent_config(self, tmp_path: Path) -> None:
        """Test loading when config file doesn't exist."""
        config_file = tmp_path / "nonexistent.json"

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = load_raw_config()

        assert result == {}

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test loading when config file contains invalid JSON."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{ invalid json }")

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = load_raw_config()

        assert result == {}

    def test_load_non_dict_json_array(self, tmp_path: Path) -> None:
        """Test loading when config file contains a JSON array (not dict)."""
        config_file = tmp_path / "config.json"
        config_file.write_text('["item1", "item2"]')

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = load_raw_config()

        assert result == {}

    def test_load_non_dict_json_string(self, tmp_path: Path) -> None:
        """Test loading when config file contains a JSON string (not dict)."""
        config_file = tmp_path / "config.json"
        config_file.write_text('"just a string"')

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = load_raw_config()

        assert result == {}


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_preserves_existing_fields(self, tmp_path: Path) -> None:
        """Test that save_config preserves fields not in Settings."""
        config_file = tmp_path / "config.json"
        # Existing config with per-filter paths and other fields
        existing_config: dict[str, Any] = {
            "alsaDevice": "hw:OLD",
            "upsampleRatio": 8,
            "filterPath44kMin": "data/coefficients/filter_44k_16x_2m_linear_phase.bin",
            "filterPath48kMin": "data/coefficients/filter_48k_16x_2m_linear_phase.bin",
            "filterPath44kLinear": "data/coefficients/filter_44k_16x_2m_linear_phase.bin",
            "filterPath48kLinear": "data/coefficients/filter_48k_16x_2m_linear_phase.bin",
            "phaseType": "minimum",
            "eqEnabled": True,
            "eqProfilePath": "/path/to/eq.txt",
        }
        config_file.write_text(json.dumps(existing_config))

        # New settings (only a subset of fields)
        new_settings = Settings(
            alsa_device="hw:NEW",
            upsample_ratio=16,
            eq_enabled=True,
            eq_profile="NewProfile",
            eq_profile_path="/path/to/new_eq.txt",
        )

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = save_config(new_settings)

        assert result is True

        # Read back and verify
        saved_config = json.loads(config_file.read_text())

        # Settings fields should be updated
        assert saved_config["alsaDevice"] == "hw:NEW"
        assert saved_config["upsampleRatio"] == 16
        assert saved_config["eqProfile"] == "NewProfile"

        # Non-Settings fields should be preserved
        assert (
            saved_config["filterPath44kMin"]
            == "data/coefficients/filter_44k_16x_2m_linear_phase.bin"
        )
        assert (
            saved_config["filterPath48kMin"]
            == "data/coefficients/filter_48k_16x_2m_linear_phase.bin"
        )
        assert (
            saved_config["filterPath44kLinear"]
            == "data/coefficients/filter_44k_16x_2m_linear_phase.bin"
        )
        assert (
            saved_config["filterPath48kLinear"]
            == "data/coefficients/filter_48k_16x_2m_linear_phase.bin"
        )
        assert saved_config["phaseType"] == "minimum"
        assert saved_config["eqEnabled"] is True
        assert saved_config["eqProfilePath"] == "/path/to/new_eq.txt"
        assert saved_config["output"]["mode"] == "usb"
        assert (
            saved_config["output"]["options"]["usb"]["preferredDevice"] == "hw:NEW"
        )

        # Auto-negotiated fields should be removed
        assert "inputRate" not in saved_config
        assert "outputRate" not in saved_config
        assert "inputSampleRate" not in saved_config

    def test_save_creates_new_file(self, tmp_path: Path) -> None:
        """Test that save_config creates a new file if it doesn't exist."""
        config_file = tmp_path / "config.json"
        assert not config_file.exists()

        new_settings = Settings(
            alsa_device="hw:USB",
            upsample_ratio=8,
        )

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = save_config(new_settings)

        assert result is True
        assert config_file.exists()

        saved_config = json.loads(config_file.read_text())
        assert saved_config["alsaDevice"] == "hw:USB"
        assert saved_config["upsampleRatio"] == 8
        assert saved_config["output"]["options"]["usb"]["preferredDevice"] == "hw:USB"

    def test_save_with_none_eq_profile(self, tmp_path: Path) -> None:
        """Test that save_config handles None eq_profile."""
        config_file = tmp_path / "config.json"
        existing_config: dict[str, Any] = {}
        config_file.write_text(json.dumps(existing_config))

        new_settings = Settings(
            alsa_device="hw:USB",
            upsample_ratio=8,
            eq_profile=None,
        )

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = save_config(new_settings)

        assert result is True

        saved_config = json.loads(config_file.read_text())
        assert saved_config["eqProfile"] is None


class TestOutputModeConfig:
    """Tests for load_output_mode/save_output_mode helpers."""

    def test_load_output_mode_defaults(self, tmp_path: Path, monkeypatch) -> None:
        """When output section missing, defaults should be returned."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        monkeypatch.setattr("web.services.config.CONFIG_PATH", config_file)

        state = load_output_mode()
        assert state["mode"] == "usb"
        assert state["available_modes"] == ["usb"]
        assert state["options"]["usb"]["preferred_device"] == "hw:USB"

    def test_save_output_mode_updates_config(self, tmp_path: Path, monkeypatch) -> None:
        """save_output_mode should persist output + legacy alsa fields."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"alsaDevice": "hw:OLD"}))
        monkeypatch.setattr("web.services.config.CONFIG_PATH", config_file)

        assert save_output_mode("usb", "hw:USB2")

        saved_config = json.loads(config_file.read_text())
        assert saved_config["alsaDevice"] == "hw:USB2"
        assert saved_config["output"]["mode"] == "usb"
        assert saved_config["output"]["options"]["usb"]["preferredDevice"] == "hw:USB2"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_returns_settings(self, tmp_path: Path) -> None:
        """Test that load_config returns a Settings object."""
        config_file = tmp_path / "config.json"
        config_data = {
            "alsaDevice": "hw:AUDIO",
            "upsampleRatio": 16,
            "eqProfile": "MyProfile",
        }
        config_file.write_text(json.dumps(config_data))

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = load_config()

        assert isinstance(result, Settings)
        assert result.alsa_device == "hw:AUDIO"
        assert result.upsample_ratio == 16
        assert result.eq_profile == "MyProfile"

    def test_load_config_uses_defaults(self, tmp_path: Path) -> None:
        """Test that load_config uses defaults when file doesn't exist."""
        config_file = tmp_path / "nonexistent.json"

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = load_config()

        assert isinstance(result, Settings)
        # Check defaults from Settings model
        assert result.alsa_device == "default"
        assert result.upsample_ratio == 8


class TestCrossfeedConfig:
    """Tests for crossfeed configuration."""

    def test_load_config_crossfeed_defaults(self, tmp_path: Path) -> None:
        """Test default crossfeed settings when not specified."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        with patch("web.services.config.CONFIG_PATH", config_file):
            settings = load_config()

        assert settings.crossfeed.enabled is False
        assert settings.crossfeed.head_size == "m"
        assert settings.crossfeed.hrtf_path == "data/crossfeed/hrtf/"

    def test_load_config_crossfeed_enabled(self, tmp_path: Path) -> None:
        """Test loading crossfeed settings."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "crossfeed": {
                        "enabled": True,
                        "headSize": "l",
                        "hrtfPath": "custom/hrtf/",
                    }
                }
            )
        )

        with patch("web.services.config.CONFIG_PATH", config_file):
            settings = load_config()

        assert settings.crossfeed.enabled is True
        assert settings.crossfeed.head_size == "l"
        assert settings.crossfeed.hrtf_path == "custom/hrtf/"

    def test_load_config_crossfeed_partial(self, tmp_path: Path) -> None:
        """Test loading partial crossfeed settings (keeps defaults for missing)."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"crossfeed": {"enabled": True}}))

        with patch("web.services.config.CONFIG_PATH", config_file):
            settings = load_config()

        assert settings.crossfeed.enabled is True
        # Missing fields should use defaults
        assert settings.crossfeed.head_size == "m"
        assert settings.crossfeed.hrtf_path == "data/crossfeed/hrtf/"

    def test_save_config_preserves_crossfeed(self, tmp_path: Path) -> None:
        """Test saving crossfeed settings."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        settings = Settings(
            crossfeed=CrossfeedSettings(
                enabled=True,
                head_size="xl",
                hrtf_path="my/hrtf/",
            )
        )

        with patch("web.services.config.CONFIG_PATH", config_file):
            save_config(settings)

        saved = json.loads(config_file.read_text())
        assert saved["crossfeed"]["enabled"] is True
        assert saved["crossfeed"]["headSize"] == "xl"
        assert saved["crossfeed"]["hrtfPath"] == "my/hrtf/"

    def test_save_config_crossfeed_default_values(self, tmp_path: Path) -> None:
        """Test saving default crossfeed settings."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        settings = Settings()  # All defaults

        with patch("web.services.config.CONFIG_PATH", config_file):
            save_config(settings)

        saved = json.loads(config_file.read_text())
        assert saved["crossfeed"]["enabled"] is False
        assert saved["crossfeed"]["headSize"] == "m"
        assert saved["crossfeed"]["hrtfPath"] == "data/crossfeed/hrtf/"

    def test_roundtrip_crossfeed_settings(self, tmp_path: Path) -> None:
        """Test saving and loading crossfeed settings (roundtrip)."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        original_settings = Settings(
            alsa_device="hw:TEST",
            crossfeed=CrossfeedSettings(
                enabled=True,
                head_size="s",
                hrtf_path="roundtrip/hrtf/",
            ),
        )

        with patch("web.services.config.CONFIG_PATH", config_file):
            save_config(original_settings)
            loaded_settings = load_config()

        assert loaded_settings.crossfeed.enabled is True
        assert loaded_settings.crossfeed.head_size == "s"
        assert loaded_settings.crossfeed.hrtf_path == "roundtrip/hrtf/"
        assert loaded_settings.alsa_device == "hw:TEST"

    def test_load_config_invalid_head_size_returns_defaults(
        self, tmp_path: Path
    ) -> None:
        """Test that invalid head_size falls back to defaults."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "crossfeed": {
                        "enabled": True,
                        "headSize": "invalid_size",  # Invalid value
                        "hrtfPath": "some/path/",
                    }
                }
            )
        )

        with patch("web.services.config.CONFIG_PATH", config_file):
            settings = load_config()

        # Should return default Settings due to validation error
        assert settings.crossfeed.enabled is False
        assert settings.crossfeed.head_size == "m"
        assert settings.crossfeed.hrtf_path == "data/crossfeed/hrtf/"

    def test_load_config_crossfeed_not_object_returns_defaults(
        self, tmp_path: Path
    ) -> None:
        """Test that crossfeed as non-object returns defaults."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"crossfeed": "not an object"}))

        with patch("web.services.config.CONFIG_PATH", config_file):
            settings = load_config()

        # Should use default crossfeed settings
        assert settings.crossfeed.enabled is False
        assert settings.crossfeed.head_size == "m"
