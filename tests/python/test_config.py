"""Tests for web/services/config.py."""

import json
from pathlib import Path
from unittest.mock import patch


from web.services.config import load_config, load_raw_config, save_config
from web.models import Settings


class TestLoadRawConfig:
    """Tests for load_raw_config function."""

    def test_load_existing_config(self, tmp_path: Path) -> None:
        """Test loading an existing config file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "alsaDevice": "hw:AUDIO",
            "upsampleRatio": 16,
            "quadPhaseEnabled": True,
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


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_preserves_existing_fields(self, tmp_path: Path) -> None:
        """Test that save_config preserves fields not in Settings."""
        config_file = tmp_path / "config.json"
        # Existing config with quadPhaseEnabled and other fields
        existing_config = {
            "alsaDevice": "hw:OLD",
            "upsampleRatio": 8,
            "quadPhaseEnabled": True,
            "filterPath44kMin": "data/coefficients/filter_44k_2m_min_phase.bin",
            "filterPath48kMin": "data/coefficients/filter_48k_2m_min_phase.bin",
            "filterPath44kLinear": "data/coefficients/filter_44k_16x_2m_linear.bin",
            "filterPath48kLinear": "data/coefficients/filter_48k_16x_2m_linear.bin",
            "phaseType": "minimum",
            "inputSampleRate": 44100,
            "eqEnabled": True,
            "eqProfilePath": "/path/to/eq.txt",
        }
        config_file.write_text(json.dumps(existing_config))

        # New settings (only a subset of fields)
        new_settings = Settings(
            alsa_device="hw:NEW",
            upsample_ratio=16,
            eq_profile="NewProfile",
            input_rate=48000,
            output_rate=768000,
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
        assert saved_config["inputRate"] == 48000
        assert saved_config["outputRate"] == 768000

        # Non-Settings fields should be preserved
        assert saved_config["quadPhaseEnabled"] is True
        assert (
            saved_config["filterPath44kMin"]
            == "data/coefficients/filter_44k_2m_min_phase.bin"
        )
        assert (
            saved_config["filterPath48kMin"]
            == "data/coefficients/filter_48k_2m_min_phase.bin"
        )
        assert (
            saved_config["filterPath44kLinear"]
            == "data/coefficients/filter_44k_16x_2m_linear.bin"
        )
        assert (
            saved_config["filterPath48kLinear"]
            == "data/coefficients/filter_48k_16x_2m_linear.bin"
        )
        assert saved_config["phaseType"] == "minimum"
        assert saved_config["inputSampleRate"] == 44100
        assert saved_config["eqEnabled"] is True
        assert saved_config["eqProfilePath"] == "/path/to/eq.txt"

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

    def test_save_with_none_eq_profile(self, tmp_path: Path) -> None:
        """Test that save_config handles None eq_profile."""
        config_file = tmp_path / "config.json"
        existing_config = {"quadPhaseEnabled": True}
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
        assert saved_config["quadPhaseEnabled"] is True


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_returns_settings(self, tmp_path: Path) -> None:
        """Test that load_config returns a Settings object."""
        config_file = tmp_path / "config.json"
        config_data = {
            "alsaDevice": "hw:AUDIO",
            "upsampleRatio": 16,
            "eqProfile": "MyProfile",
            "inputRate": 44100,
            "outputRate": 705600,
        }
        config_file.write_text(json.dumps(config_data))

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = load_config()

        assert isinstance(result, Settings)
        assert result.alsa_device == "hw:AUDIO"
        assert result.upsample_ratio == 16
        assert result.eq_profile == "MyProfile"
        assert result.input_rate == 44100
        assert result.output_rate == 705600

    def test_load_config_uses_defaults(self, tmp_path: Path) -> None:
        """Test that load_config uses defaults when file doesn't exist."""
        config_file = tmp_path / "nonexistent.json"

        with patch("web.services.config.CONFIG_PATH", config_file):
            result = load_config()

        assert isinstance(result, Settings)
        # Check defaults from Settings model
        assert result.alsa_device == "default"
        assert result.upsample_ratio == 8
