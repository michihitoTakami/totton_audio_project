"""Tests for docker/jetson/entrypoint.sh config initialization."""

import json
import os
import subprocess
from pathlib import Path

ENTRYPOINT = Path(__file__).parent.parent.parent / "docker" / "jetson" / "entrypoint.sh"


def _run_entrypoint(
    tmp_path: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess:
    """Execute entrypoint with custom environment and return result."""
    merged_env = os.environ.copy()
    merged_env.update(env)
    return subprocess.run(
        ["bash", str(ENTRYPOINT), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        env=merged_env,
    )


def test_config_init_seeds_default_when_missing(tmp_path: Path) -> None:
    """Config-init should copy the default config when none exists."""
    default_file = tmp_path / "default" / "config.json"
    default_file.parent.mkdir(parents=True)
    default_payload = {"alsaDevice": "hw:AUDIO", "eqEnabled": False}
    default_file.write_text(json.dumps(default_payload))

    config_dir = tmp_path / "config"
    symlink_path = tmp_path / "config-link.json"

    result = _run_entrypoint(
        tmp_path,
        {
            "MAGICBOX_CONFIG_DIR": str(config_dir),
            "MAGICBOX_CONFIG_SYMLINK": str(symlink_path),
            "MAGICBOX_DEFAULT_CONFIG": str(default_file),
        },
        "config-init",
    )

    assert result.returncode == 0, result.stderr
    config_data = json.loads((config_dir / "config.json").read_text())
    assert config_data == default_payload
    assert symlink_path.exists()
    assert Path(os.path.realpath(symlink_path)) == config_dir / "config.json"


def test_config_init_preserves_existing_without_reset(tmp_path: Path) -> None:
    """Existing config should be kept unless reset is requested."""
    default_file = tmp_path / "default" / "config.json"
    default_file.parent.mkdir(parents=True)
    default_file.write_text(json.dumps({"default": True}))

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({"custom": "keep"}))

    result = _run_entrypoint(
        tmp_path,
        {
            "MAGICBOX_CONFIG_DIR": str(config_dir),
            "MAGICBOX_CONFIG_SYMLINK": str(tmp_path / "config-link.json"),
            "MAGICBOX_DEFAULT_CONFIG": str(default_file),
        },
        "config-init",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(config_file.read_text()) == {"custom": "keep"}


def test_config_init_resets_when_flag_true(tmp_path: Path) -> None:
    """Reset flag should overwrite with default config."""
    default_file = tmp_path / "default" / "config.json"
    default_file.parent.mkdir(parents=True)
    default_payload = {"reset": "applied"}
    default_file.write_text(json.dumps(default_payload))

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({"custom": "old"}))

    result = _run_entrypoint(
        tmp_path,
        {
            "MAGICBOX_CONFIG_DIR": str(config_dir),
            "MAGICBOX_CONFIG_SYMLINK": str(tmp_path / "config-link.json"),
            "MAGICBOX_DEFAULT_CONFIG": str(default_file),
            "MAGICBOX_RESET_CONFIG": "true",
        },
        "config-init",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(config_file.read_text()) == default_payload


def test_config_init_recovers_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON should be backed up and default restored."""
    default_file = tmp_path / "default" / "config.json"
    default_file.parent.mkdir(parents=True)
    default_payload = {"safe": True}
    default_file.write_text(json.dumps(default_payload))

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.json"
    config_file.write_text("{ invalid json }")

    result = _run_entrypoint(
        tmp_path,
        {
            "MAGICBOX_CONFIG_DIR": str(config_dir),
            "MAGICBOX_CONFIG_SYMLINK": str(tmp_path / "config-link.json"),
            "MAGICBOX_DEFAULT_CONFIG": str(default_file),
        },
        "config-init",
    )

    assert result.returncode == 0, result.stderr
    # Default should be restored
    assert json.loads(config_file.read_text()) == default_payload
    # Backup should exist with original broken content
    backup = config_file.with_suffix(".json.bak")
    assert backup.exists()
    assert "{ invalid json }" in backup.read_text()
