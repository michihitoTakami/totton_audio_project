"""Configuration loading and saving."""

import json
from typing import Any

from ..constants import CONFIG_PATH
from ..models import Settings


def load_config() -> Settings:
    """Load configuration from JSON file."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            # Convert camelCase to snake_case
            return Settings(
                alsa_device=data.get("alsaDevice", "default"),
                upsample_ratio=data.get("upsampleRatio", 8),
                eq_profile=data.get("eqProfile"),
                input_rate=data.get("inputRate", 44100),
                output_rate=data.get("outputRate", 352800),
            )
        except (json.JSONDecodeError, KeyError):
            pass
    return Settings()


def load_raw_config() -> dict[str, Any]:
    """Load raw config.json as dictionary, preserving all fields."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_config(settings: Settings) -> bool:
    """Save configuration to JSON file, preserving existing fields.

    This function merges the Settings fields into the existing config.json,
    preserving any fields not managed by Settings (e.g., quadPhaseEnabled,
    filterPath*, etc.).
    """
    try:
        # Load existing config to preserve unmanaged fields
        existing = load_raw_config()

        # Update only the fields managed by Settings
        existing["alsaDevice"] = settings.alsa_device
        existing["upsampleRatio"] = settings.upsample_ratio
        existing["eqProfile"] = settings.eq_profile
        existing["inputRate"] = settings.input_rate
        existing["outputRate"] = settings.output_rate

        with open(CONFIG_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        return True
    except IOError:
        return False
