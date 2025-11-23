"""Configuration loading and saving."""

import json

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


def save_config(settings: Settings) -> bool:
    """Save configuration to JSON file."""
    try:
        # Convert snake_case to camelCase for JSON
        data = {
            "alsaDevice": settings.alsa_device,
            "upsampleRatio": settings.upsample_ratio,
            "eqProfile": settings.eq_profile,
            "inputRate": settings.input_rate,
            "outputRate": settings.output_rate,
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except IOError:
        return False
