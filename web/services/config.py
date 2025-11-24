"""Configuration loading and saving."""

import json
from pathlib import Path

from ..constants import CONFIG_PATH, EQ_PROFILES_DIR
from ..models import Settings


def _build_profile_path(profile_name: str | None) -> str | None:
    """Return full path for the given EQ profile name, or None."""
    if not profile_name:
        return None
    return str(EQ_PROFILES_DIR / f"{profile_name}.txt")


def load_config() -> Settings:
    """Load configuration from JSON file."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            # Convert camelCase to snake_case
            eq_profile = data.get("eqProfile")
            eq_profile_path = data.get("eqProfilePath")
            eq_enabled = data.get("eqEnabled")

            # Migration / normalization
            if eq_profile_path is None:
                if eq_enabled is None and eq_profile:
                    # Old style: only eqProfile present
                    eq_profile_path = _build_profile_path(eq_profile)
                else:
                    # Explicitly enabled but missing path -> treat as disabled
                    eq_enabled = False

            if eq_enabled is None:
                eq_enabled = bool(eq_profile_path)

            if eq_profile is None and eq_profile_path:
                eq_profile = Path(eq_profile_path).stem

            return Settings(
                alsa_device=data.get("alsaDevice", "default"),
                upsample_ratio=data.get("upsampleRatio", 8),
                eq_enabled=bool(eq_enabled and eq_profile_path),
                eq_profile=eq_profile,
                eq_profile_path=eq_profile_path,
                input_rate=data.get("inputRate", 44100),
                output_rate=data.get("outputRate", 352800),
            )
        except (json.JSONDecodeError, KeyError):
            pass
    return Settings()


def save_config(settings: Settings) -> bool:
    """Save configuration to JSON file."""
    try:
        eq_profile_path = settings.eq_profile_path or _build_profile_path(
            settings.eq_profile
        )
        eq_enabled = settings.eq_enabled and bool(eq_profile_path)

        # Convert snake_case to camelCase for JSON
        data = {
            "alsaDevice": settings.alsa_device,
            "upsampleRatio": settings.upsample_ratio,
            "eqEnabled": eq_enabled,
            "eqProfile": settings.eq_profile if eq_enabled else None,
            "eqProfilePath": eq_profile_path if eq_enabled else None,
            "inputRate": settings.input_rate,
            "outputRate": settings.output_rate,
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except IOError:
        return False
