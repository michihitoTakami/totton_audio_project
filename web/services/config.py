"""Configuration loading and saving."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..constants import (
    CONFIG_PATH,
    EQ_PROFILES_DIR,
    PHASE_TYPE_MINIMUM,
)
from ..models import (
    CrossfeedSettings,
    InputMode,
    PartitionedConvolutionSettings,
    Settings,
)


def _build_profile_path(profile_name: str | None) -> str | None:
    """Return full path for the given EQ profile name, or None."""
    if not profile_name:
        return None
    return str(EQ_PROFILES_DIR / f"{profile_name}.txt")


def _resolve_input_mode(config_data: dict[str, Any]) -> InputMode:
    """Return input mode string based on RTP section."""
    rtp_section = config_data.get("rtp", {})
    enabled = False
    if isinstance(rtp_section, dict):
        enabled = bool(rtp_section.get("enabled"))
    return "rtp" if enabled else "pipewire"


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

            # Crossfeed settings (ensure crossfeed is a dict)
            crossfeed_data = data.get("crossfeed", {})
            if not isinstance(crossfeed_data, dict):
                crossfeed_data = {}
            crossfeed = CrossfeedSettings(
                enabled=crossfeed_data.get("enabled", False),
                head_size=crossfeed_data.get("headSize", "m"),
                hrtf_path=crossfeed_data.get("hrtfPath", "data/crossfeed/hrtf/"),
            )

            input_mode = _resolve_input_mode(data)

            return Settings(
                alsa_device=data.get("alsaDevice", "default"),
                upsample_ratio=data.get("upsampleRatio", 8),
                eq_enabled=bool(eq_enabled and eq_profile_path),
                eq_profile=eq_profile,
                eq_profile_path=eq_profile_path,
                input_rate=data.get("inputRate", 44100),
                output_rate=data.get("outputRate", 352800),
                crossfeed=crossfeed,
                rtp_enabled=input_mode == "rtp",
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # ValueError catches Pydantic validation errors (e.g., invalid head_size)
            pass
    return Settings()


def load_raw_config() -> dict[str, Any]:
    """Load raw config.json as dictionary, preserving all fields.

    Returns an empty dict if the file doesn't exist, is invalid JSON,
    or contains non-dict JSON (e.g., array or string).
    """
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            # Guard: ensure we got a dict, not array/string/etc
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def load_partitioned_convolution_settings() -> PartitionedConvolutionSettings:
    """Load partitioned convolution settings from config.json."""
    defaults = PartitionedConvolutionSettings()
    raw = load_raw_config()
    section = raw.get("partitionedConvolution", {})
    if not isinstance(section, dict):
        section = {}
    try:
        return PartitionedConvolutionSettings(
            enabled=section.get("enabled", defaults.enabled),
            fast_partition_taps=section.get(
                "fastPartitionTaps", defaults.fast_partition_taps
            ),
            min_partition_taps=section.get(
                "minPartitionTaps", defaults.min_partition_taps
            ),
            max_partitions=section.get("maxPartitions", defaults.max_partitions),
            tail_fft_multiple=section.get(
                "tailFftMultiple", defaults.tail_fft_multiple
            ),
        )
    except (ValidationError, ValueError, TypeError):
        return defaults


def save_partitioned_convolution_settings(
    settings: PartitionedConvolutionSettings,
) -> bool:
    """Persist partitioned convolution settings to config.json."""
    try:
        existing = load_raw_config()
        existing["partitionedConvolution"] = {
            "enabled": settings.enabled,
            "fastPartitionTaps": settings.fast_partition_taps,
            "minPartitionTaps": settings.min_partition_taps,
            "maxPartitions": settings.max_partitions,
            "tailFftMultiple": settings.tail_fft_multiple,
        }
        if settings.enabled:
            phase_type = str(existing.get("phaseType", "minimum")).lower()
            if phase_type != "minimum":
                existing["phaseType"] = "minimum"
        with open(CONFIG_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        return True
    except IOError:
        return False


def save_phase_type(phase_type: str) -> bool:
    """Persist phaseType field in config.json."""
    phase = str(phase_type).lower()
    normalized = phase if phase in ["minimum", "linear"] else PHASE_TYPE_MINIMUM
    try:
        existing = load_raw_config()
        existing["phaseType"] = normalized
        with open(CONFIG_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        return True
    except IOError:
        return False


def save_config(settings: Settings) -> bool:
    """Save configuration to JSON file, preserving existing fields.

    This function merges the Settings fields into the existing config.json,
    preserving any fields not managed by Settings (e.g., filterPath*, etc.).
    """
    try:
        # Load existing config to preserve unmanaged fields
        existing = load_raw_config()

        eq_profile_path = settings.eq_profile_path or _build_profile_path(
            settings.eq_profile
        )
        eq_enabled = settings.eq_enabled and bool(eq_profile_path)

        # Update only the fields managed by Settings
        existing["alsaDevice"] = settings.alsa_device
        existing["upsampleRatio"] = settings.upsample_ratio
        existing["eqEnabled"] = eq_enabled
        existing["eqProfile"] = settings.eq_profile if eq_enabled else None
        existing["eqProfilePath"] = eq_profile_path if eq_enabled else None

        # Remove deprecated fields if present (inputRate/outputRate are auto-negotiated)
        existing.pop("inputRate", None)
        existing.pop("outputRate", None)
        existing.pop("inputSampleRate", None)

        # Crossfeed settings (camelCase for JSON)
        existing["crossfeed"] = {
            "enabled": settings.crossfeed.enabled,
            "headSize": settings.crossfeed.head_size,
            "hrtfPath": settings.crossfeed.hrtf_path,
        }
        # Preserve RTP settings but update enabled flag when provided
        if settings.rtp_enabled:
            rtp_section = existing.get("rtp", {})
            if not isinstance(rtp_section, dict):
                rtp_section = {}
            rtp_section["enabled"] = True
            existing["rtp"] = rtp_section

        with open(CONFIG_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        return True
    except IOError:
        return False


def get_input_mode() -> InputMode:
    """Read current input mode from config.json."""
    raw = load_raw_config()
    return _resolve_input_mode(raw)


def save_input_mode(mode: InputMode) -> bool:
    """Persist input mode (PipeWire or RTP) to config.json."""
    normalized: InputMode = "rtp" if mode == "rtp" else "pipewire"
    try:
        existing = load_raw_config()
        rtp_section = existing.get("rtp", {})
        if not isinstance(rtp_section, dict):
            rtp_section = {}
        rtp_section["enabled"] = normalized == "rtp"
        existing["rtp"] = rtp_section
        with open(CONFIG_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        return True
    except IOError:
        return False
