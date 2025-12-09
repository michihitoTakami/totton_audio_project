"""Modern Target (KB5000_7) correction declarations and helpers."""

import sys
from dataclasses import dataclass
from typing import Any

from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from opra import MODERN_TARGET_CORRECTION_BAND  # noqa: E402


@dataclass(frozen=True)
class CorrectionFilter:
    """Immutable representation of a correction band."""

    frequency: float
    gain: float
    q: float


@dataclass(frozen=True)
class CorrectionTolerance:
    """Per-parameter tolerance for matching correction bands."""

    frequency: float
    gain: float
    q: float


MODERN_TARGET_PRIMARY = CorrectionFilter(
    frequency=float(MODERN_TARGET_CORRECTION_BAND["frequency"]),
    gain=float(MODERN_TARGET_CORRECTION_BAND["gain_db"]),
    q=float(MODERN_TARGET_CORRECTION_BAND["q"]),
)

MODERN_TARGET_SECONDARY = CorrectionFilter(frequency=2350.0, gain=-0.9, q=2.0)

MODERN_TARGET_FILTERS: tuple[CorrectionFilter, ...] = (
    MODERN_TARGET_PRIMARY,
    MODERN_TARGET_SECONDARY,
)

MODERN_TARGET_TOLERANCE = CorrectionTolerance(frequency=2.0, gain=0.1, q=0.05)


def _is_close(value: float | None, target: float, tolerance: float) -> bool:
    """Check if value is within tolerance of target."""
    return value is not None and abs(value - target) <= tolerance


def is_modern_target_filter(parsed_filter: dict[str, Any] | None) -> bool:
    """Detect whether a parsed filter matches the KB5000_7 correction bands."""
    if not parsed_filter:
        return False

    for target in MODERN_TARGET_FILTERS:
        if (
            _is_close(
                parsed_filter.get("frequency"),
                target.frequency,
                MODERN_TARGET_TOLERANCE.frequency,
            )
            and _is_close(
                parsed_filter.get("gain"), target.gain, MODERN_TARGET_TOLERANCE.gain
            )
            and _is_close(parsed_filter.get("q"), target.q, MODERN_TARGET_TOLERANCE.q)
        ):
            return True

    return False
