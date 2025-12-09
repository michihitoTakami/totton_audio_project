"""Modern Target (KB5000_7) correction declarations and helpers."""

import sys
from dataclasses import dataclass
from typing import Any

from pathlib import Path

# Keep opra import in one place so KB5000_7定義をここだけで完結させる
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


@dataclass(frozen=True)
class CorrectionSpec:
    """Bundle of KB5000_7 correction parameters."""

    name: str
    primary: CorrectionFilter
    secondary: CorrectionFilter
    tolerance: CorrectionTolerance

    @property
    def filters(self) -> tuple[CorrectionFilter, ...]:
        return (self.primary, self.secondary)


# ============================================================================
# KB5000_7 Modern Target — このブロックだけ触れば補正値を変えられる
# ============================================================================
MODERN_TARGET_SPEC = CorrectionSpec(
    name="KB5000_7",
    primary=CorrectionFilter(
        frequency=float(MODERN_TARGET_CORRECTION_BAND["frequency"]),
        gain=float(MODERN_TARGET_CORRECTION_BAND["gain_db"]),
        q=float(MODERN_TARGET_CORRECTION_BAND["q"]),
    ),
    # Secondary tweak to smooth upper-mid region
    secondary=CorrectionFilter(frequency=2350.0, gain=-0.9, q=2.0),
    # Tolerance for matching filters when parsing text (allows rounding / missing numbers)
    tolerance=CorrectionTolerance(frequency=2.0, gain=0.1, q=0.05),
)
# ============================================================================


def _is_close(value: float | None, target: float, tolerance: float) -> bool:
    """Check if value is within tolerance of target."""
    return value is not None and abs(value - target) <= tolerance


def is_modern_target_filter(
    parsed_filter: dict[str, Any] | None, spec: CorrectionSpec = MODERN_TARGET_SPEC
) -> bool:
    """Detect whether a parsed filter matches the KB5000_7 correction bands."""
    if not parsed_filter:
        return False

    for target in spec.filters:
        if (
            _is_close(
                parsed_filter.get("frequency"),
                target.frequency,
                spec.tolerance.frequency,
            )
            and _is_close(parsed_filter.get("gain"), target.gain, spec.tolerance.gain)
            and _is_close(parsed_filter.get("q"), target.q, spec.tolerance.q)
        ):
            return True

    return False
