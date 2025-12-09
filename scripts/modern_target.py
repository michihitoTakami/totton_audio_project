"""KB5000_7 Modern Target correction parameters in one place."""

from dataclasses import dataclass
from typing import TypedDict


class CorrectionBandDict(TypedDict):
    """Type definition for correction band parameters."""

    filter_type: str
    frequency: float
    gain_db: float
    q: float


@dataclass(frozen=True)
class CorrectionTolerance:
    """Tolerance for matching correction bands (used in parsing)."""

    frequency: float
    gain_db: float
    q: float


@dataclass(frozen=True)
class ModernTargetSpec:
    """Bundle of KB5000_7 correction parameters and tolerances."""

    name: str
    primary: CorrectionBandDict
    secondary: CorrectionBandDict
    tolerance: CorrectionTolerance

    @property
    def filters(self) -> tuple[CorrectionBandDict, CorrectionBandDict]:
        return (self.primary, self.secondary)


# ============================================================================
# Modern Target Correction (KB5000_7)
# ここを修正すれば一次/二次バンドと許容値をまとめて変えられる
# ============================================================================
MODERN_TARGET_SPEC = ModernTargetSpec(
    name="KB5000_7",
    primary={
        "filter_type": "PK",
        "frequency": 5366.0,
        "gain_db": 2.8,
        "q": 1.5,
    },
    secondary={
        "filter_type": "PK",
        "frequency": 2350.0,
        "gain_db": -0.9,
        "q": 2.0,
    },
    tolerance=CorrectionTolerance(
        frequency=2.0,  # Hz
        gain_db=0.1,  # dB
        q=0.05,
    ),
)
# Backward-compatible alias for legacy imports
MODERN_TARGET_CORRECTION_BAND: CorrectionBandDict = MODERN_TARGET_SPEC.primary
