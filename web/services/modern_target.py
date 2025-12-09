"""Modern Target (KB5000_7) correction declarations and helpers."""

import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 単一の定義元に集約（一次/二次バンドと許容値を scripts/modern_target.py に保持）
from modern_target import MODERN_TARGET_SPEC  # type: ignore  # noqa: E402


def _is_close(value: float | None, target: float, tolerance: float) -> bool:
    """Check if value is within tolerance of target."""
    return value is not None and abs(value - target) <= tolerance


def is_modern_target_filter(
    parsed_filter: dict[str, Any] | None, spec=MODERN_TARGET_SPEC
) -> bool:
    """Detect whether a parsed filter matches the KB5000_7 correction bands."""
    if not parsed_filter:
        return False

    for target in spec.filters:
        if (
            _is_close(
                parsed_filter.get("frequency"),
                target["frequency"],
                spec.tolerance.frequency,
            )
            and _is_close(
                parsed_filter.get("gain"), target["gain_db"], spec.tolerance.gain_db
            )
            and _is_close(parsed_filter.get("q"), target["q"], spec.tolerance.q)
        ):
            return True

    return False
