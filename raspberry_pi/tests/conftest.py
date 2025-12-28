from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """
    Allow running `pytest raspberry_pi/tests` directly.

    In some environments pytest changes the working directory during collection,
    so the repository root may not be on sys.path and `import raspberry_pi` fails.
    """

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
