"""pytest configuration and fixtures for Python tests."""

import sys
from pathlib import Path

import pytest

# Add scripts and project root directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_rate_44k():
    """44.1kHz sample rate."""
    return 44100


@pytest.fixture
def sample_rate_48k():
    """48kHz sample rate."""
    return 48000


@pytest.fixture
def coefficients_dir():
    """Path to filter coefficients directory."""
    return Path(__file__).parent.parent.parent / "data" / "coefficients"
