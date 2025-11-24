"""pytest configuration and fixtures for Python tests."""

import subprocess
import sys
from pathlib import Path

import pytest

# Add scripts and project root directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


def _is_daemon_running() -> bool:
    """Check if audio daemon is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "gpu_upsampler_alsa"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _restart_daemon() -> bool:
    """Restart the audio daemon if it was running before tests."""
    daemon_script = PROJECT_ROOT / "scripts" / "daemon.sh"
    if not daemon_script.exists():
        return False
    try:
        subprocess.run(
            [str(daemon_script), "restart"],
            capture_output=True,
            timeout=30,
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture(scope="session", autouse=True)
def daemon_state_manager():
    """Restore daemon state after test session.

    If daemon was running before tests, restart it after all tests complete.
    This prevents tests from leaving the daemon in a stopped/broken state.
    """
    daemon_was_running = _is_daemon_running()
    yield
    # After all tests complete
    if daemon_was_running and not _is_daemon_running():
        print("\n[conftest] Daemon was stopped during tests, restarting...")
        _restart_daemon()


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
