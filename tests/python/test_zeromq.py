"""
Tests for ZeroMQ daemon communication.

These tests verify the Python DaemonClient class and ZeroMQ API endpoints.
Tests that require the daemon to NOT be running will stop daemon if running.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path so we can import web as a package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the DaemonClient from web module
from web.constants import ZEROMQ_IPC_PATH  # noqa: E402
from web.services.daemon_client import DaemonClient, get_daemon_client  # noqa: E402


def _expected_zmq_endpoint() -> str:
    """Return the endpoint DaemonClient should default to."""
    return os.environ.get("ZMQ_ENDPOINT", ZEROMQ_IPC_PATH)


def get_daemon_pids() -> list[int]:
    """Get PIDs of running daemon processes."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "gpu_upsampler_alsa"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return [int(pid) for pid in result.stdout.strip().split("\n") if pid]
        return []
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return []


def ensure_daemon_stopped(timeout_seconds: int = 5) -> None:
    """Stop daemon if running, raise error if not stopped within timeout."""
    pids = get_daemon_pids()
    if not pids:
        return

    # Send SIGTERM to all daemon processes
    for pid in pids:
        try:
            subprocess.run(["kill", "-TERM", str(pid)], timeout=1)
        except subprocess.TimeoutExpired:
            pass

    # Wait for daemon to stop
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if not get_daemon_pids():
            return
        time.sleep(0.1)

    # If still running, try SIGKILL
    pids = get_daemon_pids()
    for pid in pids:
        try:
            subprocess.run(["kill", "-KILL", str(pid)], timeout=1)
        except subprocess.TimeoutExpired:
            pass

    # Final check
    time.sleep(0.5)
    if get_daemon_pids():
        raise RuntimeError(
            f"Failed to stop daemon within {timeout_seconds} seconds. "
            "Please stop daemon manually before running tests."
        )


@pytest.fixture(scope="module", autouse=True)
def stop_daemon_before_tests():
    """Ensure daemon is stopped before running ZeroMQ tests."""
    ensure_daemon_stopped(timeout_seconds=5)


class TestDaemonClient:
    """Test suite for DaemonClient class."""

    def test_client_initialization(self):
        """Test that DaemonClient initializes with correct defaults."""
        client = DaemonClient()
        assert client.endpoint == _expected_zmq_endpoint()
        assert client.timeout_ms == 3000
        assert client._context is None
        assert client._socket is None

    def test_client_custom_endpoint(self):
        """Test DaemonClient with custom endpoint."""
        custom_endpoint = "ipc:///tmp/test.sock"
        client = DaemonClient(endpoint=custom_endpoint, timeout_ms=5000)
        assert client.endpoint == custom_endpoint
        assert client.timeout_ms == 5000

    def test_client_close_without_connection(self):
        """Test that close() works when not connected."""
        client = DaemonClient()
        # Should not raise any exception
        client.close()
        assert client._socket is None
        assert client._context is None

    def test_context_manager_enter_exit(self):
        """Test that context manager enters and exits correctly."""
        with DaemonClient() as client:
            assert isinstance(client, DaemonClient)
        # After exit, socket should be closed
        assert client._socket is None
        assert client._context is None

    def test_client_respects_env_endpoint(self, monkeypatch):
        """DaemonClient should prefer ZMQ_ENDPOINT when set."""
        custom_endpoint = "ipc:///tmp/custom.sock"
        monkeypatch.setenv("ZMQ_ENDPOINT", custom_endpoint)
        client = DaemonClient()
        assert client.endpoint == custom_endpoint
        client.close()

    def test_send_command_timeout_when_daemon_not_running(self):
        """Test that send_command returns timeout error when daemon is not running."""
        client = DaemonClient(timeout_ms=500)  # Short timeout for test
        success, message = client.send_command("PING")
        assert success is False
        assert "timeout" in message.lower() or "not responding" in message.lower()
        client.close()

    def test_reload_config_when_daemon_not_running(self):
        """Test reload_config returns appropriate error when daemon not running."""
        client = DaemonClient(timeout_ms=500)
        success, message = client.reload_config()
        assert success is False
        client.close()

    def test_ping_when_daemon_not_running(self):
        """Test ping returns False when daemon not running."""
        client = DaemonClient(timeout_ms=500)
        result = client.ping()
        assert result is False
        client.close()

    def test_get_stats_when_daemon_not_running(self):
        """Test get_stats returns error when daemon not running."""
        client = DaemonClient(timeout_ms=500)
        success, result = client.get_stats()
        assert success is False
        client.close()

    def test_context_manager_timeout(self):
        """Test context manager handles timeout correctly."""
        with get_daemon_client(timeout_ms=500) as client:
            success, message = client.send_command("PING")
            assert success is False
        # Socket should be cleaned up
        assert client._socket is None


class TestDaemonClientMocked:
    """Test DaemonClient with mocked ZeroMQ."""

    def test_send_command_success_ok(self):
        """Test successful command with OK response."""
        client = DaemonClient()

        mock_socket = MagicMock()
        mock_socket.recv_string.return_value = "OK"

        with patch.object(client, "_ensure_connected", return_value=mock_socket):
            success, message = client.send_command("PING")

        assert success is True
        assert message == "Command executed"
        mock_socket.send_string.assert_called_once_with("PING")

    def test_send_command_success_with_data(self):
        """Test successful command with OK:data response."""
        client = DaemonClient()

        mock_socket = MagicMock()
        mock_socket.recv_string.return_value = 'OK:{"key":"value"}'

        with patch.object(client, "_ensure_connected", return_value=mock_socket):
            success, message = client.send_command("STATS")

        assert success is True
        assert message == '{"key":"value"}'

    def test_send_command_error(self):
        """Test command with ERR response."""
        client = DaemonClient()

        mock_socket = MagicMock()
        mock_socket.recv_string.return_value = "ERR:Unknown command"

        with patch.object(client, "_ensure_connected", return_value=mock_socket):
            success, message = client.send_command("INVALID")

        assert success is False
        assert (
            "Unknown command" in message
        )  # Message format changed to include error code prefix

    def test_reload_config_success(self):
        """Test reload_config with mocked success."""
        client = DaemonClient()

        mock_socket = MagicMock()
        mock_socket.recv_string.return_value = "OK"

        with patch.object(client, "_ensure_connected", return_value=mock_socket):
            success, message = client.reload_config()

        assert success is True
        mock_socket.send_string.assert_called_once_with("RELOAD")

    def test_get_stats_parses_json(self):
        """Test get_stats correctly parses JSON response."""
        client = DaemonClient()

        stats_json = '{"clip_count":100,"total_samples":1000,"clip_rate":0.1}'
        mock_socket = MagicMock()
        mock_socket.recv_string.return_value = f"OK:{stats_json}"

        with patch.object(client, "_ensure_connected", return_value=mock_socket):
            success, stats = client.get_stats()

        assert success is True
        assert isinstance(stats, dict)
        assert stats["clip_count"] == 100
        assert stats["total_samples"] == 1000
        assert stats["clip_rate"] == 0.1

    def test_get_stats_invalid_json(self):
        """Test get_stats handles invalid JSON gracefully."""
        client = DaemonClient()

        mock_socket = MagicMock()
        mock_socket.recv_string.return_value = "OK:not-valid-json"

        with patch.object(client, "_ensure_connected", return_value=mock_socket):
            success, result = client.get_stats()

        assert success is False
        assert "Invalid JSON" in result


class TestFactoryFunction:
    """Test the get_daemon_client factory function."""

    def test_factory_returns_client(self):
        """Test that factory returns DaemonClient instance."""
        client = get_daemon_client()
        assert isinstance(client, DaemonClient)
        assert client.endpoint == _expected_zmq_endpoint()
        client.close()

    def test_factory_custom_timeout(self):
        """Test factory with custom timeout."""
        client = get_daemon_client(timeout_ms=1000)
        assert client.timeout_ms == 1000
        client.close()

    def test_factory_context_manager(self):
        """Test factory works as context manager."""
        with get_daemon_client() as client:
            assert isinstance(client, DaemonClient)
        # Socket should be cleaned up after context exit
        assert client._socket is None

    def test_factory_uses_env_endpoint(self, monkeypatch):
        """Factory should respect ZMQ_ENDPOINT overrides."""
        custom_endpoint = "ipc:///tmp/factory.sock"
        monkeypatch.setenv("ZMQ_ENDPOINT", custom_endpoint)
        client = get_daemon_client()
        assert client.endpoint == custom_endpoint
        client.close()
