"""
Tests for ZeroMQ daemon communication.

These tests verify the Python DaemonClient class and ZeroMQ API endpoints.
Tests that require the daemon to NOT be running are skipped when daemon is detected.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path so we can import web as a package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the DaemonClient from web module
from web.constants import ZEROMQ_IPC_PATH  # noqa: E402
from web.services.daemon_client import DaemonClient, get_daemon_client  # noqa: E402


def is_daemon_running() -> bool:
    """Check if daemon is running (used for skip conditions)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "gpu_upsampler_alsa"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Skip marker for tests that require daemon to NOT be running
requires_daemon_not_running = pytest.mark.skipif(
    is_daemon_running(),
    reason="Test requires daemon to NOT be running (tests timeout/error behavior)",
)


class TestDaemonClient:
    """Test suite for DaemonClient class."""

    def test_client_initialization(self):
        """Test that DaemonClient initializes with correct defaults."""
        client = DaemonClient()
        assert client.endpoint == ZEROMQ_IPC_PATH
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

    def test_factory_function_returns_client(self):
        """Test that get_daemon_client returns a DaemonClient instance."""
        client = get_daemon_client()
        assert isinstance(client, DaemonClient)
        assert client.timeout_ms == 3000
        client.close()

    def test_factory_function_custom_timeout(self):
        """Test that get_daemon_client respects custom timeout."""
        client = get_daemon_client(timeout_ms=500)
        assert client.timeout_ms == 500
        client.close()

    @requires_daemon_not_running
    def test_send_command_timeout_when_daemon_not_running(self):
        """Test that send_command returns timeout error when daemon is not running."""
        client = DaemonClient(timeout_ms=500)  # Short timeout for test
        success, message = client.send_command("PING")
        assert success is False
        assert "timeout" in message.lower() or "not responding" in message.lower()
        client.close()

    @requires_daemon_not_running
    def test_reload_config_when_daemon_not_running(self):
        """Test reload_config returns appropriate error when daemon not running."""
        client = DaemonClient(timeout_ms=500)
        success, message = client.reload_config()
        assert success is False
        client.close()

    @requires_daemon_not_running
    def test_ping_when_daemon_not_running(self):
        """Test ping returns False when daemon not running."""
        client = DaemonClient(timeout_ms=500)
        result = client.ping()
        assert result is False
        client.close()

    @requires_daemon_not_running
    def test_get_stats_when_daemon_not_running(self):
        """Test get_stats returns error when daemon not running."""
        client = DaemonClient(timeout_ms=500)
        success, result = client.get_stats()
        assert success is False
        client.close()

    @requires_daemon_not_running
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
        assert message == "Unknown command"

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
        assert client.endpoint == ZEROMQ_IPC_PATH
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
