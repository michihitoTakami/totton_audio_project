"""
Tests for ZeroMQ daemon communication.

These tests verify the Python DaemonClient class and ZeroMQ API endpoints.
Note: Most tests require the daemon to NOT be running, as they test fallback behavior.
"""

from unittest.mock import patch, MagicMock

# Import the DaemonClient from web module
from web.main import DaemonClient, daemon_client, ZEROMQ_IPC_PATH


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


class TestGlobalDaemonClient:
    """Test the global daemon_client instance."""

    def test_global_client_exists(self):
        """Test that global daemon_client is initialized."""
        assert daemon_client is not None
        assert isinstance(daemon_client, DaemonClient)

    def test_global_client_has_correct_endpoint(self):
        """Test global client uses correct endpoint."""
        assert daemon_client.endpoint == ZEROMQ_IPC_PATH
