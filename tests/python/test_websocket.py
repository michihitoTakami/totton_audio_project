"""
Tests for WebSocket stats streaming endpoint.
"""

from fastapi.testclient import TestClient

from web.main import app


class TestWebSocketStats:
    """Test suite for /ws/stats WebSocket endpoint."""

    def test_websocket_connection(self):
        """Test that WebSocket connection can be established."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/stats") as websocket:
                # Connection should be established
                data = websocket.receive_json()
                assert data is not None

    def test_websocket_returns_required_fields(self):
        """Test that WebSocket response contains all required fields."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/stats") as websocket:
                data = websocket.receive_json()

                # Check required fields exist
                assert "clip_rate" in data
                assert "clip_count" in data
                assert "total_samples" in data
                assert "daemon_running" in data

    def test_websocket_field_types(self):
        """Test that WebSocket response fields have correct types."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/stats") as websocket:
                data = websocket.receive_json()

                # Check field types
                assert isinstance(data["clip_rate"], (int, float))
                assert isinstance(data["clip_count"], int)
                assert isinstance(data["total_samples"], int)
                assert isinstance(data["daemon_running"], bool)

    def test_websocket_clip_rate_non_negative(self):
        """Test that clip_rate is non-negative."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/stats") as websocket:
                data = websocket.receive_json()
                assert data["clip_rate"] >= 0

    def test_websocket_clip_count_non_negative(self):
        """Test that clip_count is non-negative."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/stats") as websocket:
                data = websocket.receive_json()
                assert data["clip_count"] >= 0

    def test_websocket_total_samples_non_negative(self):
        """Test that total_samples is non-negative."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/stats") as websocket:
                data = websocket.receive_json()
                assert data["total_samples"] >= 0

    def test_websocket_multiple_messages(self):
        """Test that WebSocket sends multiple messages over time."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/stats") as websocket:
                # Receive first message
                data1 = websocket.receive_json()
                assert data1 is not None

                # Receive second message (should arrive after ~1 second)
                data2 = websocket.receive_json()
                assert data2 is not None

                # Both should have required fields
                for data in [data1, data2]:
                    assert "clip_rate" in data
                    assert "daemon_running" in data

    def test_websocket_graceful_disconnect(self):
        """Test that WebSocket handles client disconnect gracefully."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/stats") as websocket:
                data = websocket.receive_json()
                assert data is not None
            # Connection should close without error when context exits
