"""
Integration tests for Crossfeed (HRTF) API endpoints.

Tests the following endpoints:
- GET /crossfeed/status - Get crossfeed status
- POST /crossfeed/enable - Enable crossfeed
- POST /crossfeed/disable - Disable crossfeed
- POST /crossfeed/size/{size} - Set head size
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from web.main import app  # noqa: E402
from web.services.daemon_client import DaemonError, DaemonResponse  # noqa: E402


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app, raise_server_exceptions=False)


def create_success_response(data: dict | None = None) -> DaemonResponse:
    """Helper to create successful DaemonResponse."""
    return DaemonResponse(success=True, data=data)


def create_error_response(
    error_code: str, message: str, inner_error: dict | None = None
) -> DaemonResponse:
    """Helper to create DaemonResponse with error."""
    error = DaemonError(
        error_code=error_code,
        message=message,
        inner_error=inner_error,
    )
    return DaemonResponse(success=False, error=error)


@pytest.fixture
def mock_daemon_client():
    """Fixture to mock daemon client."""
    from contextlib import contextmanager

    @contextmanager
    def _mock(response: DaemonResponse):
        with patch("web.routers.crossfeed.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.crossfeed_get_status.return_value = response
            mock_client.crossfeed_enable.return_value = response
            mock_client.crossfeed_disable.return_value = response
            mock_client.crossfeed_set_size.return_value = response
            mock_factory.return_value.__enter__.return_value = mock_client
            yield mock_client

    return _mock


class TestCrossfeedStatus:
    """Tests for GET /crossfeed/status endpoint."""

    def test_get_status_enabled(self, client, mock_daemon_client):
        """Get status when crossfeed is enabled."""
        response_data = {
            "enabled": True,
            "initialized": True,
            "head_size": "m",
        }
        with mock_daemon_client(create_success_response(response_data)):
            response = client.get("/crossfeed/status")

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert data["headSize"] == "m"
        assert "availableSizes" in data
        assert len(data["availableSizes"]) == 5
        assert "xs" in data["availableSizes"]
        assert "s" in data["availableSizes"]
        assert "m" in data["availableSizes"]
        assert "l" in data["availableSizes"]
        assert "xl" in data["availableSizes"]

    def test_get_status_disabled(self, client, mock_daemon_client):
        """Get status when crossfeed is disabled."""
        response_data = {
            "enabled": False,
            "initialized": True,
            "head_size": "l",
        }
        with mock_daemon_client(create_success_response(response_data)):
            response = client.get("/crossfeed/status")

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
        assert data["headSize"] == "l"

    def test_get_status_not_initialized(self, client, mock_daemon_client):
        """Get status when HRTF processor is not initialized."""
        response_data = {
            "enabled": False,
            "initialized": False,
            "head_size": None,
        }
        with mock_daemon_client(create_success_response(response_data)):
            response = client.get("/crossfeed/status")

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
        assert data["headSize"] is None

    def test_get_status_daemon_error(self, client, mock_daemon_client):
        """Get status when daemon returns error."""
        error_response = create_error_response(
            "IPC_TIMEOUT", "Daemon not responding"
        )
        with mock_daemon_client(error_response):
            response = client.get("/crossfeed/status")

        assert response.status_code == 504


class TestCrossfeedEnable:
    """Tests for POST /crossfeed/enable endpoint."""

    def test_enable_success(self, client, mock_daemon_client):
        """Enable crossfeed successfully."""
        response_data = {"message": "Crossfeed enabled"}
        with mock_daemon_client(create_success_response(response_data)):
            response = client.post("/crossfeed/enable")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Crossfeed enabled"

    def test_enable_daemon_error(self, client, mock_daemon_client):
        """Enable crossfeed when daemon returns error."""
        error_response = create_error_response(
            "CROSSFEED_NOT_INITIALIZED", "HRTF processor not initialized"
        )
        with mock_daemon_client(error_response):
            response = client.post("/crossfeed/enable")

        assert response.status_code == 500


class TestCrossfeedDisable:
    """Tests for POST /crossfeed/disable endpoint."""

    def test_disable_success(self, client, mock_daemon_client):
        """Disable crossfeed successfully."""
        response_data = {"message": "Crossfeed disabled"}
        with mock_daemon_client(create_success_response(response_data)):
            response = client.post("/crossfeed/disable")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Crossfeed disabled"

    def test_disable_daemon_error(self, client, mock_daemon_client):
        """Disable crossfeed when daemon returns error."""
        error_response = create_error_response(
            "IPC_TIMEOUT", "Daemon not responding"
        )
        with mock_daemon_client(error_response):
            response = client.post("/crossfeed/disable")

        assert response.status_code == 504


class TestCrossfeedSetSize:
    """Tests for POST /crossfeed/size/{size} endpoint."""

    def test_set_size_success(self, client, mock_daemon_client):
        """Set head size successfully."""
        response_data = {"head_size": "l"}
        with mock_daemon_client(create_success_response(response_data)):
            response = client.post("/crossfeed/size/l")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["headSize"] == "l"

    def test_set_size_invalid_size(self, client):
        """Set invalid head size should return 400."""
        response = client.post("/crossfeed/size/invalid")

        assert response.status_code == 400
        assert "Invalid head size" in response.json()["detail"]

    def test_set_size_case_insensitive(self, client, mock_daemon_client):
        """Head size should be case-insensitive."""
        response_data = {"head_size": "m"}
        with mock_daemon_client(create_success_response(response_data)):
            response = client.post("/crossfeed/size/M")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_set_size_all_valid_sizes(self, client, mock_daemon_client):
        """Test all valid head sizes."""
        valid_sizes = ["xs", "s", "m", "l", "xl"]
        for size in valid_sizes:
            response_data = {"head_size": size}
            with mock_daemon_client(create_success_response(response_data)):
                response = client.post(f"/crossfeed/size/{size}")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["headSize"] == size

    def test_set_size_daemon_error(self, client, mock_daemon_client):
        """Set head size when daemon returns error."""
        error_response = create_error_response(
            "CROSSFEED_NOT_INITIALIZED", "HRTF processor not initialized"
        )
        with mock_daemon_client(error_response):
            response = client.post("/crossfeed/size/m")

        assert response.status_code == 500

    def test_set_size_switch_failed(self, client, mock_daemon_client):
        """Set head size when switch fails."""
        error_response = create_error_response(
            "CROSSFEED_SIZE_SWITCH_FAILED", "Failed to switch head size"
        )
        with mock_daemon_client(error_response):
            response = client.post("/crossfeed/size/xl")

        assert response.status_code == 500

