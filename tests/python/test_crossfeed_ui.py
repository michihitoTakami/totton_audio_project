"""
Tests for Crossfeed UI component.

Tests the following:
- HTML template rendering with crossfeed UI elements
- Crossfeed UI component structure and elements
- JavaScript functions for crossfeed control
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from web.main import app  # noqa: E402
from web.services.daemon_client import DaemonResponse  # noqa: E402


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app, raise_server_exceptions=False)


def create_success_response(data: dict | None = None) -> DaemonResponse:
    """Helper to create successful DaemonResponse."""
    return DaemonResponse(success=True, data=data)


@pytest.fixture
def mock_daemon_client():
    """Fixture to mock daemon client."""
    from contextlib import contextmanager

    @contextmanager
    def _mock(response: DaemonResponse):
        with patch("web.routers.status.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.get_status.return_value = response
            mock_factory.return_value.__enter__.return_value = mock_client
            yield mock_client

    return _mock


class TestCrossfeedUITemplate:
    """Tests for crossfeed UI component in HTML template."""

    def test_root_endpoint_returns_html(self, client):
        """Test that root endpoint returns HTML."""
        response = client.get("/?lang=ja")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<!DOCTYPE html>" in response.text

    def test_crossfeed_section_exists(self, client):
        """Test that crossfeed section exists in HTML."""
        response = client.get("/?lang=ja")
        html = response.text

        # Check for crossfeed section header (Japanese)
        assert "🎧 クロスフィード" in html
        # Toggle switch uses Alpine.js, check for toggle-switch class
        assert 'class="toggle-switch"' in html
        assert '@click="toggleCrossfeed"' in html

    def test_crossfeed_toggle_switch_exists(self, client):
        """Test that toggle switch element exists."""
        response = client.get("/?lang=ja")
        html = response.text

        assert 'class="toggle-switch"' in html
        assert '@click="toggleCrossfeed"' in html

    def test_head_size_buttons_exist(self, client):
        """Test that all head size buttons exist."""
        response = client.get("/?lang=ja")
        html = response.text

        # Check for all head size buttons (Alpine.js style)
        assert "@click=\"setHeadSize('xs')\"" in html
        assert "@click=\"setHeadSize('s')\"" in html
        assert "@click=\"setHeadSize('m')\"" in html
        assert "@click=\"setHeadSize('l')\"" in html
        assert "@click=\"setHeadSize('xl')\"" in html

        # Check for button labels
        assert "XS" in html
        assert "S" in html
        assert "M" in html
        assert "L" in html
        assert "XL" in html

    def test_crossfeed_info_text_exists(self, client):
        """Test that info text exists."""
        response = client.get("/?lang=ja")
        html = response.text

        assert "正三角形配置" in html
        assert "±30°" in html
        assert "スピーカーリスニング" in html

    def test_loading_state_exists(self, client):
        """Test that loading state variable exists."""
        response = client.get("/?lang=ja")
        html = response.text

        # Alpine.js uses crossfeed.loading variable
        assert "crossfeed.loading" in html

    def test_status_display_exists(self, client):
        """Test that status display exists."""
        response = client.get("/?lang=ja")
        html = response.text

        # Check status display in status cards
        assert "クロスフィード" in html
        assert "status.crossfeed_enabled" in html

    """Tests for responsive design elements."""

    def test_viewport_meta_tag_exists(self, client):
        """Test that viewport meta tag exists for responsive design."""
        response = client.get("/?lang=ja")
        html = response.text

        assert 'name="viewport"' in html
        assert "width=device-width" in html
        assert "initial-scale=1.0" in html

        assert "display: flex" in html
        assert ".head-size-group" in html
        assert ".toggle-container" in html
