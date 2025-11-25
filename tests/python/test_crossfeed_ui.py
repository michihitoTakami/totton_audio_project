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
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<!DOCTYPE html>" in response.text

    def test_crossfeed_section_exists(self, client):
        """Test that crossfeed section exists in HTML."""
        response = client.get("/")
        html = response.text

        # Check for crossfeed section header
        assert "🎧 クロスフィード" in html
        assert 'id="crossfeedToggle"' in html

    def test_crossfeed_toggle_switch_exists(self, client):
        """Test that toggle switch element exists."""
        response = client.get("/")
        html = response.text

        assert 'class="toggle-switch"' in html
        assert 'id="crossfeedToggle"' in html

    def test_head_size_buttons_exist(self, client):
        """Test that all head size buttons exist."""
        response = client.get("/")
        html = response.text

        # Check for all head size buttons
        assert 'data-size="xs"' in html
        assert 'data-size="s"' in html
        assert 'data-size="m"' in html
        assert 'data-size="l"' in html
        assert 'data-size="xl"' in html

        # Check for button labels
        assert "XS" in html
        assert "S" in html
        assert "M" in html
        assert "L" in html
        assert "XL" in html

    def test_crossfeed_info_text_exists(self, client):
        """Test that info text exists."""
        response = client.get("/")
        html = response.text

        assert "正三角形配置" in html
        assert "±30°" in html
        assert "スピーカーリスニング" in html

    def test_loading_indicator_exists(self, client):
        """Test that loading indicator exists."""
        response = client.get("/")
        html = response.text

        assert 'id="crossfeedLoading"' in html
        assert "適用中" in html
        assert 'class="loading-indicator"' in html

    def test_status_display_exists(self, client):
        """Test that status display exists."""
        response = client.get("/")
        html = response.text

        assert 'id="crossfeedStatus"' in html
        assert 'id="crossfeedStatusIndicator"' in html
        assert 'id="crossfeedStatusText"' in html
        assert "ステータス" in html

    def test_message_display_exists(self, client):
        """Test that message display exists."""
        response = client.get("/")
        html = response.text

        assert 'id="crossfeedMessage"' in html
        assert 'class="message"' in html


class TestCrossfeedJavaScript:
    """Tests for crossfeed JavaScript functions in HTML template."""

    def test_crossfeed_state_object_exists(self, client):
        """Test that crossfeedState object is defined."""
        response = client.get("/")
        html = response.text

        assert "crossfeedState" in html
        assert "enabled" in html
        assert "headSize" in html
        assert "isApplying" in html

    def test_fetch_crossfeed_status_function_exists(self, client):
        """Test that fetchCrossfeedStatus function exists."""
        response = client.get("/")
        html = response.text

        assert "fetchCrossfeedStatus" in html
        assert "/crossfeed/status" in html

    def test_toggle_crossfeed_function_exists(self, client):
        """Test that toggleCrossfeed function exists."""
        response = client.get("/")
        html = response.text

        assert "toggleCrossfeed" in html
        assert "/crossfeed/enable" in html
        assert "/crossfeed/disable" in html

    def test_set_head_size_function_exists(self, client):
        """Test that setHeadSize function exists."""
        response = client.get("/")
        html = response.text

        assert "setHeadSize" in html
        assert "/crossfeed/size/" in html

    def test_update_crossfeed_toggle_function_exists(self, client):
        """Test that updateCrossfeedToggle function exists."""
        response = client.get("/")
        html = response.text

        assert "updateCrossfeedToggle" in html

    def test_update_head_size_buttons_function_exists(self, client):
        """Test that updateHeadSizeButtons function exists."""
        response = client.get("/")
        html = response.text

        assert "updateHeadSizeButtons" in html

    def test_set_crossfeed_loading_function_exists(self, client):
        """Test that setCrossfeedLoading function exists."""
        response = client.get("/")
        html = response.text

        assert "setCrossfeedLoading" in html

    def test_event_listeners_attached(self, client):
        """Test that event listeners are attached."""
        response = client.get("/")
        html = response.text

        # Check for event listener attachments
        assert "addEventListener" in html
        assert "crossfeedToggle" in html
        assert "head-size-btn" in html

    def test_status_polling_configured(self, client):
        """Test that status polling is configured."""
        response = client.get("/")
        html = response.text

        assert "setInterval" in html
        assert "fetchCrossfeedStatus" in html
        assert "5000" in html


class TestCrossfeedCSS:
    """Tests for crossfeed CSS styles in HTML template."""

    def test_toggle_switch_styles_exist(self, client):
        """Test that toggle switch CSS exists."""
        response = client.get("/")
        html = response.text

        assert ".toggle-switch" in html
        assert ".toggle-switch.active" in html

    def test_head_size_button_styles_exist(self, client):
        """Test that head size button CSS exists."""
        response = client.get("/")
        html = response.text

        assert ".head-size-btn" in html
        assert ".head-size-btn.active" in html
        assert ".head-size-btn:hover" in html

    def test_loading_indicator_styles_exist(self, client):
        """Test that loading indicator CSS exists."""
        response = client.get("/")
        html = response.text

        assert ".loading-indicator" in html
        assert ".loading-indicator.visible" in html

    def test_status_display_styles_exist(self, client):
        """Test that status display CSS exists."""
        response = client.get("/")
        html = response.text

        assert ".status-display" in html
        assert ".status-indicator" in html
        assert ".status-indicator.active" in html


class TestCrossfeedUIResponsiveDesign:
    """Tests for responsive design elements."""

    def test_viewport_meta_tag_exists(self, client):
        """Test that viewport meta tag exists for responsive design."""
        response = client.get("/")
        html = response.text

        assert 'name="viewport"' in html
        assert "width=device-width" in html
        assert "initial-scale=1.0" in html

    def test_flexbox_layout_used(self, client):
        """Test that flexbox is used for layout."""
        response = client.get("/")
        html = response.text

        assert "display: flex" in html
        assert ".head-size-group" in html
        assert ".toggle-container" in html


class TestCrossfeedUIErrorHandling:
    """Tests for error handling in UI."""

    def test_error_message_display_exists(self, client):
        """Test that error message display exists."""
        response = client.get("/")
        html = response.text

        assert "showCrossfeedMessage" in html
        assert ".message.error" in html
        assert ".message.success" in html

    def test_api_error_handling_exists(self, client):
        """Test that API error handling exists."""
        response = client.get("/")
        html = response.text

        # Check for error handling in fetch calls
        assert "catch" in html
        assert "res.ok" in html
        assert "data.detail" in html or "data.message" in html

