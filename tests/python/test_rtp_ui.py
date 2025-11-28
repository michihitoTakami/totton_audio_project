"""
Tests for the RTP session management UI (Issue #360).

These tests ensure:
- `/rtp` route serves HTML successfully
- Key form fields and validation hooks are present
- JavaScript client references the Control Plane API endpoints
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project root so `web` can be imported when running via pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from web.main import app  # noqa: E402


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app, raise_server_exceptions=False)


class TestRtpUiTemplate:
    """Template-level tests."""

    def test_rtp_page_is_served(self, client):
        """GET /rtp should return the HTML UI."""
        response = client.get("/rtp")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "RTPセッション管理" in response.text

    def test_required_form_fields_exist(self, client):
        """Form fields for session creation should be present."""
        html = client.get("/rtp").text
        for field_id in ["sessionId", "bindAddress", "port", "sdpBody", "srtpKey"]:
            assert f'id="{field_id}"' in html
        assert 'name="syncMode"' in html
        assert 'id="srtpToggle"' in html

    def test_session_list_placeholder_exists(self, client):
        """Active session list container should exist."""
        html = client.get("/rtp").text
        assert 'id="sessionList"' in html
        assert 'id="sessionListEmpty"' in html
        assert "Active Sessions" in html


class TestRtpUiJavaScript:
    """JavaScript helper tests."""

    def test_api_client_references_control_plane(self, client):
        """rtpApi helper should target the Control Plane endpoints."""
        html = client.get("/rtp").text
        assert "/api/rtp/sessions" in html
        assert "rtpApi" in html
        assert "buildPayload" in html

    def test_validation_helpers_exist(self, client):
        """Validation and rendering helpers should exist."""
        html = client.get("/rtp").text
        for token in ["validateField", "validateForm", "renderSessions", "handleStopSession"]:
            assert token in html

    def test_sync_presets_declared(self, client):
        """Sync preset options should be embedded for UX switches."""
        html = client.get("/rtp").text
        assert "SYNC_PRESETS" in html
        assert 'data-preset="ptp"' in html

