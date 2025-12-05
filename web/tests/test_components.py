"""
Tests for reusable UI components (Phase 1)
"""

import pytest
from fastapi.testclient import TestClient
from web.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestUtilsJsGlobalAvailability:
    """Test that utils.js is properly loaded on all pages."""

    def test_utils_js_loaded_on_dashboard(self, client):
        """Verify utils.js is loaded on dashboard page."""
        response = client.get("/")
        assert response.status_code == 200
        assert '<script src="/static/js/utils.js"></script>' in response.text

    def test_utils_js_loaded_on_eq_settings(self, client):
        """Verify utils.js is loaded on EQ settings page."""
        response = client.get("/eq")
        assert response.status_code == 200
        assert '<script src="/static/js/utils.js"></script>' in response.text


class TestToggleSwitchComponent:
    """Test toggle switch component rendering."""

    def test_toggle_switch_present_in_dashboard(self, client):
        """Verify toggle switch components are rendered on dashboard."""
        response = client.get("/")
        assert response.status_code == 200

        # Check for toggle-container class (from component)
        assert 'class="toggle-container"' in response.text

        # Check for toggle-switch with Alpine.js bindings
        assert 'class="toggle-switch"' in response.text
        assert ":class=\"{ 'active':" in response.text

    def test_low_latency_toggle_structure(self, client):
        """Verify low latency toggle switch has correct structure."""
        response = client.get("/")
        assert response.status_code == 200

        # Should have toggle-label and toggle-desc
        assert 'class="toggle-label"' in response.text
        assert 'class="toggle-desc"' in response.text

        # Should have Alpine.js click handler
        assert '@click="toggleLowLatency"' in response.text

    def test_crossfeed_toggle_structure(self, client):
        """Verify crossfeed toggle switch has correct structure."""
        response = client.get("/")
        assert response.status_code == 200

        # Should have Alpine.js click handler for crossfeed
        assert '@click="toggleCrossfeed"' in response.text


class TestActionButtonComponent:
    """Test action button component rendering."""

    def test_action_buttons_present_in_dashboard(self, client):
        """Verify action buttons are rendered on dashboard."""
        response = client.get("/")
        assert response.status_code == 200

        # Check for action-button class
        assert 'class="action-button' in response.text

        # Check for button types
        assert "btn-primary" in response.text
        assert "btn-secondary" in response.text

    def test_action_buttons_present_in_eq_settings(self, client):
        """Verify action buttons are rendered on EQ settings page."""
        response = client.get("/eq")
        assert response.status_code == 200

        # Check for action-button class
        assert 'class="action-button' in response.text

        # Check for icon buttons
        assert 'class="btn-icon' in response.text

    def test_primary_button_with_icon(self, client):
        """Verify primary button with icon structure."""
        response = client.get("/")
        assert response.status_code == 200

        # Should have action-icon and action-label
        assert 'class="action-icon"' in response.text
        assert 'class="action-label"' in response.text

    def test_icon_only_button_structure(self, client):
        """Verify icon-only button structure in EQ settings."""
        response = client.get("/eq")
        assert response.status_code == 200

        # Should have btn-icon class
        assert 'class="btn-icon btn-primary"' in response.text
        assert 'class="btn-icon btn-danger"' in response.text

        # Should have title attributes for accessibility
        assert "title=" in response.text

    def test_button_disabled_alpine_binding(self, client):
        """Verify buttons have Alpine.js disabled binding."""
        response = client.get("/")
        assert response.status_code == 200

        # Should have :disabled Alpine.js binding
        assert ":disabled=" in response.text

    def test_button_click_handlers(self, client):
        """Verify buttons have Alpine.js click handlers."""
        response = client.get("/")
        assert response.status_code == 200

        # Check for various click handlers
        assert '@click="saveOutputMode"' in response.text
        assert '@click="applyEQ"' in response.text
        assert '@click="deactivateEQ"' in response.text
        assert '@click="restartDaemon"' in response.text

    def test_button_style_attribute_support(self, client):
        """Verify button style attribute is properly applied."""
        response = client.get("/eq")
        assert response.status_code == 200

        # The OPRA apply button should have margin-top style
        # This checks that button_style parameter works correctly
        html = response.text
        # Look for button with margin-top style near OPRA section
        assert 'style="margin-top: 12px;"' in html or "margin-top: 12px" in html


class TestComponentIntegration:
    """Test component integration with existing functionality."""

    def test_no_duplicate_showtoast_implementations(self, client):
        """Verify old showToast implementations are removed."""
        dashboard_response = client.get("/")
        eq_response = client.get("/eq")

        # Old inline showToast function should not exist
        # (it was inside Alpine.js data function)
        dashboard_html = dashboard_response.text
        eq_html = eq_response.text

        # Check that showToast is not defined inline
        # The global version is in utils.js, not in the HTML
        assert "function showToast(" not in dashboard_html.split("<script>")[-1]
        assert "function showToast(" not in eq_html.split("<script>")[-1]

    def test_components_dont_break_alpine_context(self, client):
        """Verify components work within Alpine.js context."""
        response = client.get("/")
        assert response.status_code == 200

        # Alpine.js data function should still be present
        assert "x-data=" in response.text

        # Components should have Alpine.js bindings
        assert ":class=" in response.text
        assert "@click=" in response.text
        assert ":disabled=" in response.text

    def test_translation_keys_work_in_components(self, client):
        """Verify translation system works with components."""
        # Test with English
        response_en = client.get("/", headers={"Accept-Language": "en"})
        assert response_en.status_code == 200
        assert len(response_en.text) > 0

        # Test with Japanese
        response_ja = client.get("/", headers={"Accept-Language": "ja"})
        assert response_ja.status_code == 200
        assert len(response_ja.text) > 0

        # Both should have button elements (components rendered)
        assert "<button" in response_en.text
        assert "<button" in response_ja.text


class TestComponentCodeQuality:
    """Test component code quality and best practices."""

    def test_no_inline_styles_in_toggle_component(self, client):
        """Verify toggle switch uses CSS classes, not inline styles."""
        response = client.get("/")
        html = response.text

        # Toggle switches should use toggle-container class
        assert 'class="toggle-container"' in html
        assert 'class="toggle-switch"' in html

        # Should NOT have lots of inline styles in toggle elements
        # (Some inline styles are OK for dynamic Alpine.js stuff)
        toggle_sections = html.split("toggle-container")
        for section in toggle_sections[1:]:  # Skip first split (before any toggle)
            # Get just the toggle section (until next div closes)
            toggle_html = section[:500]  # First 500 chars should cover the toggle
            # Count inline styles - should be minimal
            inline_style_count = toggle_html.count("style=")
            assert inline_style_count <= 1, "Toggle should not have many inline styles"

    def test_components_have_proper_documentation(self):
        """Verify component files have parameter documentation."""
        import os

        component_path = "/home/michihito/Working/gpu_os/worktrees/561-phase1-components/web/templates/components"

        # Check toggle_switch.html
        with open(os.path.join(component_path, "toggle_switch.html")) as f:
            content = f.read()
            assert "Parameters:" in content
            assert "toggle_label" in content
            assert "alpine_model" in content

        # Check action_button.html
        with open(os.path.join(component_path, "action_button.html")) as f:
            content = f.read()
            assert "Parameters:" in content
            assert "button_label" in content
            assert "alpine_click" in content
            assert "button_style" in content  # New parameter added
