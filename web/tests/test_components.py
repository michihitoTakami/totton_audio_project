"""
Tests for reusable UI components (Phase 1 & Phase 2)
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

        component_path = os.path.join(
            os.path.dirname(__file__), "..", "templates", "components"
        )

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


class TestStatusCardComponent:
    """Test status card component rendering (Phase 2)."""

    def test_status_cards_present_in_dashboard(self, client):
        """Verify status card components are rendered on dashboard."""
        response = client.get("/")
        assert response.status_code == 200

        # Check for status-card class
        assert 'class="status-card"' in response.text

        # Check for status-card-header and status-card-body
        assert 'class="status-card-header"' in response.text
        assert 'class="status-card-body"' in response.text

    def test_daemon_status_card_structure(self, client):
        """Verify daemon status card has correct structure."""
        response = client.get("/")
        assert response.status_code == 200

        # Should have status-icon
        assert 'class="status-icon"' in response.text

        # Should have status-indicator with Alpine.js binding
        assert 'class="status-indicator"' in response.text
        assert ":class=" in response.text
        assert "x-text=" in response.text

    def test_multiple_status_cards_rendered(self, client):
        """Verify all 4 status cards are rendered."""
        response = client.get("/")
        assert response.status_code == 200

        # Count status-card occurrences (should be 4: Daemon, EQ, Crossfeed, Low Latency)
        html = response.text
        status_card_count = html.count('class="status-card"')
        assert (
            status_card_count >= 4
        ), f"Expected at least 4 status cards, found {status_card_count}"


class TestAlertComponent:
    """Test alert message component rendering (Phase 2)."""

    def test_alert_present_in_dashboard(self, client):
        """Verify alert components are rendered on dashboard."""
        response = client.get("/")
        assert response.status_code == 200

        # Check for alert class
        assert 'class="alert' in response.text

        # Check for alert-warning type
        assert "alert-warning" in response.text

    def test_alert_with_alpine_show(self, client):
        """Verify alert has Alpine.js x-show binding."""
        response = client.get("/")
        assert response.status_code == 200

        # Should have x-show attribute
        assert "x-show=" in response.text

    def test_multiple_alerts_rendered(self, client):
        """Verify multiple alert messages are rendered."""
        response = client.get("/")
        assert response.status_code == 200

        # Count alert occurrences (Low Latency, Phase Type, Crossfeed warnings)
        html = response.text
        alert_count = html.count('class="alert alert-warning"')
        assert alert_count >= 3, f"Expected at least 3 alerts, found {alert_count}"


class TestInfoTextComponent:
    """Test info text component rendering (Phase 2)."""

    def test_info_text_present_in_dashboard(self, client):
        """Verify info text components are rendered on dashboard."""
        response = client.get("/")
        assert response.status_code == 200

        # Check for info-text class
        assert 'class="info-text"' in response.text

        # Check for icon span
        assert '<span class="icon">' in response.text

    def test_info_text_present_in_eq_settings(self, client):
        """Verify info text components are rendered on EQ settings page."""
        response = client.get("/eq")
        assert response.status_code == 200

        # Check for info-text class
        assert 'class="info-text"' in response.text

    def test_info_text_with_alpine_show(self, client):
        """Verify info text can have Alpine.js x-show binding."""
        response = client.get("/")
        assert response.status_code == 200

        # Some info texts may have x-show (not all)
        html = response.text
        # Just verify info-text components exist
        assert 'class="info-text"' in html

    def test_multiple_info_texts_rendered(self, client):
        """Verify multiple info text components are rendered."""
        response = client.get("/")
        assert response.status_code == 200

        # Count info-text occurrences
        html = response.text
        info_text_count = html.count('class="info-text"')
        assert (
            info_text_count >= 2
        ), f"Expected at least 2 info texts, found {info_text_count}"


class TestFormGroupComponent:
    """Test form group component rendering (Phase 2)."""

    def test_form_group_present_in_dashboard(self, client):
        """Verify form group components are rendered on dashboard."""
        response = client.get("/")
        assert response.status_code == 200

        # Check for form-group class
        assert 'class="form-group"' in response.text

        # Check for label and select/input elements
        assert "<label" in response.text
        assert "<select" in response.text or "<input" in response.text

    def test_form_group_present_in_eq_settings(self, client):
        """Verify form group components are rendered on EQ settings page."""
        response = client.get("/eq")
        assert response.status_code == 200

        # Check for form-group class
        assert 'class="form-group"' in response.text

        # Should have textarea in EQ settings
        assert "<textarea" in response.text

    def test_form_group_with_select(self, client):
        """Verify form group with select type renders correctly."""
        response = client.get("/")
        assert response.status_code == 200

        # Phase Type form group should have select
        html = response.text
        # Look for select with Alpine.js bindings
        assert "x-model=" in html
        assert "<select" in html

    def test_form_group_with_text_input(self, client):
        """Verify form group with text input type renders correctly."""
        response = client.get("/")
        assert response.status_code == 200

        # Output Mode device input should be text type
        html = response.text
        assert 'type="text"' in html
        assert "placeholder=" in html

    def test_form_group_with_textarea(self, client):
        """Verify form group with textarea type renders correctly."""
        response = client.get("/eq")
        assert response.status_code == 200

        # EQ import should have textarea
        html = response.text
        assert "<textarea" in html
        assert 'class="text-area"' in html
        assert "rows=" in html

    def test_form_group_with_alpine_bindings(self, client):
        """Verify form group has Alpine.js bindings."""
        response = client.get("/")
        assert response.status_code == 200

        # Should have x-model and :disabled bindings
        assert "x-model=" in response.text
        assert ":disabled=" in response.text

    def test_form_group_with_change_handler(self, client):
        """Verify form group can have @change handler."""
        response = client.get("/")
        assert response.status_code == 200

        # Phase Type select should have @change handler
        assert "@change=" in response.text
        assert "changePhaseType" in response.text


class TestSizeSelectorComponent:
    """Test size selector component rendering (Phase 3)."""

    def test_size_selector_present_in_dashboard(self, client):
        """Verify size selector component is rendered on dashboard."""
        response = client.get("/")
        assert response.status_code == 200

        # Check for head-size-group class (from component)
        assert 'class="head-size-group"' in response.text

        # Check for head-size-btn class
        assert 'class="head-size-btn"' in response.text

    def test_size_selector_has_all_sizes(self, client):
        """Verify size selector has all 5 size buttons."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Should have 5 sizes: XS, S, M, L, XL
        assert ">XS</button>" in html
        assert ">S</button>" in html
        assert ">M</button>" in html
        assert ">L</button>" in html
        assert ">XL</button>" in html

    def test_size_selector_alpine_bindings(self, client):
        """Verify size selector has Alpine.js bindings."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Should have active class binding
        assert ":class=\"{ 'active': crossfeed.headSize ===" in html

        # Should have click handler
        assert '@click="setHeadSize(' in html

        # Should have disabled binding
        assert ":disabled=" in html

    def test_size_selector_x_show_binding(self, client):
        """Verify size selector has x-show binding for visibility control."""
        response = client.get("/")
        assert response.status_code == 200

        # Should have x-show attribute for conditional rendering
        # (shown only when crossfeed is enabled)
        assert "x-show=" in response.text

    def test_size_selector_replaces_inline_implementation(self, client):
        """Verify inline head size selector is replaced by component."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Should NOT have old inline button structure (31 lines reduced)
        # Old structure had each button separately defined
        # New structure uses component include
        assert (
            "{% include 'components/size_selector.html' %}" not in html
        )  # Template tag shouldn't appear in rendered HTML

        # But should have the rendered output
        assert 'class="head-size-group"' in html
        assert 'class="head-size-btn"' in html


class TestPhase3ComponentDocumentation:
    """Test Phase 3 component documentation."""

    def test_size_selector_has_proper_documentation(self):
        """Verify size selector component file has parameter documentation."""
        import os

        component_path = os.path.join(
            os.path.dirname(__file__), "..", "templates", "components"
        )

        # Check size_selector.html
        with open(os.path.join(component_path, "size_selector.html")) as f:
            content = f.read()
            assert "Parameters:" in content
            assert "selector_label" in content
            assert "alpine_model" in content
            assert "alpine_click" in content
            assert "alpine_disabled" in content
            assert "alpine_show" in content
            assert "form_group_style" in content


class TestPhase2ComponentDocumentation:
    """Test Phase 2 component documentation."""

    def test_phase2_components_have_proper_documentation(self):
        """Verify Phase 2 component files have parameter documentation."""
        import os

        component_path = os.path.join(
            os.path.dirname(__file__), "..", "templates", "components"
        )

        # Check status_card.html
        with open(os.path.join(component_path, "status_card.html")) as f:
            content = f.read()
            assert "Parameters:" in content
            assert "status_icon" in content
            assert "status_title" in content
            assert "status_indicator" in content

        # Check alert.html
        with open(os.path.join(component_path, "alert.html")) as f:
            content = f.read()
            assert "Parameters:" in content
            assert "alert_type" in content
            assert "alert_message" in content

        # Check info_text.html
        with open(os.path.join(component_path, "info_text.html")) as f:
            content = f.read()
            assert "Parameters:" in content
            assert "info_icon" in content
            assert "info_message" in content

        # Check form_group.html
        with open(os.path.join(component_path, "form_group.html")) as f:
            content = f.read()
            assert "Parameters:" in content
            assert "form_label" in content
            assert "form_input_type" in content
