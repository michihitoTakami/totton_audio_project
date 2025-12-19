"""Tests for Pi Settings page rendering."""

from web.templates.pages import render_pi_settings


def test_render_pi_settings_english():
    """Pi Settings page renders with main sections in English."""
    html = render_pi_settings(lang="en", current_page="pi")

    assert "Pi Bridge Status" in html
    assert "Pi → Jetson Status" in html
    assert "Pi Bridge Config" in html


def test_render_pi_settings_japanese():
    """Pi Settings page renders with Japanese translations."""
    html = render_pi_settings(lang="ja", current_page="pi")

    assert "Piブリッジ状態" in html
    assert "Pi → Jetson 状態" in html
    assert "Piブリッジ設定" in html


def test_pi_settings_page_structure():
    """Pi Settings page includes expected API references."""
    html = render_pi_settings(lang="en")

    assert "/pi/status" in html
    assert "/pi/config" in html
    assert "/i2s/peer-status" in html
    assert "/pi/actions/restart" in html
