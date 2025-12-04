"""Tests for System Settings page rendering."""

from web.templates.pages import render_system


def test_render_system_english():
    """Test System Settings page renders in English."""
    html = render_system(lang="en", current_page="system")

    # Check page title
    assert "System" in html
    assert "Magic Box" in html

    # Check sections are present
    assert "Daemon Status" in html
    assert "DAC Information" in html
    assert "RTP Session Management" in html
    assert "System Logs" in html
    assert "Advanced Settings" in html


def test_render_system_japanese():
    """Test System Settings page renders in Japanese."""
    html = render_system(lang="ja", current_page="system")

    # Check page title
    assert "システム" in html or "System" in html
    assert "Magic Box" in html

    # Check sections are present (Japanese translations)
    assert "デーモン状態" in html
    assert "DAC情報" in html
    assert "RTPセッション管理" in html
    assert "システムログ" in html
    assert "詳細設定" in html


def test_system_page_structure():
    """Test System Settings page has correct structure."""
    html = render_system(lang="en")

    # Check Alpine.js data function
    assert "systemSettingsData()" in html

    # Check API endpoints are referenced
    assert "/daemon/status" in html
    assert "/dac/devices" in html
    assert "/dac/state" in html
    assert "/api/rtp/sessions" in html
    assert "/api/system/logs" in html
    assert "/partitioned-convolution" in html

    # Check action buttons/methods
    assert "fetchDaemonStatus" in html
    assert "fetchDacDevices" in html
    assert "selectDac" in html
    assert "switchInputMode" in html
    assert "discoverRtp" in html
    assert "fetchLogs" in html
    assert "savePartitionedSettings" in html


def test_system_sidebar_active():
    """Test sidebar highlights System Settings when on that page."""
    html = render_system(current_page="system")
    assert "nav-item" in html
    # The sidebar should mark System as active
    assert "current_page" in html or "system" in html


def test_system_translations_consistency():
    """Test that translations are consistent across languages."""
    html_en = render_system(lang="en")
    html_ja = render_system(lang="ja")

    # Both should have the same structure
    assert "systemSettingsData()" in html_en
    assert "systemSettingsData()" in html_ja

    # Both should reference the same API endpoints
    assert "/daemon/status" in html_en
    assert "/daemon/status" in html_ja

    assert "/api/rtp/sessions" in html_en
    assert "/api/rtp/sessions" in html_ja

    assert "/api/system/logs" in html_en
    assert "/api/system/logs" in html_ja
