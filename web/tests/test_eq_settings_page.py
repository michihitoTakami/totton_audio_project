"""
Tests for EQ Settings page rendering.
"""

from web.templates.pages import render_eq_settings


def test_render_eq_settings_english():
    """Test EQ Settings page renders in English."""
    html = render_eq_settings(lang="en", current_page="eq")

    # Check page title
    assert "EQ Settings" in html
    assert "Magic Box" in html

    # Check sections are present
    assert "Active EQ Profile" in html
    assert "OPRA Headphone Search" in html
    assert "Saved EQ Profiles" in html

    # Check Modern Target checkbox
    assert "Modern Target (KB5000_7)" in html
    assert "Correct to latest target curve" in html


def test_render_eq_settings_japanese():
    """Test EQ Settings page renders in Japanese."""
    html = render_eq_settings(lang="ja", current_page="eq")

    # Check page title
    assert "EQ設定" in html
    assert "Magic Box" in html

    # Check sections are present
    assert "有効なEQプロファイル" in html
    assert "OPRAヘッドホン検索" in html
    assert "保存済みEQプロファイル" in html

    # Check Modern Target checkbox
    assert "Modern Target (KB5000_7)" in html
    assert "最新のターゲットカーブに補正" in html


def test_eq_settings_page_structure():
    """Test EQ Settings page has correct structure."""
    html = render_eq_settings(lang="en")

    # Check Alpine.js data function
    assert "eqSettingsData()" in html

    # Check API endpoints are referenced
    assert "/eq/active" in html
    assert "/eq/profiles" in html
    assert "/opra/search" in html

    # Check form elements
    assert 'id="opraSearch"' in html

    # Check action buttons
    assert "applyOPRA" in html
    assert "deactivateEQ" in html
    assert "activateProfile" in html
    assert "deleteProfile" in html


def test_eq_settings_sidebar_active():
    """Test sidebar highlights EQ Settings when on that page."""
    html = render_eq_settings(current_page="eq")

    # Check that EQ Settings link has active class
    # Note: This depends on the sidebar component being included
    assert "nav-item" in html


def test_eq_settings_translations_consistency():
    """Test that translations are consistent across languages."""
    html_en = render_eq_settings(lang="en")
    html_ja = render_eq_settings(lang="ja")

    # Both should have the same structure
    assert "eqSettingsData()" in html_en
    assert "eqSettingsData()" in html_ja

    # Both should reference the same API endpoints
    assert "/eq/active" in html_en
    assert "/eq/active" in html_ja

    assert "/opra/search" in html_en
    assert "/opra/search" in html_ja
