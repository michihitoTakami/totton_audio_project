"""Tests for System Settings page rendering."""

from web.templates.pages import render_system


def test_render_system_english():
    """System page renders with main sections in English."""
    html = render_system(lang="en", current_page="system")

    assert "System Health" in html
    assert "Daemon" in html
    assert "DAC" in html
    assert "Partitioned Convolution" in html
    assert "Buffer" in html


def test_render_system_japanese():
    """System page renders with Japanese translations."""
    html = render_system(lang="ja", current_page="system")

    assert "システムヘルス" in html
    assert "デーモン" in html
    assert "DAC" in html
    assert "パーティションド畳み込み" in html
    assert "バッファ" in html


def test_system_page_structure():
    """System page includes expected API references."""
    html = render_system(lang="en")

    assert "/status" in html
    assert "/daemon/status" in html
    assert "/daemon/restart" in html
    assert "/dac/devices" in html
    assert "/dac/state" in html
    assert "/dac/rescan" in html
    assert "/dac/select" in html
    assert "/partitioned-convolution" in html


def test_language_switch_controls_present():
    """Language switch UI is rendered with links."""
    html = render_system(lang="en", current_page="system")

    assert "Language" in html
    assert "English" in html
    assert "Japanese" in html


def test_language_switch_translated_to_japanese():
    """Language selector uses Japanese labels when lang=ja."""
    html = render_system(lang="ja", current_page="system")

    assert "言語設定" in html
    assert "英語" in html
    assert "日本語" in html
