"""Tests for De-limiter page rendering."""

from web.templates.pages import render_delimiter


def test_render_delimiter_english():
    """Delimiter page renders with English labels."""
    html = render_delimiter(lang="en", current_page="delimiter")

    assert "AI Loudness Care" in html
    assert "loudness care" in html.lower()
    assert "queue" in html.lower()
    assert "/delimiter/status" in html
    assert "/delimiter/enable" in html
    assert "/delimiter/disable" in html


def test_render_delimiter_japanese():
    """Delimiter page renders with Japanese labels."""
    html = render_delimiter(lang="ja", current_page="delimiter")

    assert "AIラウドネスケア" in html
    assert "キュー" in html or "queue" in html.lower()


def test_delimiter_toggle_and_metrics_present():
    """UI includes toggle and telemetry fields."""
    html = render_delimiter(lang="en", current_page="delimiter")

    assert "delimiter-toggle" in html
    assert "queue_samples" in html
    assert "last_inference_ms" in html
    assert "backend_available" in html
    assert "fallback_reason" in html


def test_delimiter_latency_notice_present():
    """Latency notice should be rendered in both languages."""
    html_en = render_delimiter(lang="en", current_page="delimiter")
    html_ja = render_delimiter(lang="ja", current_page="delimiter")

    assert "Processes ~4s buffered chunks" in html_en
    assert "latency_notice" not in html_en  # key should not leak to markup
    assert "約4秒分のバッファ" in html_ja


def test_delimiter_backend_warning_only_when_enabled_template():
    """Warning panel should be tied to enabled && !backendAvailable condition."""
    html = render_delimiter(lang="en", current_page="delimiter")
    assert 'x-show="delimiter.enabled && !delimiter.backendAvailable"' in html
