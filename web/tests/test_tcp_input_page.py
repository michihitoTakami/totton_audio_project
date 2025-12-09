"""Tests for TCP Input UI page rendering and routing."""

from fastapi.testclient import TestClient

from web.main import app
from web.templates.pages import render_tcp_input


def test_render_tcp_input_english():
    """TCP Input page renders expected sections in English."""
    html = render_tcp_input(lang="en", current_page="tcp-input")

    assert "TCP Input" in html
    assert "Connection Status" in html
    assert "Stream Format" in html
    assert "Performance Metrics" in html
    assert "TCP Input Settings" in html
    assert "tcpInputPage()" in html


def test_render_tcp_input_japanese():
    """TCP Input page renders expected sections in Japanese."""
    html = render_tcp_input(lang="ja", current_page="tcp-input")

    assert "TCP入力" in html
    assert "接続ステータス" in html
    assert "ストリームフォーマット" in html
    assert "パフォーマンスメトリクス" in html
    assert "TCP入力設定" in html


def test_tcp_input_route_serves_page():
    """FastAPI route should serve the TCP Input page with scripts."""
    client = TestClient(app)
    response = client.get("/tcp-input")

    assert response.status_code == 200
    assert '<script src="/static/js/tcp_input.js"></script>' in response.text
    assert "tcpInputPage()" in response.text
