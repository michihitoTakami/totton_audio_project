"""レスポンシブ最終調整の回帰テスト."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.main import app


@pytest.fixture
def client():
    """FastAPIテストクライアントを提供."""
    return TestClient(app)


def test_mobile_toggle_has_accessibility_attrs(client):
    """モバイルメニューのアクセシビリティ属性を検証."""
    html = client.get("/").text

    assert 'class="mobile-menu-toggle"' in html
    assert 'data-testid="mobile-menu-toggle"' in html
    assert 'aria-controls="primary-sidebar"' in html
    assert ':aria-expanded="mobileMenuOpen"' in html


def test_menu_overlay_present(client):
    """メニューオーバーレイが描画されることを確認."""
    html = client.get("/").text

    assert 'class="menu-overlay"' in html
    assert 'data-testid="menu-overlay"' in html


def test_sidebar_navigation_landmark(client):
    """サイドバーのナビゲーションランドマークを確認."""
    html = client.get("/").text

    assert 'id="primary-sidebar"' in html
    assert 'role="navigation"' in html
    assert 'x-ref="firstNavLink"' in html
    assert 'aria-current="page"' in html


def test_responsive_breakpoints_present():
    """ブレークポイントとフォーカススタイルが保持されていることを確認."""
    css_path = Path(__file__).parent.parent / "static" / "css" / "main.css"
    content = css_path.read_text()

    assert "@media (max-width: 1024px)" in content
    assert "@media (max-width: 1100px)" in content
    assert "@media (min-width: 1200px)" in content
    assert "--sidebar-width" in content
    assert ".nav-item:focus-visible" in content
