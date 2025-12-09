"""レスポンシブ最終調整の回帰テスト."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.main import app
from web.i18n import normalize_lang


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


def test_language_persists_via_cookie(client):
    """langクエリ指定時にCookieへ永続化されることを確認."""
    res = client.get("/?lang=ja")
    assert res.cookies.get("lang") == "ja"

    # Subsequent request without query should reuse cookie
    res2 = client.get("/")
    assert 'lang="ja"' in res2.text or 'lang=\\"ja\\"' in res2.text


def test_language_falls_back_to_cookie_when_query_missing(client):
    """クエリなしでもCookieがあればその言語を使う."""
    # set cookie manually
    cookie_lang = normalize_lang("ja")
    res = client.get("/", cookies={"lang": cookie_lang})
    assert 'lang="ja"' in res.text or 'lang=\\"ja\\"' in res.text
