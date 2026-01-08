"""
Pages module for Totton Audio Project Web UI.

This module provides template rendering functions for each page.
"""

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

from ..i18n import get_translations, normalize_lang


# Initialize Jinja2 environment
template_dir = Path(__file__).parent
_delimiter_template_path = template_dir / "pages" / "delimiter.html"
_delimiter_ui_available = _delimiter_template_path.exists()
env = Environment(
    loader=FileSystemLoader(str(template_dir)),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_dashboard(lang: str = "en", current_page: str = "dashboard") -> str:
    """
    Render the Dashboard page.

    Args:
        lang: Language code ("en" or "ja")
        current_page: Current page name for sidebar highlighting

    Returns:
        Rendered HTML string
    """
    normalized_lang = normalize_lang(lang)
    template = env.get_template("pages/dashboard.html")
    return template.render(
        t=get_translations(normalized_lang),
        current_page=current_page,
        lang=normalized_lang,
        enable_delimiter_ui=_delimiter_ui_available,
    )


def render_eq_settings(lang: str = "en", current_page: str = "eq") -> str:
    """
    Render the EQ Settings page.

    Args:
        lang: Language code ("en" or "ja")
        current_page: Current page name for sidebar highlighting

    Returns:
        Rendered HTML string
    """
    normalized_lang = normalize_lang(lang)
    template = env.get_template("pages/eq_settings.html")
    return template.render(
        t=get_translations(normalized_lang),
        current_page=current_page,
        lang=normalized_lang,
        enable_delimiter_ui=_delimiter_ui_available,
    )


def render_system(lang: str = "en", current_page: str = "system") -> str:
    """
    Render the System page.

    Args:
        lang: Language code ("en" or "ja")
        current_page: Current page name for sidebar highlighting

    Returns:
        Rendered HTML string
    """
    normalized_lang = normalize_lang(lang)
    template = env.get_template("pages/system_settings.html")
    return template.render(
        t=get_translations(normalized_lang),
        current_page=current_page,
        lang=normalized_lang,
        enable_delimiter_ui=_delimiter_ui_available,
    )


def is_delimiter_ui_available() -> bool:
    """delimiter UIテンプレが利用可能か（public exportで削除される場合がある）."""
    return _delimiter_ui_available


def render_delimiter(lang: str = "en", current_page: str = "delimiter") -> str:
    """
    Render the De-limiter page.

    Args:
        lang: Language code ("en" or "ja")
        current_page: Current page name for sidebar highlighting

    Returns:
        Rendered HTML string
    """
    if not _delimiter_ui_available:
        raise FileNotFoundError("delimiter.html is not available")
    normalized_lang = normalize_lang(lang)
    template = env.get_template("pages/delimiter.html")
    return template.render(
        t=get_translations(normalized_lang),
        current_page=current_page,
        lang=normalized_lang,
        enable_delimiter_ui=_delimiter_ui_available,
    )


def render_pi_settings(lang: str = "en", current_page: str = "pi") -> str:
    """
    Render the Pi Settings page.

    Args:
        lang: Language code ("en" or "ja")
        current_page: Current page name for sidebar highlighting

    Returns:
        Rendered HTML string
    """
    normalized_lang = normalize_lang(lang)
    template = env.get_template("pages/pi_settings.html")
    return template.render(
        t=get_translations(normalized_lang),
        current_page=current_page,
        lang=normalized_lang,
        enable_delimiter_ui=_delimiter_ui_available,
    )
