"""
Pages module for Magic Box Web UI.

This module provides template rendering functions for each page.
"""

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

from ..i18n import get_translations


# Initialize Jinja2 environment
template_dir = Path(__file__).parent
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
    template = env.get_template("pages/dashboard.html")
    return template.render(
        t=get_translations(lang),
        current_page=current_page,
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
    template = env.get_template("pages/eq_settings.html")
    return template.render(
        t=get_translations(lang),
        current_page=current_page,
    )


def render_crossfeed(lang: str = "en", current_page: str = "crossfeed") -> str:
    """
    Render the Crossfeed page.

    Args:
        lang: Language code ("en" or "ja")
        current_page: Current page name for sidebar highlighting

    Returns:
        Rendered HTML string
    """
    # TODO: Implement in Issue #412
    template = env.get_template("base.html")
    return template.render(
        t=get_translations(lang),
        current_page=current_page,
        content="<h2>Crossfeed - Coming Soon</h2>",
    )


def render_rtp_management(lang: str = "en", current_page: str = "rtp") -> str:
    """
    Render the RTP Management page.

    Args:
        lang: Language code ("en" or "ja")
        current_page: Current page name for sidebar highlighting

    Returns:
        Rendered HTML string
    """
    # TODO: Implement in Issue #413
    template = env.get_template("base.html")
    return template.render(
        t=get_translations(lang),
        current_page=current_page,
        content="<h2>RTP Management - Coming Soon</h2>",
    )


def render_system(lang: str = "en", current_page: str = "system") -> str:
    """
    Render the System Settings page.

    Args:
        lang: Language code ("en" or "ja")
        current_page: Current page name for sidebar highlighting

    Returns:
        Rendered HTML string
    """
    template = env.get_template("pages/system_settings.html")
    return template.render(
        t=get_translations(lang),
        current_page=current_page,
    )
