"""
i18n (Internationalization) module for Magic Box Web UI.

Basic structure for server-side i18n using Python dictionaries and Jinja2.
Actual language switching functionality will be implemented in Issue #415.
"""

TRANSLATIONS = {
    "en": {
        "app.title": "Magic Box",
        "app.tagline": "Ultimate Audio Experience",
        "nav.dashboard": "Dashboard",
        "nav.eq": "EQ Settings",
        "nav.crossfeed": "Crossfeed",
        "nav.rtp": "RTP Management",
        "nav.system": "System",
        "common.coming_soon": "Coming Soon",
    },
    "ja": {
        "app.title": "Magic Box",
        "app.tagline": "究極のオーディオ体験",
        "nav.dashboard": "ダッシュボード",
        "nav.eq": "EQ設定",
        "nav.crossfeed": "クロスフィード",
        "nav.rtp": "RTP管理",
        "nav.system": "システム",
        "common.coming_soon": "近日公開",
    },
}


def get_text(key: str, lang: str = "en") -> str:
    """
    Get translated text by key.

    Args:
        key: Translation key (e.g., "app.title")
        lang: Language code ("en" or "ja")

    Returns:
        Translated string, or the key itself if not found
    """
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


def get_translations(lang: str = "en") -> dict:
    """
    Get all translations for a specific language.

    Args:
        lang: Language code ("en" or "ja")

    Returns:
        Dictionary of translations
    """
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"])
