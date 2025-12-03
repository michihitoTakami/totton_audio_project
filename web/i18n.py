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
        # Dashboard page
        "dashboard.subtitle": "System Overview & Input Mode Switching",
        "dashboard.daemon": "Daemon",
        "dashboard.input_mode": "Input Mode",
        "dashboard.eq": "EQ",
        "dashboard.sample_rate": "Sample Rate",
        "dashboard.input": "Input",
        "dashboard.output": "Output",
        "dashboard.status.running": "Running",
        "dashboard.status.stopped": "Stopped",
        "dashboard.status.on": "ON",
        "dashboard.status.off": "OFF",
        "dashboard.mode_switch.title": "Input Mode Switching",
        "dashboard.mode_switch.pipewire": "PipeWire",
        "dashboard.mode_switch.pipewire_desc": "Local input (minimum latency)",
        "dashboard.mode_switch.rtp": "RTP",
        "dashboard.mode_switch.rtp_desc": "Network input",
        "dashboard.mode_switch.switching": "Switching mode...",
        "dashboard.quick_actions": "Quick Actions",
        "dashboard.action.restart_daemon": "Restart Daemon",
        "dashboard.action.eq_settings": "EQ Settings",
        "dashboard.action.crossfeed": "Crossfeed Settings",
        "dashboard.action.rtp_management": "RTP Management",
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
        # Dashboard page
        "dashboard.subtitle": "システム概要と入力モード切替",
        "dashboard.daemon": "デーモン",
        "dashboard.input_mode": "入力モード",
        "dashboard.eq": "EQ",
        "dashboard.sample_rate": "サンプルレート",
        "dashboard.input": "Input",
        "dashboard.output": "Output",
        "dashboard.status.running": "Running",
        "dashboard.status.stopped": "Stopped",
        "dashboard.status.on": "ON",
        "dashboard.status.off": "OFF",
        "dashboard.mode_switch.title": "入力モード切替",
        "dashboard.mode_switch.pipewire": "PipeWire",
        "dashboard.mode_switch.pipewire_desc": "ローカル入力（最小遅延）",
        "dashboard.mode_switch.rtp": "RTP",
        "dashboard.mode_switch.rtp_desc": "ネットワーク入力",
        "dashboard.mode_switch.switching": "モード切り替え中...",
        "dashboard.quick_actions": "クイックアクション",
        "dashboard.action.restart_daemon": "デーモン再起動",
        "dashboard.action.eq_settings": "EQ設定",
        "dashboard.action.crossfeed": "クロスフィード設定",
        "dashboard.action.rtp_management": "RTP管理",
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
