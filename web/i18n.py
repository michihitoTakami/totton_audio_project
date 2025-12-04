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
        # Dashboard cards
        "dashboard.crossfeed": "Crossfeed",
        "dashboard.low_latency": "Low Latency Mode",
        # Low Latency Mode section
        "dashboard.low_latency.title": "Low Latency Mode (Partitioned Convolution)",
        "dashboard.low_latency.toggle": "Low Latency Partition",
        "dashboard.low_latency.exclusive": "Exclusive with crossfeed",
        "dashboard.low_latency.warning": "Crossfeed is not available in low latency mode.",
        "dashboard.low_latency.info": "Ultra-low latency with 32k-tap fast partition.",
        # Phase Type section
        "dashboard.phase_type.title": "Phase Type",
        "dashboard.phase_type.label": "Filter Phase",
        "dashboard.phase_type.minimum": "Minimum Phase (Recommended)",
        "dashboard.phase_type.linear": "Linear Phase (Full-band linear)",
        "dashboard.phase_type.info_minimum": "Full-band minimum phase processing (minimum latency)",
        "dashboard.phase_type.info_linear": "Full-band linear phase with constant group delay (~0.45s latency @ 705.6kHz)",
        "dashboard.phase_type.warning": "Linear phase adds ~0.45s latency (@ 705.6kHz) and is not compatible with low latency mode.",
        # EQ section
        "dashboard.eq.title": "Headphone EQ (OPRA)",
        "dashboard.eq.search_label": "Search Headphones",
        "dashboard.eq.search_placeholder": "e.g. HD650, DT770, AirPods...",
        "dashboard.eq.variant": "EQ Variation",
        "dashboard.eq.modern_target": "Modern Target (KB5000_7)",
        "dashboard.eq.modern_target_desc": "Correct to latest target curve",
        "dashboard.eq.apply": "Apply EQ",
        "dashboard.eq.off": "EQ Off",
        "dashboard.eq.license": "EQ data:",
        "dashboard.eq.license_link": "OPRA Project",
        # Crossfeed section
        "dashboard.crossfeed.title": "Crossfeed (HRTF)",
        "dashboard.crossfeed.toggle": "Crossfeed",
        "dashboard.crossfeed.toggle_desc": "Reproduce speaker listening",
        "dashboard.crossfeed.head_size": "Head Size:",
        "dashboard.crossfeed.warning": "Crossfeed is not compatible with low latency mode.",
        "dashboard.crossfeed.info": "Reproduces speaker listening with equilateral triangle placement (±30°)",
        "dashboard.crossfeed.license": "HRTF data:",
        "dashboard.crossfeed.license_link": "HUTUBS, TU Berlin",
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
        # Dashboard cards
        "dashboard.crossfeed": "クロスフィード",
        "dashboard.low_latency": "低遅延モード",
        # Low Latency Mode section
        "dashboard.low_latency.title": "⚡ 低遅延モード (Partitioned Convolution)",
        "dashboard.low_latency.toggle": "低遅延パーティション",
        "dashboard.low_latency.exclusive": "クロスフィードと排他利用",
        "dashboard.low_latency.warning": "⚠️ クロスフィードは低遅延モードでは利用できません。",
        "dashboard.low_latency.info": "32kタップの高速パーティションで超低遅延を実現します。",
        # Phase Type section
        "dashboard.phase_type.title": "🌊 位相タイプ",
        "dashboard.phase_type.label": "Filter Phase",
        "dashboard.phase_type.minimum": "Minimum Phase (推奨)",
        "dashboard.phase_type.linear": "Linear Phase (全帯域線形)",
        "dashboard.phase_type.info_minimum": "全帯域を最小位相で処理（最小レイテンシ）",
        "dashboard.phase_type.info_linear": "全帯域で群遅延が一定（完全な位相直線性、約0.45秒のレイテンシ @ 705.6kHz）",
        "dashboard.phase_type.warning": "⚠️ 線形位相は約0.45秒のレイテンシが発生し（@ 705.6kHz）、低遅延モードとは併用できません。",
        # EQ section
        "dashboard.eq.title": "🎚️ ヘッドホンEQ (OPRA)",
        "dashboard.eq.search_label": "Search Headphones",
        "dashboard.eq.search_placeholder": "e.g. HD650, DT770, AirPods...",
        "dashboard.eq.variant": "EQバリエーション",
        "dashboard.eq.modern_target": "Modern Target (KB5000_7)",
        "dashboard.eq.modern_target_desc": "最新のターゲットカーブに補正",
        "dashboard.eq.apply": "Apply EQ",
        "dashboard.eq.off": "EQ Off",
        "dashboard.eq.license": "EQ data:",
        "dashboard.eq.license_link": "OPRA Project",
        # Crossfeed section
        "dashboard.crossfeed.title": "🎧 クロスフィード (HRTF)",
        "dashboard.crossfeed.toggle": "クロスフィード",
        "dashboard.crossfeed.toggle_desc": "スピーカーリスニングを再現",
        "dashboard.crossfeed.head_size": "頭のサイズ:",
        "dashboard.crossfeed.warning": "⚠️ クロスフィードは低遅延モードと併用できません。",
        "dashboard.crossfeed.info": "正三角形配置（±30°）でスピーカーリスニングを再現",
        "dashboard.crossfeed.license": "HRTF data:",
        "dashboard.crossfeed.license_link": "HUTUBS, TU Berlin",
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
