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
        "nav.system": "System",
        "common.coming_soon": "Coming Soon",
        # Dashboard page
        "dashboard.subtitle": "System Overview",
        "dashboard.daemon": "Daemon",
        "dashboard.eq": "EQ",
        "dashboard.sample_rate": "Sample Rate",
        "dashboard.input": "Input",
        "dashboard.output": "Output",
        "dashboard.status.running": "Running",
        "dashboard.status.stopped": "Stopped",
        "dashboard.status.on": "ON",
        "dashboard.status.off": "OFF",
        "dashboard.quick_actions": "Quick Actions",
        "dashboard.action.restart_daemon": "Restart Daemon",
        "dashboard.action.eq_settings": "EQ Settings",
        "dashboard.action.crossfeed": "Crossfeed Settings",
        # Dashboard cards
        "dashboard.crossfeed": "Crossfeed",
        "dashboard.low_latency": "Low Latency Mode",
        # Low Latency Mode section
        "dashboard.low_latency.title": "Low Latency Mode (Partitioned Convolution)",
        "dashboard.low_latency.toggle": "Low Latency Partition",
        "dashboard.low_latency.exclusive": "Exclusive with crossfeed",
        "dashboard.low_latency.warning": "Crossfeed is not available in low latency mode.",
        "dashboard.low_latency.info": "Ultra-low latency with 32k-tap fast partition.",
        "dashboard.output_mode.title": "Output Mode",
        "dashboard.output_mode.subtitle": "Select pipeline mode and preferred ALSA device",
        "dashboard.output_mode.mode_label": "Mode",
        "dashboard.output_mode.device_label": "Preferred ALSA Device",
        "dashboard.output_mode.save": "Save Output Mode",
        "dashboard.output_mode.success": "Output mode updated",
        "dashboard.output_mode.error": "Failed to update output mode",
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
        # EQ Settings page
        "eq.subtitle": "Manage Headphone EQ Profiles & Phase Settings",
        "eq.active.title": "Active EQ Profile",
        "eq.active.source": "Source",
        "eq.active.modern_target": "KB5000_7 Applied",
        "eq.active.deactivate": "Deactivate",
        "eq.active.none": "No EQ profile is currently active.",
        "eq.active.filters_title": "Active Filters",
        "eq.phase_type.title": "Phase Type",
        "eq.phase_type.label": "Filter Phase",
        "eq.phase_type.minimum": "Minimum Phase (Recommended)",
        "eq.phase_type.linear": "Linear Phase (Full-band linear)",
        "eq.phase_type.info_minimum": "Full-band minimum phase processing (minimum latency)",
        "eq.phase_type.info_linear": "Full-band linear phase with constant group delay (~0.45s latency @ 705.6kHz)",
        "eq.opra.title": "OPRA Headphone Search",
        "eq.opra.search_label": "Search Headphones",
        "eq.opra.search_placeholder": "e.g. HD650, DT770, AirPods...",
        "eq.opra.variant": "EQ Variation",
        "eq.opra.modern_target": "Modern Target (KB5000_7)",
        "eq.opra.modern_target_desc": "Correct to latest target curve",
        "eq.opra.apply": "Apply EQ",
        "eq.opra.license": "EQ data:",
        "eq.opra.license_link": "OPRA Project",
        "eq.import.title": "Import EQ from Text",
        "eq.import.name_label": "Profile Name",
        "eq.import.name_placeholder": "e.g. My Custom EQ",
        "eq.import.content_label": "EQ Content (Equalizer APO format)",
        "eq.import.content_placeholder": "Paste your EQ settings here...\nFilter 1: ON PK Fc 105 Hz Gain -3.9 dB Q 0.7\nFilter 2: ON PK Fc 200 Hz Gain 5.4 dB Q 1.2",
        "eq.import.import_button": "Import EQ",
        "eq.profiles.title": "Saved EQ Profiles",
        "eq.profiles.activate": "Activate",
        "eq.profiles.active": "Active",
        "eq.profiles.delete": "Delete",
        "eq.profiles.empty": "No saved EQ profiles.",
        "common.units.ms": "ms",
        "common.units.khz": "kHz",
    },
    "ja": {
        "app.title": "Magic Box",
        "app.tagline": "究極のオーディオ体験",
        "nav.dashboard": "ダッシュボード",
        "nav.eq": "EQ設定",
        "nav.crossfeed": "クロスフィード",
        "nav.system": "システム",
        "common.coming_soon": "近日公開",
        # Dashboard page
        "dashboard.subtitle": "システム概要",
        "dashboard.daemon": "デーモン",
        "dashboard.eq": "EQ",
        "dashboard.sample_rate": "サンプルレート",
        "dashboard.input": "Input",
        "dashboard.output": "Output",
        "dashboard.status.running": "Running",
        "dashboard.status.stopped": "Stopped",
        "dashboard.status.on": "ON",
        "dashboard.status.off": "OFF",
        "dashboard.quick_actions": "クイックアクション",
        "dashboard.action.restart_daemon": "デーモン再起動",
        "dashboard.action.eq_settings": "EQ設定",
        "dashboard.action.crossfeed": "クロスフィード設定",
        # Dashboard cards
        "dashboard.crossfeed": "クロスフィード",
        "dashboard.low_latency": "低遅延モード",
        # Low Latency Mode section
        "dashboard.low_latency.title": "⚡ 低遅延モード (Partitioned Convolution)",
        "dashboard.low_latency.toggle": "低遅延パーティション",
        "dashboard.low_latency.exclusive": "クロスフィードと排他利用",
        "dashboard.low_latency.warning": "⚠️ クロスフィードは低遅延モードでは利用できません。",
        "dashboard.low_latency.info": "32kタップの高速パーティションで超低遅延を実現します。",
        "dashboard.output_mode.title": "🔊 出力モード",
        "dashboard.output_mode.subtitle": "出力パイプラインと優先ALSAデバイスを設定",
        "dashboard.output_mode.mode_label": "モード",
        "dashboard.output_mode.device_label": "優先ALSAデバイス",
        "dashboard.output_mode.save": "出力モードを保存",
        "dashboard.output_mode.success": "出力モードを更新しました",
        "dashboard.output_mode.error": "出力モードの更新に失敗しました",
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
        # EQ Settings page
        "eq.subtitle": "ヘッドホンEQプロファイルと位相設定を管理",
        "eq.active.title": "有効なEQプロファイル",
        "eq.active.source": "ソース",
        "eq.active.modern_target": "KB5000_7適用済み",
        "eq.active.deactivate": "無効化",
        "eq.active.none": "EQプロファイルは現在有効化されていません。",
        "eq.active.filters_title": "適用中のフィルター",
        "eq.phase_type.title": "🌊 位相タイプ",
        "eq.phase_type.label": "Filter Phase",
        "eq.phase_type.minimum": "Minimum Phase (推奨)",
        "eq.phase_type.linear": "Linear Phase (全帯域線形)",
        "eq.phase_type.info_minimum": "全帯域を最小位相で処理（最小レイテンシ）",
        "eq.phase_type.info_linear": "全帯域で群遅延が一定（完全な位相直線性、約0.45秒のレイテンシ @ 705.6kHz）",
        "eq.opra.title": "🎚️ OPRAヘッドホン検索",
        "eq.opra.search_label": "Search Headphones",
        "eq.opra.search_placeholder": "e.g. HD650, DT770, AirPods...",
        "eq.opra.variant": "EQバリエーション",
        "eq.opra.modern_target": "Modern Target (KB5000_7)",
        "eq.opra.modern_target_desc": "最新のターゲットカーブに補正",
        "eq.opra.apply": "Apply EQ",
        "eq.opra.license": "EQ data:",
        "eq.opra.license_link": "OPRA Project",
        "eq.import.title": "テキストからEQをインポート",
        "eq.import.name_label": "プロファイル名",
        "eq.import.name_placeholder": "例: カスタムEQ",
        "eq.import.content_label": "EQ内容 (Equalizer APO形式)",
        "eq.import.content_placeholder": "EQ設定をここに貼り付けてください...\nFilter 1: ON PK Fc 105 Hz Gain -3.9 dB Q 0.7\nFilter 2: ON PK Fc 200 Hz Gain 5.4 dB Q 1.2",
        "eq.import.import_button": "EQをインポート",
        "eq.profiles.title": "保存済みEQプロファイル",
        "eq.profiles.activate": "有効化",
        "eq.profiles.active": "アクティブ",
        "eq.profiles.delete": "削除",
        "eq.profiles.empty": "保存済みのEQプロファイルがありません。",
        "common.units.ms": "ms",
        "common.units.khz": "kHz",
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
