"""Unit tests for i18n module."""

from web.i18n import TRANSLATIONS, get_text, get_translations, normalize_lang


class TestGetText:
    """Test get_text function."""

    def test_get_text_english_app_title(self):
        """Test getting app title in English."""
        assert get_text("app.title", "en") == "Magic Box"

    def test_get_text_japanese_app_title(self):
        """Test getting app title in Japanese."""
        assert get_text("app.title", "ja") == "Magic Box"

    def test_get_text_english_tagline(self):
        """Test getting tagline in English."""
        assert get_text("app.tagline", "en") == "Ultimate Audio Experience"

    def test_get_text_japanese_tagline(self):
        """Test getting tagline in Japanese."""
        assert get_text("app.tagline", "ja") == "究極のオーディオ体験"

    def test_get_text_default_language_english(self):
        """Test default language is English."""
        assert get_text("app.title") == "Magic Box"

    def test_get_text_missing_key_returns_key(self):
        """Test that missing keys return the key itself."""
        assert get_text("nonexistent.key", "en") == "nonexistent.key"
        assert get_text("another.missing.key", "ja") == "another.missing.key"

    def test_get_text_invalid_language_falls_back_to_english(self):
        """Test that invalid language codes fall back to English."""
        assert get_text("app.title", "invalid_lang") == "Magic Box"


class TestGetTranslations:
    """Test get_translations function."""

    def test_get_translations_english(self):
        """Test getting all English translations."""
        translations = get_translations("en")
        assert isinstance(translations, dict)
        assert translations["app.title"] == "Magic Box"
        assert "nav.dashboard" in translations

    def test_get_translations_japanese(self):
        """Test getting all Japanese translations."""
        translations = get_translations("ja")
        assert isinstance(translations, dict)
        assert translations["app.title"] == "Magic Box"
        assert translations["app.tagline"] == "究極のオーディオ体験"

    def test_get_translations_default_english(self):
        """Test default language is English."""
        translations = get_translations()
        assert translations["app.title"] == "Magic Box"

    def test_get_translations_invalid_language_returns_english(self):
        """Test invalid language returns English translations."""
        translations = get_translations("invalid")
        assert translations == TRANSLATIONS["en"]


class TestDashboardTranslations:
    """Test dashboard-specific translation keys."""

    def test_dashboard_status_keys_exist_in_english(self):
        """Test all dashboard status keys exist in English."""
        keys = [
            "dashboard.subtitle",
            "dashboard.daemon",
            "dashboard.eq",
            "dashboard.sample_rate",
            "dashboard.status.running",
            "dashboard.status.stopped",
            "dashboard.status.on",
            "dashboard.status.off",
        ]
        for key in keys:
            assert key in TRANSLATIONS["en"], f"Missing key: {key}"
            assert isinstance(TRANSLATIONS["en"][key], str)

    def test_dashboard_status_keys_exist_in_japanese(self):
        """Test all dashboard status keys exist in Japanese."""
        keys = [
            "dashboard.subtitle",
            "dashboard.daemon",
            "dashboard.eq",
            "dashboard.sample_rate",
            "dashboard.status.running",
            "dashboard.status.stopped",
            "dashboard.status.on",
            "dashboard.status.off",
        ]
        for key in keys:
            assert key in TRANSLATIONS["ja"], f"Missing key: {key}"
            assert isinstance(TRANSLATIONS["ja"][key], str)

    def test_low_latency_mode_keys_exist(self):
        """Test low latency mode keys exist in both languages."""
        keys = [
            "dashboard.low_latency.title",
            "dashboard.low_latency.toggle",
            "dashboard.low_latency.exclusive",
            "dashboard.low_latency.warning",
            "dashboard.low_latency.info",
        ]
        for key in keys:
            assert key in TRANSLATIONS["en"], f"Missing EN key: {key}"
            assert key in TRANSLATIONS["ja"], f"Missing JA key: {key}"

    def test_phase_type_keys_exist(self):
        """Test phase type keys exist in both languages."""
        keys = [
            "dashboard.phase_type.title",
            "dashboard.phase_type.label",
            "dashboard.phase_type.minimum",
            "dashboard.phase_type.linear",
            "dashboard.phase_type.info_minimum",
            "dashboard.phase_type.info_linear",
            "dashboard.phase_type.warning",
        ]
        for key in keys:
            assert key in TRANSLATIONS["en"], f"Missing EN key: {key}"
            assert key in TRANSLATIONS["ja"], f"Missing JA key: {key}"

    def test_eq_section_keys_exist(self):
        """Test EQ section keys exist in both languages."""
        keys = [
            "dashboard.eq.title",
            "dashboard.eq.search_label",
            "dashboard.eq.search_placeholder",
            "dashboard.eq.variant",
            "dashboard.eq.modern_target",
            "dashboard.eq.modern_target_desc",
            "dashboard.eq.apply",
            "dashboard.eq.off",
            "dashboard.eq.license",
            "dashboard.eq.license_link",
        ]
        for key in keys:
            assert key in TRANSLATIONS["en"], f"Missing EN key: {key}"
            assert key in TRANSLATIONS["ja"], f"Missing JA key: {key}"

    def test_crossfeed_section_keys_exist(self):
        """Test crossfeed section keys exist in both languages."""
        keys = [
            "dashboard.crossfeed.title",
            "dashboard.crossfeed.toggle",
            "dashboard.crossfeed.toggle_desc",
            "dashboard.crossfeed.head_size",
            "dashboard.crossfeed.warning",
            "dashboard.crossfeed.info",
            "dashboard.crossfeed.license",
            "dashboard.crossfeed.license_link",
        ]
        for key in keys:
            assert key in TRANSLATIONS["en"], f"Missing EN key: {key}"
            assert key in TRANSLATIONS["ja"], f"Missing JA key: {key}"

    def test_output_mode_keys_exist(self):
        """Test output mode section keys exist in both languages."""
        keys = [
            "dashboard.output_mode.title",
            "dashboard.output_mode.subtitle",
            "dashboard.output_mode.mode_label",
            "dashboard.output_mode.device_label",
            "dashboard.output_mode.save",
            "dashboard.output_mode.success",
            "dashboard.output_mode.error",
        ]
        for key in keys:
            assert key in TRANSLATIONS["en"], f"Missing EN key: {key}"
            assert key in TRANSLATIONS["ja"], f"Missing JA key: {key}"


class TestTranslationConsistency:
    """Test consistency between English and Japanese translations."""

    def test_same_keys_in_both_languages(self):
        """Test that English and Japanese have the same set of keys."""
        en_keys = set(TRANSLATIONS["en"].keys())
        ja_keys = set(TRANSLATIONS["ja"].keys())

        missing_in_ja = en_keys - ja_keys
        missing_in_en = ja_keys - en_keys

        assert not missing_in_ja, f"Keys missing in JA: {missing_in_ja}"
        assert not missing_in_en, f"Keys missing in EN: {missing_in_en}"

    def test_no_empty_translations(self):
        """Test that no translations are empty strings."""
        for lang in ["en", "ja"]:
            for key, value in TRANSLATIONS[lang].items():
                assert value.strip() != "", f"Empty translation for {lang}.{key}"


class TestNormalizeLang:
    """Tests for language normalization helper."""

    def test_normalize_lang_handles_none(self):
        """None should default to English."""
        assert normalize_lang(None) == "en"

    def test_normalize_lang_lowercases_codes(self):
        """Uppercase language codes should be normalized."""
        assert normalize_lang("JA") == "ja"

    def test_normalize_lang_invalid_defaults_to_english(self):
        """Unsupported language codes should fall back to English."""
        assert normalize_lang("fr") == "en"


class TestLanguageSwitchKeys:
    """Ensure language switch translation keys exist."""

    def test_language_keys_exist_in_both_locales(self):
        """Language switch strings must be defined for all locales."""
        keys = [
            "system.language.title",
            "system.language.description",
            "system.language.english",
            "system.language.japanese",
        ]
        for key in keys:
            assert key in TRANSLATIONS["en"], f"Missing EN key: {key}"
            assert key in TRANSLATIONS["ja"], f"Missing JA key: {key}"
