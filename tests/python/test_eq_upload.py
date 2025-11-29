"""Tests for EQ profile upload security and validation."""

import sys
from pathlib import Path

# Add project root to path so we can import web as a package
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.constants import (  # noqa: E402
    FREQ_MAX_HZ,
    FREQ_MIN_HZ,
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    MAX_EQ_FILE_SIZE,
    MAX_EQ_FILTERS,
    PREAMP_MAX_DB,
    PREAMP_MIN_DB,
    Q_MAX,
    Q_MIN,
)
from web.services.eq import (  # noqa: E402
    sanitize_filename,
    validate_eq_profile_content,
)


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_valid_simple_filename(self):
        """Valid simple filename should pass."""
        assert sanitize_filename("my_profile.txt") == "my_profile.txt"

    def test_valid_with_numbers(self):
        """Filename with numbers should pass."""
        assert sanitize_filename("eq_profile_123.txt") == "eq_profile_123.txt"

    def test_valid_with_hyphen(self):
        """Filename with hyphen should pass."""
        assert sanitize_filename("my-eq-profile.txt") == "my-eq-profile.txt"

    def test_valid_with_dots(self):
        """Filename with multiple dots should pass."""
        assert sanitize_filename("profile.v2.txt") == "profile.v2.txt"

    def test_path_traversal_basic(self):
        """Path traversal with ../ - basename extracts safe filename."""
        # os.path.basename correctly extracts "passwd.txt" from "../etc/passwd.txt"
        # This is safe because only the basename is used for the destination path
        assert sanitize_filename("../etc/passwd.txt") == "passwd.txt"

    def test_path_traversal_encoded(self):
        """Path traversal even after basename extraction."""
        # os.path.basename handles this, but we double check
        result = sanitize_filename("..%2F..%2Fetc%2Fpasswd.txt")
        # Should be None because % is not allowed
        assert result is None

    def test_path_traversal_windows(self):
        """Windows-style path traversal - basename extracts safe filename."""
        # Backslashes are normalized to forward slashes, then basename extracted
        assert sanitize_filename("..\\..\\windows\\system32.txt") == "system32.txt"

    def test_absolute_path_unix(self):
        """Absolute Unix path should extract basename."""
        assert sanitize_filename("/etc/passwd.txt") == "passwd.txt"

    def test_absolute_path_windows(self):
        """Absolute Windows path - basename should be extracted."""
        # Backslashes are normalized, so basename is correctly extracted
        assert sanitize_filename("C:\\Windows\\system.txt") == "system.txt"

    def test_empty_filename(self):
        """Empty filename should be rejected."""
        assert sanitize_filename("") is None

    def test_none_filename(self):
        """None filename should be rejected."""
        assert sanitize_filename(None) is None

    def test_special_characters_rejected(self):
        """Special characters like spaces should be rejected."""
        assert sanitize_filename("my profile.txt") is None
        assert sanitize_filename("profile<script>.txt") is None
        assert sanitize_filename("profile;ls.txt") is None

    def test_wrong_extension(self):
        """Wrong file extension should be rejected."""
        assert sanitize_filename("profile.exe") is None
        assert sanitize_filename("profile.sh") is None
        assert sanitize_filename("profile") is None

    def test_double_extension_safe(self):
        """Double extension should be safe if ends with .txt."""
        assert sanitize_filename("profile.tar.txt") == "profile.tar.txt"

    def test_hidden_file(self):
        """Hidden files (starting with dot) should still be allowed if valid."""
        assert sanitize_filename(".hidden_profile.txt") == ".hidden_profile.txt"

    def test_double_dots_in_name(self):
        """Double dots in name (not path traversal) should be rejected."""
        assert sanitize_filename("pro..file.txt") is None


class TestValidateEqProfileContent:
    """Tests for EQ profile content validation."""

    def test_valid_simple_profile(self):
        """Valid simple EQ profile should pass."""
        content = """Preamp: -6.5 dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
Filter 2: ON PK Fc 1000 Hz Gain 3.5 dB Q 2.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True
        assert result["filter_count"] == 2
        assert result["preamp_db"] == -6.5
        assert len(result["errors"]) == 0

    def test_empty_content(self):
        """Empty content should fail."""
        result = validate_eq_profile_content("")
        assert result["valid"] is False
        assert "Empty file" in result["errors"]

    def test_whitespace_only(self):
        """Whitespace-only content should fail."""
        result = validate_eq_profile_content("   \n\n  ")
        assert result["valid"] is False

    def test_missing_preamp(self):
        """Missing Preamp line should fail."""
        content = """Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("Preamp" in e for e in result["errors"])

    def test_preamp_too_low(self):
        """Preamp below minimum should fail."""
        content = f"""Preamp: {PREAMP_MIN_DB - 10} dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("out of range" in e for e in result["errors"])

    def test_preamp_too_high(self):
        """Preamp above maximum should fail."""
        content = f"""Preamp: {PREAMP_MAX_DB + 10} dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("out of range" in e for e in result["errors"])

    def test_frequency_too_low(self):
        """Frequency below minimum should fail."""
        content = f"""Preamp: -6 dB
Filter 1: ON PK Fc {FREQ_MIN_HZ - 10} Hz Gain -2.0 dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("Frequency" in e and "out of range" in e for e in result["errors"])

    def test_frequency_too_high(self):
        """Frequency above maximum should fail."""
        content = f"""Preamp: -6 dB
Filter 1: ON PK Fc {FREQ_MAX_HZ + 1000} Hz Gain -2.0 dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("Frequency" in e and "out of range" in e for e in result["errors"])

    def test_gain_too_low(self):
        """Gain below minimum should fail."""
        content = f"""Preamp: -6 dB
Filter 1: ON PK Fc 1000 Hz Gain {GAIN_MIN_DB - 10} dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("Gain" in e and "out of range" in e for e in result["errors"])

    def test_gain_too_high(self):
        """Gain above maximum should fail."""
        content = f"""Preamp: -6 dB
Filter 1: ON PK Fc 1000 Hz Gain {GAIN_MAX_DB + 10} dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("Gain" in e and "out of range" in e for e in result["errors"])

    def test_q_too_low(self):
        """Q below minimum should fail."""
        content = f"""Preamp: -6 dB
Filter 1: ON PK Fc 1000 Hz Gain -2.0 dB Q {Q_MIN - 0.001}
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("Q" in e and "out of range" in e for e in result["errors"])

    def test_q_too_high(self):
        """Q above maximum should fail."""
        content = f"""Preamp: -6 dB
Filter 1: ON PK Fc 1000 Hz Gain -2.0 dB Q {Q_MAX + 10}
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("Q" in e and "out of range" in e for e in result["errors"])

    def test_too_many_filters(self):
        """Too many filters should fail."""
        filters = "\n".join(
            f"Filter {i}: ON PK Fc {100 + i * 10} Hz Gain -1.0 dB Q 1.0"
            for i in range(1, MAX_EQ_FILTERS + 10)
        )
        content = f"""Preamp: -6 dB
{filters}
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("Too many filters" in e for e in result["errors"])

    def test_valid_filter_types(self):
        """Various valid filter types should pass."""
        content = """Preamp: -6 dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
Filter 2: ON LS Fc 200 Hz Gain 3.0 dB Q 0.7
Filter 3: ON HS Fc 8000 Hz Gain -1.5 dB Q 0.7
Filter 4: ON LP Fc 20000 Hz
Filter 5: ON HP Fc 30 Hz
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True
        assert result["filter_count"] == 5

    def test_all_new_filter_types(self):
        """All new filter types should be recognized."""
        content = """Preamp: -6 dB
Filter 1: ON MODAL Fc 100 Hz Gain -2.0 dB Q 1.0
Filter 2: ON PEQ Fc 200 Hz Gain 3.0 dB Q 1.5
Filter 3: ON LPQ Fc 10000 Hz Q 0.7
Filter 4: ON HPQ Fc 20 Hz Q 0.7
Filter 5: ON BP Fc 1000 Hz Q 2.0
Filter 6: ON NO Fc 500 Hz Q 5.0
Filter 7: ON AP Fc 2000 Hz Gain 0 dB Q 1.0
Filter 8: ON LSC Fc 100 Hz Gain 2.0 dB Q 0.7
Filter 9: ON HSC Fc 8000 Hz Gain -2.0 dB Q 0.7
Filter 10: ON LSQ Fc 150 Hz Gain 3.0 dB Q 0.5
Filter 11: ON HSQ Fc 7000 Hz Gain -1.0 dB Q 0.5
Filter 12: ON LS 6DB Fc 200 Hz Gain 4.0 dB
Filter 13: ON LS 12DB Fc 250 Hz Gain 5.0 dB
Filter 14: ON HS 6DB Fc 6000 Hz Gain -3.0 dB
Filter 15: ON HS 12DB Fc 5000 Hz Gain -4.0 dB
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True
        assert result["filter_count"] == 15
        assert len(result["warnings"]) == 0

    def test_off_filters_accepted(self):
        """OFF filters should be accepted."""
        content = """Preamp: -6 dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
Filter 2: OFF PK Fc 200 Hz Gain 3.0 dB Q 1.5
Filter 3: ON LS Fc 300 Hz Gain 2.0 dB Q 0.7
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True
        assert result["filter_count"] == 3

    def test_subsonic_frequency(self):
        """Frequency down to 10Hz should be accepted."""
        content = f"""Preamp: -6 dB
Filter 1: ON HP Fc {FREQ_MIN_HZ} Hz
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True

    def test_low_q_value(self):
        """Q value down to 0.01 should be accepted."""
        content = f"""Preamp: -6 dB
Filter 1: ON PK Fc 1000 Hz Gain 2.0 dB Q {Q_MIN}
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True

    def test_optional_gain_parameter(self):
        """Filters without Gain (when not required) should pass."""
        content = """Preamp: -6 dB
Filter 1: ON LP Fc 15000 Hz Q 0.707
Filter 2: ON HP Fc 25 Hz Q 0.707
Filter 3: ON BP Fc 1000 Hz Q 2.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True
        assert result["filter_count"] == 3

    def test_missing_required_gain(self):
        """Filters requiring Gain but missing it should fail."""
        content = """Preamp: -6 dB
Filter 1: ON PK Fc 1000 Hz Q 1.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("requires Gain" in e for e in result["errors"])

    def test_missing_required_q(self):
        """Filters requiring Q but missing it should fail."""
        content = """Preamp: -6 dB
Filter 1: ON PK Fc 1000 Hz Gain 2.0 dB
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is False
        assert any("requires Q" in e for e in result["errors"])

    def test_unknown_filter_type_warning(self):
        """Unknown filter type should generate warning but still be valid."""
        content = """Preamp: -6 dB
Filter 1: ON UNKNOWN Fc 100 Hz Gain -2.0 dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        # Should be valid but with warning
        assert result["valid"] is True
        assert any("Unknown type" in w for w in result["warnings"])

    def test_comments_ignored(self):
        """Comments should be ignored."""
        content = """# This is a comment
Preamp: -6 dB
# Another comment
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True

    def test_opra_style_header(self):
        """OPRA-style header with comments should be valid."""
        content = """# OPRA: Sennheiser HD650
# Author: oratory1990
# License: CC BY-SA 4.0
# Source: https://github.com/opra-project/OPRA

Preamp: -6.5 dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True

    def test_preamp_only_warning(self):
        """Preamp with no filters should generate warning."""
        content = """Preamp: -6 dB
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True  # Still valid
        assert any("No filter lines" in w for w in result["warnings"])

    def test_malformed_filter_line_warning(self):
        """Malformed filter line should generate warning."""
        content = """Preamp: -6 dB
Filter 1: ON PK Fc 100 Hz
Filter 2: ON PK Fc 200 Hz Gain -2.0 dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        # Filter 1 is now recognized but missing required parameters
        assert result["valid"] is False
        assert result["filter_count"] == 2
        assert any("requires Gain" in e for e in result["errors"])
        assert any("requires Q" in e for e in result["errors"])

    def test_case_insensitive_filter_type(self):
        """Filter type should be case insensitive."""
        content = """Preamp: -6 dB
Filter 1: ON pk Fc 100 Hz Gain -2.0 dB Q 1.0
Filter 2: ON Pk Fc 200 Hz Gain 3.0 dB Q 2.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True
        assert result["filter_count"] == 2


class TestSecurityLimits:
    """Tests for security limit constants."""

    def test_file_size_limit_reasonable(self):
        """File size limit should be reasonable (1MB)."""
        assert MAX_EQ_FILE_SIZE == 1 * 1024 * 1024

    def test_filter_count_limit_reasonable(self):
        """Filter count limit should be reasonable."""
        assert MAX_EQ_FILTERS == 100

    def test_preamp_range_reasonable(self):
        """Preamp range should be reasonable."""
        assert PREAMP_MIN_DB == -100.0
        assert PREAMP_MAX_DB == 20.0

    def test_frequency_range_reasonable(self):
        """Frequency range should cover audible spectrum and subsonic."""
        assert FREQ_MIN_HZ == 10.0  # Extended to 10Hz for subsonic
        assert FREQ_MAX_HZ == 24000.0

    def test_gain_range_reasonable(self):
        """Gain range should be reasonable for EQ."""
        assert GAIN_MIN_DB == -30.0
        assert GAIN_MAX_DB == 30.0

    def test_q_range_reasonable(self):
        """Q range should be reasonable for parametric EQ."""
        assert Q_MIN == 0.01  # Extended to 0.01 for wider Q range
        assert Q_MAX == 100.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_windows_line_endings(self):
        """Windows line endings (CRLF) should work."""
        content = "Preamp: -6 dB\r\nFilter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0\r\n"
        result = validate_eq_profile_content(content)
        assert result["valid"] is True

    def test_mixed_line_endings(self):
        """Mixed line endings should work."""
        content = "Preamp: -6 dB\r\nFilter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0\n"
        result = validate_eq_profile_content(content)
        assert result["valid"] is True

    def test_trailing_whitespace(self):
        """Trailing whitespace should be handled."""
        content = "Preamp: -6 dB   \nFilter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0   \n"
        result = validate_eq_profile_content(content)
        assert result["valid"] is True

    def test_unicode_in_comments(self):
        """Unicode characters in comments should be fine."""
        content = """# 日本語コメント
# Émoji: 🎧
Preamp: -6 dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True

    def test_scientific_notation_not_supported(self):
        """Scientific notation in values should fail (treated as invalid)."""
        content = """Preamp: -6 dB
Filter 1: ON PK Fc 1e3 Hz Gain -2.0 dB Q 1.0
"""
        result = validate_eq_profile_content(content)
        # Should have a warning about unparseable line or just not parse the filter
        assert result["filter_count"] == 0 or len(result["warnings"]) > 0

    def test_boundary_values_valid(self):
        """Boundary values should be valid."""
        content = f"""Preamp: {PREAMP_MIN_DB} dB
Filter 1: ON PK Fc {FREQ_MIN_HZ} Hz Gain {GAIN_MIN_DB} dB Q {Q_MIN}
Filter 2: ON PK Fc {FREQ_MAX_HZ} Hz Gain {GAIN_MAX_DB} dB Q {Q_MAX}
"""
        result = validate_eq_profile_content(content)
        assert result["valid"] is True
        assert result["filter_count"] == 2
