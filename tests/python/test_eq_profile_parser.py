"""Tests for EQ profile parsing functionality in web/main.py."""

from pathlib import Path


# Import from web module (conftest.py adds project root to path)
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "web"))
from main import parse_eq_profile_content


class TestParseEqProfileContent:
    """Tests for parse_eq_profile_content() function."""

    def test_file_not_found(self, tmp_path: Path):
        """Returns error when file doesn't exist."""
        result = parse_eq_profile_content(tmp_path / "nonexistent.txt")
        assert "error" in result
        assert result["error"] == "File not found"

    def test_custom_profile_basic(self, tmp_path: Path):
        """Parses basic custom profile correctly."""
        content = """Preamp: -10.5 dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
Filter 2: ON PK Fc 200 Hz Gain 1.5 dB Q 1.2
"""
        profile_path = tmp_path / "custom.txt"
        profile_path.write_text(content)

        result = parse_eq_profile_content(profile_path)

        assert result["source_type"] == "custom"
        assert result["has_modern_target"] is False
        assert result["opra_info"] is None
        assert len(result["opra_filters"]) == 3
        assert result["opra_filters"][0] == "Preamp: -10.5 dB"
        assert len(result["original_filters"]) == 0

    def test_opra_profile_without_modern_target(self, tmp_path: Path):
        """Parses OPRA profile without Modern Target correction."""
        content = """# OPRA: Sennheiser HD650
# Author: oratory1990
# Details: harman_over-ear_2018
# License: CC BY-SA 4.0
# Source: https://github.com/opra-project/OPRA

Preamp: -6.5 dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
Filter 2: ON PK Fc 200 Hz Gain 1.5 dB Q 1.2
"""
        profile_path = tmp_path / "opra.txt"
        profile_path.write_text(content)

        result = parse_eq_profile_content(profile_path)

        assert result["source_type"] == "opra"
        assert result["has_modern_target"] is False
        assert result["opra_info"]["product"] == "Sennheiser HD650"
        assert result["opra_info"]["author"] == "oratory1990"
        assert result["opra_info"]["license"] == "CC BY-SA 4.0"
        assert len(result["opra_filters"]) == 3
        assert len(result["original_filters"]) == 0

    def test_opra_profile_with_modern_target(self, tmp_path: Path):
        """Parses OPRA profile with Modern Target correction, separating KB5000_7 filter."""
        content = """# OPRA: Sennheiser HD650
# Author: oratory1990
# Details: harman_over-ear_2018 + KB5000_7 correction
# Modern Target: KB5000_7 correction applied
# License: CC BY-SA 4.0
# Source: https://github.com/opra-project/OPRA

Preamp: -6.5 dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
Filter 2: ON PK Fc 200 Hz Gain 1.5 dB Q 1.2
Filter 11: ON PK Fc 5366 Hz Gain 2.8 dB Q 1.5
"""
        profile_path = tmp_path / "opra_kb5000_7.txt"
        profile_path.write_text(content)

        result = parse_eq_profile_content(profile_path)

        assert result["source_type"] == "opra"
        assert result["has_modern_target"] is True
        # OPRA filters should NOT include KB5000_7 correction
        assert len(result["opra_filters"]) == 3
        assert all("5366" not in f for f in result["opra_filters"])
        # Original filters should contain KB5000_7 correction
        assert len(result["original_filters"]) == 1
        assert "Fc 5366" in result["original_filters"][0]
        assert "Gain 2.8" in result["original_filters"][0]

    def test_kb5000_7_filter_detection_uses_correct_values(self, tmp_path: Path):
        """Verifies KB5000_7 detection matches MODERN_TARGET_CORRECTION_BAND values."""
        from opra import MODERN_TARGET_CORRECTION_BAND

        # Create profile with exact values from MODERN_TARGET_CORRECTION_BAND
        fc = int(MODERN_TARGET_CORRECTION_BAND["frequency"])
        gain = MODERN_TARGET_CORRECTION_BAND["gain_db"]
        q = MODERN_TARGET_CORRECTION_BAND["q"]

        content = f"""# OPRA: Test Headphone
# Modern Target: KB5000_7 correction applied

Preamp: -5.0 dB
Filter 1: ON PK Fc 100 Hz Gain -1.0 dB Q 1.0
Filter 2: ON PK Fc {fc} Hz Gain {gain} dB Q {q}
"""
        profile_path = tmp_path / "test_kb5000_7.txt"
        profile_path.write_text(content)

        result = parse_eq_profile_content(profile_path)

        assert len(result["original_filters"]) == 1
        assert f"Fc {fc}" in result["original_filters"][0]

    def test_similar_but_different_filter_not_detected_as_kb5000_7(
        self, tmp_path: Path
    ):
        """Filter with similar but different values should NOT be detected as KB5000_7."""
        content = """# OPRA: Test Headphone
# Modern Target: KB5000_7 correction applied

Preamp: -5.0 dB
Filter 1: ON PK Fc 5366 Hz Gain 3.0 dB Q 1.5
Filter 2: ON PK Fc 5367 Hz Gain 2.8 dB Q 1.5
Filter 3: ON PK Fc 5366 Hz Gain 2.8 dB Q 1.6
"""
        profile_path = tmp_path / "test_similar.txt"
        profile_path.write_text(content)

        result = parse_eq_profile_content(profile_path)

        # None of these should be detected as KB5000_7 (values don't match exactly)
        assert len(result["original_filters"]) == 0
        assert len(result["opra_filters"]) == 4  # Preamp + 3 filters

    def test_empty_file(self, tmp_path: Path):
        """Handles empty file gracefully."""
        profile_path = tmp_path / "empty.txt"
        profile_path.write_text("")

        result = parse_eq_profile_content(profile_path)

        assert result["source_type"] == "custom"
        assert len(result["opra_filters"]) == 0

    def test_raw_content_preserved(self, tmp_path: Path):
        """Raw content is preserved in result."""
        content = "Preamp: -5.0 dB\nFilter 1: ON PK Fc 100 Hz Gain -1.0 dB Q 1.0"
        profile_path = tmp_path / "test.txt"
        profile_path.write_text(content)

        result = parse_eq_profile_content(profile_path)

        assert result["raw_content"] == content

    def test_ignores_non_filter_lines(self, tmp_path: Path):
        """Non-filter lines (comments, blank lines) are ignored in filter output."""
        content = """# This is a comment
Preamp: -5.0 dB

# Another comment
Filter 1: ON PK Fc 100 Hz Gain -1.0 dB Q 1.0

Some random text
Filter 2: ON PK Fc 200 Hz Gain 1.0 dB Q 1.0
"""
        profile_path = tmp_path / "with_comments.txt"
        profile_path.write_text(content)

        result = parse_eq_profile_content(profile_path)

        # Should only have Preamp and Filter lines
        assert len(result["opra_filters"]) == 3
        assert result["opra_filters"][0] == "Preamp: -5.0 dB"
        assert result["opra_filters"][1].startswith("Filter 1:")
        assert result["opra_filters"][2].startswith("Filter 2:")
