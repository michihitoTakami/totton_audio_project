from pathlib import Path

from web.services.eq import parse_eq_profile_content


def test_parse_modern_target_separates_kb5000_filters(tmp_path: Path):
    """Modern Target corrections should be isolated from OPRA filters."""
    eq_text = """# OPRA: Example Headphone
# Modern Target (KB5000_7)
Preamp: -6.0 dB
Filter 3: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
Filter 11: ON PK Fc 5366 Hz Gain 2.8 dB Q 1.5
Filter 14: ON PK Fc 2350 Hz Gain -0.9 dB Q 2.0
Filter 21: ON PK Fc 8000 Hz Gain 2.0 dB Q 0.7
"""
    eq_file = tmp_path / "modern_target.txt"
    eq_file.write_text(eq_text)

    parsed = parse_eq_profile_content(eq_file)

    assert parsed["has_modern_target"] is True
    assert parsed["source_type"] == "opra"
    assert parsed["original_filters"] == [
        "Filter 11: ON PK Fc 5366 Hz Gain 2.8 dB Q 1.5",
        "Filter 14: ON PK Fc 2350 Hz Gain -0.9 dB Q 2.0",
    ]
    assert "Filter 3: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0" in parsed["opra_filters"]
    assert "Filter 21: ON PK Fc 8000 Hz Gain 2.0 dB Q 0.7" in parsed["opra_filters"]
    assert any(line.startswith("Preamp:") for line in parsed["opra_filters"])


def test_parse_modern_target_handles_decimal_values(tmp_path: Path):
    """Detection should tolerate decimal rounding and missing filter numbers."""
    eq_text = """# OPRA: Example Headphone
# Modern Target (KB5000_7)
Preamp: -5.0 dB
Filter: ON PK Fc 5366.4 Hz Gain 2.75 dB Q 1.48
Filter 5: ON PK Fc 2351 Hz Gain -0.92 dB Q 2.05
Filter 6: ON PK Fc 5000 Hz Gain 1.0 dB Q 1.2
"""
    eq_file = tmp_path / "modern_target_decimals.txt"
    eq_file.write_text(eq_text)

    parsed = parse_eq_profile_content(eq_file)

    assert parsed["has_modern_target"] is True
    assert parsed["source_type"] == "opra"
    assert len(parsed["original_filters"]) == 2
    assert "Filter 6: ON PK Fc 5000 Hz Gain 1.0 dB Q 1.2" in parsed["opra_filters"]
    assert all("5366" in line or "2351" in line for line in parsed["original_filters"])
