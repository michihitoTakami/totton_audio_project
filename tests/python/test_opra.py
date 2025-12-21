"""Tests for OPRA database parser and EQ converter."""

from pathlib import Path

import pytest

from scripts.integration.opra import (
    DEFAULT_OPRA_PATH,
    MODERN_TARGET_CORRECTION_BAND,
    EqBand,
    EqProfile,
    OpraDatabase,
    apply_modern_target_correction,
    convert_opra_band,
    convert_opra_to_apo,
    slope_to_q,
)

# Skip marker for tests requiring OPRA submodule
requires_opra_submodule = pytest.mark.skipif(
    not Path(DEFAULT_OPRA_PATH).exists(),
    reason="OPRA data not installed. Run OPRA sync (or init submodule for dev)",
)


class TestSlopeToQ:
    """Tests for slope_to_q conversion."""

    def test_butterworth_12db(self):
        """12 dB/oct should give Butterworth Q of 0.707."""
        assert slope_to_q(12) == pytest.approx(0.707, rel=0.01)

    def test_6db_slope(self):
        """6 dB/oct (1st order) should give Q of 0.5."""
        assert slope_to_q(6) == 0.5

    def test_unknown_slope_defaults_to_butterworth(self):
        """Unknown slope should default to Butterworth Q."""
        assert slope_to_q(15) == 0.707


class TestConvertOpraBand:
    """Tests for converting OPRA band to APO format."""

    def test_peak_dip_conversion(self):
        """Test peak_dip band conversion."""
        opra_band = {
            "type": "peak_dip",
            "frequency": 1000.0,
            "gain_db": -3.5,
            "q": 1.41,
        }
        band = convert_opra_band(opra_band)

        assert band is not None
        assert band.filter_type == "PK"
        assert band.frequency == 1000.0
        assert band.gain_db == -3.5
        assert band.q == 1.41

    def test_low_shelf_conversion(self):
        """Test low_shelf band conversion."""
        opra_band = {
            "type": "low_shelf",
            "frequency": 100.0,
            "gain_db": 4.0,
            "q": 0.7,
        }
        band = convert_opra_band(opra_band)

        assert band is not None
        assert band.filter_type == "LS"
        assert band.frequency == 100.0
        assert band.gain_db == 4.0
        assert band.q == 0.7

    def test_high_shelf_conversion(self):
        """Test high_shelf band conversion."""
        opra_band = {
            "type": "high_shelf",
            "frequency": 8000.0,
            "gain_db": -2.0,
            "q": 0.7,
        }
        band = convert_opra_band(opra_band)

        assert band is not None
        assert band.filter_type == "HS"
        assert band.frequency == 8000.0
        assert band.gain_db == -2.0

    def test_low_pass_with_slope(self):
        """Test low_pass with slope conversion to Q."""
        opra_band = {
            "type": "low_pass",
            "frequency": 10000.0,
            "slope": 12,
        }
        band = convert_opra_band(opra_band)

        assert band is not None
        assert band.filter_type == "LP"
        assert band.frequency == 10000.0
        assert band.q == pytest.approx(0.707, rel=0.01)
        assert band.gain_db == 0.0  # LP has no gain

    def test_high_pass_with_slope(self):
        """Test high_pass with slope conversion to Q."""
        opra_band = {
            "type": "high_pass",
            "frequency": 20.0,
            "slope": 24,
        }
        band = convert_opra_band(opra_band)

        assert band is not None
        assert band.filter_type == "HP"
        assert band.frequency == 20.0
        assert band.q == 0.541

    def test_unsupported_band_type_returns_none(self):
        """Unsupported band types should return None."""
        opra_band = {
            "type": "band_pass",
            "frequency": 1000.0,
            "q": 2.0,
        }
        band = convert_opra_band(opra_band)
        assert band is None

    def test_missing_q_defaults_to_1(self):
        """Missing Q should default to 1.0 for peaking filters."""
        opra_band = {
            "type": "peak_dip",
            "frequency": 1000.0,
            "gain_db": 2.0,
        }
        band = convert_opra_band(opra_band)

        assert band is not None
        assert band.q == 1.0


class TestEqProfile:
    """Tests for EqProfile APO format generation."""

    def test_empty_profile(self):
        """Empty profile should produce empty string."""
        profile = EqProfile()
        assert profile.to_apo_format() == ""

    def test_preamp_only(self):
        """Profile with only preamp."""
        profile = EqProfile(preamp_db=-6.0)
        apo = profile.to_apo_format()
        assert "Preamp: -6.0 dB" in apo

    def test_single_peaking_band(self):
        """Profile with single peaking band."""
        profile = EqProfile(
            bands=[EqBand(filter_type="PK", frequency=1000.0, gain_db=-3.0, q=1.41)]
        )
        apo = profile.to_apo_format()
        assert "Filter 1: ON PK Fc 1000.0 Hz Gain -3.0 dB Q 1.41" in apo

    def test_multiple_bands(self):
        """Profile with multiple bands."""
        profile = EqProfile(
            preamp_db=-5.0,
            bands=[
                EqBand(filter_type="LS", frequency=100.0, gain_db=3.0, q=0.7),
                EqBand(filter_type="PK", frequency=1000.0, gain_db=-2.0, q=2.0),
                EqBand(filter_type="HS", frequency=8000.0, gain_db=1.5, q=0.7),
            ],
        )
        apo = profile.to_apo_format()

        lines = apo.split("\n")
        assert lines[0] == "Preamp: -5.0 dB"
        assert "Filter 1: ON LS" in lines[1]
        assert "Filter 2: ON PK" in lines[2]
        assert "Filter 3: ON HS" in lines[3]

    def test_lp_hp_format_no_gain(self):
        """LP/HP filters should not include Gain in APO format."""
        profile = EqProfile(
            bands=[EqBand(filter_type="LP", frequency=10000.0, gain_db=0.0, q=0.707)]
        )
        apo = profile.to_apo_format()
        assert "Gain" not in apo
        assert "Filter 1: ON LP Fc 10000.0 Hz Q 0.71" in apo

    def test_disabled_band_skipped(self):
        """Disabled bands should be skipped."""
        profile = EqProfile(
            bands=[
                EqBand(
                    enabled=False,
                    filter_type="PK",
                    frequency=1000.0,
                    gain_db=-3.0,
                    q=1.0,
                ),
                EqBand(
                    enabled=True, filter_type="PK", frequency=2000.0, gain_db=2.0, q=1.0
                ),
            ]
        )
        apo = profile.to_apo_format()
        assert "1000.0" not in apo
        assert "Filter 1: ON PK Fc 2000.0" in apo  # Should be Filter 1, not 2


class TestConvertOpraToApo:
    """Tests for full OPRA to APO profile conversion."""

    def test_full_profile_conversion(self):
        """Test converting a complete OPRA profile."""
        opra_data = {
            "name": "HD650",
            "author": "oratory1990",
            "details": "Harman Target",
            "parameters": {
                "gain_db": -6.0,
                "bands": [
                    {"type": "peak_dip", "frequency": 100.0, "gain_db": 2.0, "q": 0.7},
                    {
                        "type": "peak_dip",
                        "frequency": 3000.0,
                        "gain_db": -3.0,
                        "q": 1.5,
                    },
                    {
                        "type": "high_shelf",
                        "frequency": 8000.0,
                        "gain_db": 1.0,
                        "q": 0.7,
                    },
                ],
            },
        }

        profile = convert_opra_to_apo(opra_data)

        assert profile.name == "HD650"
        assert profile.author == "oratory1990"
        assert profile.preamp_db == -6.0
        assert len(profile.bands) == 3
        assert profile.bands[0].filter_type == "PK"
        assert profile.bands[2].filter_type == "HS"

    def test_profile_with_lp_filter(self):
        """Test profile with LP filter."""
        opra_data = {
            "name": "Test",
            "author": "test",
            "parameters": {
                "gain_db": 0.0,
                "bands": [
                    {"type": "low_pass", "frequency": 10000.0, "slope": 12},
                ],
            },
        }

        profile = convert_opra_to_apo(opra_data)

        assert len(profile.bands) == 1
        assert profile.bands[0].filter_type == "LP"
        assert profile.bands[0].q == pytest.approx(0.707, rel=0.01)

    def test_unsupported_bands_filtered(self):
        """Unsupported band types should be filtered out."""
        opra_data = {
            "name": "Test",
            "author": "test",
            "parameters": {
                "gain_db": 0.0,
                "bands": [
                    {"type": "peak_dip", "frequency": 1000.0, "gain_db": 2.0, "q": 1.0},
                    {"type": "band_pass", "frequency": 500.0, "q": 2.0},  # Unsupported
                    {
                        "type": "peak_dip",
                        "frequency": 2000.0,
                        "gain_db": -1.0,
                        "q": 1.0,
                    },
                ],
            },
        }

        profile = convert_opra_to_apo(opra_data)

        assert len(profile.bands) == 2  # band_pass should be filtered out


class TestModernTargetCorrection:
    """Tests for Modern Target (KB5000_7) correction."""

    def test_correction_band_values(self):
        """Verify correction band constant values match specification."""
        # From CLAUDE.md: Filter 11: ON PK Fc 5366 Hz Gain 2.8 dB Q 1.5
        assert MODERN_TARGET_CORRECTION_BAND["filter_type"] == "PK"
        assert MODERN_TARGET_CORRECTION_BAND["frequency"] == 5366.0
        assert MODERN_TARGET_CORRECTION_BAND["gain_db"] == 2.8
        assert MODERN_TARGET_CORRECTION_BAND["q"] == 1.5

    def test_apply_correction_adds_band(self):
        """Correction should add one additional band."""
        original = EqProfile(
            name="Test",
            preamp_db=-6.0,
            bands=[
                EqBand(filter_type="PK", frequency=1000.0, gain_db=-3.0, q=1.0),
                EqBand(filter_type="PK", frequency=2000.0, gain_db=2.0, q=1.5),
            ],
            author="test",
            details="Original details",
        )

        corrected = apply_modern_target_correction(original)

        # Should have one more band
        assert len(corrected.bands) == len(original.bands) + 1

    def test_apply_correction_preserves_original_bands(self):
        """Correction should not modify original bands."""
        original = EqProfile(
            name="Test",
            preamp_db=-6.0,
            bands=[
                EqBand(filter_type="PK", frequency=1000.0, gain_db=-3.0, q=1.0),
            ],
        )

        corrected = apply_modern_target_correction(original)

        # First band should be unchanged
        assert corrected.bands[0].frequency == 1000.0
        assert corrected.bands[0].gain_db == -3.0

    def test_apply_correction_band_is_last(self):
        """Correction band should be appended at the end."""
        original = EqProfile(
            name="Test",
            bands=[
                EqBand(filter_type="PK", frequency=1000.0, gain_db=-3.0, q=1.0),
            ],
        )

        corrected = apply_modern_target_correction(original)
        last_band = corrected.bands[-1]

        assert last_band.frequency == 5366.0
        assert last_band.gain_db == 2.8
        assert last_band.q == 1.5

    def test_apply_correction_updates_details(self):
        """Correction should update details field."""
        original = EqProfile(details="Harman Target")
        corrected = apply_modern_target_correction(original)

        assert "Modern Target" in corrected.details
        assert "KB5000_7" in corrected.details
        assert "Harman Target" in corrected.details

    def test_apply_correction_empty_details(self):
        """Correction should work with empty details."""
        original = EqProfile(details="")
        corrected = apply_modern_target_correction(original)

        assert "Modern Target" in corrected.details
        assert "KB5000_7" in corrected.details

    def test_apply_correction_preserves_metadata(self):
        """Correction should preserve other profile metadata (except preamp)."""
        original = EqProfile(
            name="HD650",
            preamp_db=-5.5,
            author="oratory1990",
            source="OPRA",
        )

        corrected = apply_modern_target_correction(original)

        assert corrected.name == original.name
        assert corrected.author == original.author
        assert corrected.source == original.source

    def test_apply_correction_adjusts_preamp(self):
        """Correction should reduce preamp by correction gain to prevent clipping."""
        correction_gain = MODERN_TARGET_CORRECTION_BAND["gain_db"]  # 2.8 dB

        original = EqProfile(
            name="Test",
            preamp_db=-5.0,
        )

        corrected = apply_modern_target_correction(original)

        # Preamp should be reduced by correction gain
        expected_preamp = original.preamp_db - correction_gain
        assert corrected.preamp_db == pytest.approx(expected_preamp, rel=0.01)

    def test_apply_correction_zero_preamp(self):
        """Correction with zero preamp should result in negative preamp."""
        correction_gain = MODERN_TARGET_CORRECTION_BAND["gain_db"]

        original = EqProfile(preamp_db=0.0)
        corrected = apply_modern_target_correction(original)

        assert corrected.preamp_db == pytest.approx(-correction_gain, rel=0.01)

    def test_apply_correction_apo_format(self):
        """Correction band should appear in APO format output."""
        original = EqProfile(
            preamp_db=-6.0,
            bands=[
                EqBand(filter_type="PK", frequency=1000.0, gain_db=-3.0, q=1.0),
            ],
        )

        corrected = apply_modern_target_correction(original)
        apo = corrected.to_apo_format()

        # Should contain the correction band
        assert "5366.0" in apo
        assert "2.8" in apo
        assert "1.50" in apo  # Q is formatted to 2 decimal places

    def test_corrected_vs_uncorrected_comprehensive_diff(self):
        """Comprehensive comparison between corrected and uncorrected profiles.

        Verifies that Modern Target correction produces all expected differences:
        - Band count increases by 1
        - Preamp decreases by correction gain
        - Last band is KB5000_7 correction band
        - APO format outputs differ appropriately
        """
        # Create a realistic multi-band profile
        original = EqProfile(
            name="HD650",
            preamp_db=-6.5,
            author="test",
            details="Harman Target",
            bands=[
                EqBand(filter_type="LS", frequency=105.0, gain_db=4.5, q=0.71),
                EqBand(filter_type="PK", frequency=200.0, gain_db=1.0, q=1.41),
                EqBand(filter_type="PK", frequency=1800.0, gain_db=-2.0, q=2.0),
                EqBand(filter_type="PK", frequency=3500.0, gain_db=-1.5, q=1.0),
                EqBand(filter_type="HS", frequency=8000.0, gain_db=2.0, q=0.7),
            ],
        )

        corrected = apply_modern_target_correction(original)
        correction_gain = MODERN_TARGET_CORRECTION_BAND["gain_db"]

        # 1. Band count: corrected should have exactly 1 more band
        assert len(corrected.bands) == len(original.bands) + 1

        # 2. Preamp: corrected should be reduced by correction gain (2.8 dB)
        expected_preamp = original.preamp_db - correction_gain
        assert corrected.preamp_db == pytest.approx(expected_preamp, rel=0.01)

        # 3. Original bands preserved: first N bands should be identical
        for i, orig_band in enumerate(original.bands):
            corr_band = corrected.bands[i]
            assert corr_band.filter_type == orig_band.filter_type
            assert corr_band.frequency == orig_band.frequency
            assert corr_band.gain_db == orig_band.gain_db
            assert corr_band.q == orig_band.q

        # 4. Last band is KB5000_7 correction band
        kb_band = corrected.bands[-1]
        assert kb_band.filter_type == MODERN_TARGET_CORRECTION_BAND["filter_type"]
        assert kb_band.frequency == MODERN_TARGET_CORRECTION_BAND["frequency"]
        assert kb_band.gain_db == MODERN_TARGET_CORRECTION_BAND["gain_db"]
        assert kb_band.q == MODERN_TARGET_CORRECTION_BAND["q"]

        # 5. APO format differences
        apo_original = original.to_apo_format()
        apo_corrected = corrected.to_apo_format()

        # Original should NOT contain KB5000_7 band
        assert "5366.0" not in apo_original

        # Corrected should contain KB5000_7 band
        assert "5366.0" in apo_corrected
        assert "Gain 2.8 dB" in apo_corrected
        assert "Q 1.50" in apo_corrected

        # Different filter counts in APO format
        orig_filter_count = apo_original.count("Filter")
        corr_filter_count = apo_corrected.count("Filter")
        assert corr_filter_count == orig_filter_count + 1

        # Corrected should have KB5000_7 in details
        assert "KB5000_7" in corrected.details
        assert "KB5000_7" not in original.details


@requires_opra_submodule
class TestOpraDatabase:
    """Tests for OpraDatabase class."""

    @pytest.fixture
    def db(self):
        """Get database instance."""
        return OpraDatabase()

    def test_database_loads(self, db):
        """Database should load without errors."""
        assert db.vendor_count > 0
        assert db.product_count > 0
        assert db.eq_profile_count > 0

    def test_database_counts(self, db):
        """Database should have expected number of entries."""
        assert db.vendor_count >= 600  # OPRA has 633+ vendors
        assert db.product_count >= 5000  # OPRA has 5234+ products
        assert db.eq_profile_count >= 8000  # OPRA has 8180+ profiles

    def test_search_returns_results(self, db):
        """Search should return matching results."""
        results = db.search("HD650")
        assert len(results) >= 1

        # Should find Sennheiser HD650
        hd650 = None
        for r in results:
            if r["name"] == "HD650" and r["vendor"]["name"] == "Sennheiser":
                hd650 = r
                break

        assert hd650 is not None
        assert len(hd650["eq_profiles"]) >= 1

    def test_search_case_insensitive(self, db):
        """Search should be case-insensitive."""
        results_lower = db.search("hd650")
        results_upper = db.search("HD650")

        assert len(results_lower) == len(results_upper)

    def test_search_by_vendor(self, db):
        """Search should also match vendor name."""
        results = db.search("Sennheiser")
        assert len(results) > 10  # Sennheiser has many products

    def test_search_limit(self, db):
        """Search should respect limit parameter."""
        results = db.search("", limit=5)  # Empty query matches all
        assert len(results) <= 5

    def test_get_vendors(self, db):
        """Get vendors should return sorted list."""
        vendors = db.get_vendors()
        assert len(vendors) > 0

        # Check sorting
        names = [v.get("name", "") for v in vendors]
        assert names == sorted(names, key=str.lower)

    def test_get_product(self, db):
        """Get specific product by ID."""
        # First search for a product
        results = db.search("HD650", limit=1)
        assert len(results) > 0

        product_id = results[0]["id"]
        product = db.get_product(product_id)

        assert product is not None
        assert product["id"] == product_id
        assert "vendor" in product
        assert "eq_profiles" in product

    def test_get_eq_profile(self, db):
        """Get specific EQ profile by ID."""
        # Search for product with EQ profiles
        results = db.search("HD650", limit=1)
        assert len(results) > 0
        assert len(results[0]["eq_profiles"]) > 0

        eq_id = results[0]["eq_profiles"][0]["id"]
        eq_profile = db.get_eq_profile(eq_id)

        assert eq_profile is not None
        assert eq_profile["id"] == eq_id
        assert "parameters" in eq_profile

    def test_nonexistent_product_returns_none(self, db):
        """Get nonexistent product should return None."""
        product = db.get_product("nonexistent_product_id_12345")
        assert product is None

    def test_nonexistent_eq_returns_none(self, db):
        """Get nonexistent EQ profile should return None."""
        eq = db.get_eq_profile("nonexistent_eq_id_12345")
        assert eq is None


@requires_opra_submodule
class TestOpraIntegration:
    """Integration tests for full OPRA workflow."""

    def test_search_and_convert_workflow(self):
        """Test complete workflow: search -> get profile -> convert to APO."""
        db = OpraDatabase()

        # Search
        results = db.search("HD650")
        assert len(results) > 0

        # Get EQ profile
        product = results[0]
        assert len(product["eq_profiles"]) > 0
        eq_data = product["eq_profiles"][0]

        # Convert to APO
        profile = convert_opra_to_apo(eq_data)
        apo_text = profile.to_apo_format()

        # Verify APO format
        assert "Preamp:" in apo_text or "Filter" in apo_text
        assert profile.author != ""


@requires_opra_submodule
class TestOpraApi:
    """API layer tests for OPRA endpoints with apply_correction."""

    @pytest.fixture
    def client(self, web_app):
        """Use shared web_app fixture from conftest.py."""
        return web_app

    @pytest.fixture
    def sample_eq_id(self, client):
        """Get a sample EQ ID from search results."""
        response = client.get("/opra/search?q=HD650&limit=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) > 0
        assert len(data["results"][0]["eq_profiles"]) > 0
        return data["results"][0]["eq_profiles"][0]["id"]

    def test_opra_eq_without_correction(self, client, sample_eq_id):
        """Test /opra/eq endpoint without apply_correction."""
        response = client.get(f"/opra/eq/{sample_eq_id}")
        assert response.status_code == 200
        data = response.json()

        assert "apo_format" in data
        assert data["modern_target_applied"] is False
        assert "KB5000_7" not in data["details"]

    def test_opra_eq_with_correction(self, client, sample_eq_id):
        """Test /opra/eq endpoint with apply_correction=true."""
        response = client.get(f"/opra/eq/{sample_eq_id}?apply_correction=true")
        assert response.status_code == 200
        data = response.json()

        assert "apo_format" in data
        assert data["modern_target_applied"] is True
        assert "KB5000_7" in data["details"]
        # Correction band should appear in APO format
        assert "5366.0" in data["apo_format"]
        assert "2.8" in data["apo_format"]

    def test_opra_eq_correction_reduces_preamp(self, client, sample_eq_id):
        """Test that apply_correction reduces preamp in APO output."""
        # Get without correction
        resp_no_correction = client.get(f"/opra/eq/{sample_eq_id}")
        assert resp_no_correction.status_code == 200
        apo_no_correction = resp_no_correction.json()["apo_format"]

        # Get with correction
        resp_with_correction = client.get(
            f"/opra/eq/{sample_eq_id}?apply_correction=true"
        )
        assert resp_with_correction.status_code == 200
        apo_with_correction = resp_with_correction.json()["apo_format"]

        # Parse preamp values
        def parse_preamp(apo_text):
            for line in apo_text.split("\n"):
                if line.startswith("Preamp:"):
                    # "Preamp: -6.0 dB" -> -6.0
                    return float(line.split(":")[1].strip().replace(" dB", ""))
            return 0.0

        preamp_no_correction = parse_preamp(apo_no_correction)
        preamp_with_correction = parse_preamp(apo_with_correction)

        # Preamp should be reduced by 2.8 dB (correction gain)
        correction_gain = MODERN_TARGET_CORRECTION_BAND["gain_db"]
        expected_preamp = preamp_no_correction - correction_gain
        assert preamp_with_correction == pytest.approx(expected_preamp, rel=0.01)

    def test_opra_apply_with_correction_filename(self, tmp_path, monkeypatch):
        """Test /opra/apply endpoint generates correct filename suffix.

        Uses monkeypatch to isolate test files from real workspace.
        """
        import json

        # Setup isolated test directories
        temp_eq_dir = tmp_path / "EQ"
        temp_eq_dir.mkdir()
        temp_config = tmp_path / "config.json"
        temp_config.write_text("{}")

        # Patch the modules that actually USE the constants
        from web.routers import opra
        from web.services import config as config_service

        monkeypatch.setattr(opra, "EQ_PROFILES_DIR", temp_eq_dir)
        monkeypatch.setattr(config_service, "CONFIG_PATH", temp_config)

        # Also need to patch constants for config service to use temp path
        from web import constants

        monkeypatch.setattr(constants, "CONFIG_PATH", temp_config)
        monkeypatch.setattr(constants, "EQ_PROFILES_DIR", temp_eq_dir)

        # Create a new client with patched modules
        from fastapi.testclient import TestClient
        from web import main

        client = TestClient(main.app)

        # Get sample EQ ID
        response = client.get("/opra/search?q=HD650&limit=1")
        assert response.status_code == 200
        data = response.json()
        sample_eq_id = data["results"][0]["eq_profiles"][0]["id"]

        # Test apply endpoint with correction
        response = client.post(f"/opra/apply/{sample_eq_id}?apply_correction=true")
        assert response.status_code == 200
        data = response.json()

        assert data["data"]["modern_target_applied"] is True
        profile_name = data["data"]["profile_name"]
        assert "_kb5000_7" in profile_name

        # Verify file was created in temp directory (not real data/EQ)
        created_files = list(temp_eq_dir.glob("opra_*_kb5000_7.txt"))
        assert len(created_files) == 1

        # Verify file content has Modern Target correction
        content = created_files[0].read_text()
        assert "KB5000_7" in content
        assert "5366" in content  # Correction frequency
        assert "2.8" in content  # Correction gain

        # Verify config was updated in temp file (not real config.json)
        config_data = json.loads(temp_config.read_text())
        assert config_data.get("eqEnabled") is True
        assert "_kb5000_7" in config_data.get("eqProfilePath", "")


class TestOpraErrorHandling:
    """Tests for OPRA error handling when database is not available."""

    def test_database_not_found_raises_error(self, tmp_path):
        """OpraDatabase should raise FileNotFoundError when DB is missing."""
        nonexistent_path = tmp_path / "nonexistent" / "database.jsonl"
        db = OpraDatabase(db_path=nonexistent_path)

        with pytest.raises(FileNotFoundError) as exc_info:
            _ = db.vendor_count  # Triggers _ensure_loaded()

        assert "OPRA database not found" in str(exc_info.value)
        assert "Run OPRA sync" in str(exc_info.value)

    def test_error_message_includes_helpful_instructions(self, tmp_path):
        """Error message should include helpful instructions for users."""
        nonexistent_path = tmp_path / "nonexistent" / "database.jsonl"
        db = OpraDatabase(db_path=nonexistent_path)

        try:
            _ = db.search("test")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError as e:
            error_msg = str(e)
            # Check for essential information
            assert "OPRA database not found" in error_msg
            assert "Run OPRA sync" in error_msg
            assert str(nonexistent_path) in error_msg
