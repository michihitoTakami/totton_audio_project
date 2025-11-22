"""Tests for OPRA database parser and EQ converter."""

import pytest

from opra import (
    OpraDatabase,
    EqBand,
    EqProfile,
    convert_opra_band,
    convert_opra_to_apo,
    slope_to_q,
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
