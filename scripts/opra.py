#!/usr/bin/env python3
"""
OPRA Database Parser and EQ Converter.

Reads OPRA headphone EQ database from submodule and converts to Equalizer APO format.
Source: https://github.com/opra-project/OPRA
License: CC BY-SA 4.0

Usage:
    from opra import OpraDatabase, convert_opra_to_apo

    db = OpraDatabase()
    products = db.search("HD650")
    eq_apo = convert_opra_to_apo(products[0]["eq_profiles"][0])
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict


# Default path to OPRA submodule database
DEFAULT_OPRA_PATH = (
    Path(__file__).parent.parent / "data" / "opra-db" / "dist" / "database_v1.jsonl"
)


class CorrectionBandDict(TypedDict):
    """Type definition for correction band parameters."""

    filter_type: str
    frequency: float
    gain_db: float
    q: float


# Modern Target Correction (KB5000_7)
# Based on Dan Clark Audio research for more natural sound reproduction.
# This single band adjusts OPRA's Harman Target base to a modern reference.
# Applied at runtime to comply with CC BY-SA 4.0 (no derivative data distribution).
MODERN_TARGET_CORRECTION_BAND: CorrectionBandDict = {
    "filter_type": "PK",
    "frequency": 5366.0,
    "gain_db": 2.8,
    "q": 1.5,
}


@dataclass
class EqBand:
    """Single EQ band in Equalizer APO format."""

    enabled: bool = True
    filter_type: str = "PK"  # PK, LS, HS, LP, HP
    frequency: float = 1000.0
    gain_db: float = 0.0
    q: float = 1.0


@dataclass
class EqProfile:
    """Complete EQ profile in Equalizer APO format."""

    name: str = ""
    preamp_db: float = 0.0
    bands: list[EqBand] = field(default_factory=list)

    # Metadata for attribution (CC BY-SA 4.0)
    author: str = ""
    source: str = "OPRA"
    details: str = ""

    def to_apo_format(self) -> str:
        """Convert to Equalizer APO text format."""
        lines = []

        # Preamp line
        if self.preamp_db != 0.0:
            lines.append(f"Preamp: {self.preamp_db:.1f} dB")

        # Filter lines
        filter_num = 0
        for band in self.bands:
            if not band.enabled:
                continue

            filter_num += 1
            status = "ON"
            if band.filter_type in ("LP", "HP"):
                # LP/HP don't have gain
                line = f"Filter {filter_num}: {status} {band.filter_type} Fc {band.frequency:.1f} Hz Q {band.q:.2f}"
            else:
                line = f"Filter {filter_num}: {status} {band.filter_type} Fc {band.frequency:.1f} Hz Gain {band.gain_db:.1f} dB Q {band.q:.2f}"
            lines.append(line)

        return "\n".join(lines)


def slope_to_q(slope_db_per_oct: int) -> float:
    """
    Convert LP/HP filter slope (dB/oct) to Q value.

    For Butterworth filters:
    - 6 dB/oct (1st order): Q = 0.5
    - 12 dB/oct (2nd order): Q = 0.707
    - 18 dB/oct (3rd order): Q = 0.5 (cascaded)
    - 24 dB/oct (4th order): Q = 0.541

    This is an approximation for single biquad implementation.
    """
    # Mapping based on common filter designs
    slope_q_map = {
        6: 0.5,
        12: 0.707,  # Butterworth
        18: 0.5,
        24: 0.541,
        30: 0.5,
        36: 0.518,
    }
    return slope_q_map.get(slope_db_per_oct, 0.707)


def convert_opra_band(band_data: dict) -> EqBand | None:
    """
    Convert OPRA EQ band to Equalizer APO format.

    OPRA types: peak_dip, low_shelf, high_shelf, low_pass, high_pass, band_pass, band_stop
    APO types: PK, LS, HS, LP, HP
    """
    band_type = band_data.get("type", "")
    frequency = band_data.get("frequency", 1000.0)
    gain_db = band_data.get("gain_db", 0.0)
    q = band_data.get("q")
    slope = band_data.get("slope")  # For LP/HP filters

    # Type mapping
    type_map = {
        "peak_dip": "PK",
        "low_shelf": "LS",
        "high_shelf": "HS",
        "low_pass": "LP",
        "high_pass": "HP",
    }

    apo_type = type_map.get(band_type)
    if apo_type is None:
        # Unsupported types: band_pass, band_stop
        return None

    # Handle Q value
    if apo_type in ("LP", "HP"):
        # LP/HP use slope, convert to Q
        if slope is not None:
            q = slope_to_q(slope)
        else:
            q = 0.707  # Default Butterworth
        gain_db = 0.0  # LP/HP don't have gain
    elif q is None:
        q = 1.0  # Default Q

    return EqBand(
        enabled=True,
        filter_type=apo_type,
        frequency=frequency,
        gain_db=gain_db,
        q=q,
    )


def convert_opra_to_apo(eq_data: dict) -> EqProfile:
    """
    Convert OPRA EQ profile to Equalizer APO format.

    Args:
        eq_data: OPRA EQ data dict with 'parameters', 'author', etc.

    Returns:
        EqProfile in APO format
    """
    params = eq_data.get("parameters", {})
    bands_data = params.get("bands", [])

    bands = []
    for band_data in bands_data:
        band = convert_opra_band(band_data)
        if band is not None:
            bands.append(band)

    return EqProfile(
        name=eq_data.get("name", ""),
        preamp_db=params.get("gain_db", 0.0),
        bands=bands,
        author=eq_data.get("author", ""),
        source="OPRA",
        details=eq_data.get("details", ""),
    )


def apply_modern_target_correction(profile: EqProfile) -> EqProfile:
    """
    Apply Modern Target (KB5000_7) correction to an OPRA EQ profile.

    This adds a single peaking filter that adjusts the Harman Target base
    to a more modern reference inspired by Dan Clark Audio research.

    The correction is applied at runtime to comply with CC BY-SA 4.0
    (no derivative data distribution).

    Note: Preamp is reduced by the correction gain to prevent clipping.

    Args:
        profile: Original OPRA EQ profile

    Returns:
        New EqProfile with correction band appended and preamp adjusted
    """
    correction_gain = MODERN_TARGET_CORRECTION_BAND["gain_db"]

    correction_band = EqBand(
        enabled=True,
        filter_type=MODERN_TARGET_CORRECTION_BAND["filter_type"],
        frequency=MODERN_TARGET_CORRECTION_BAND["frequency"],
        gain_db=correction_gain,
        q=MODERN_TARGET_CORRECTION_BAND["q"],
    )

    # Reduce preamp by correction gain to prevent clipping
    adjusted_preamp = profile.preamp_db - correction_gain

    return EqProfile(
        name=profile.name,
        preamp_db=adjusted_preamp,
        bands=profile.bands + [correction_band],
        author=profile.author,
        source=profile.source,
        details=profile.details + " + Modern Target (KB5000_7)"
        if profile.details
        else "Modern Target (KB5000_7)",
    )


class OpraDatabase:
    """
    OPRA headphone EQ database reader.

    Reads from the OPRA submodule and provides search functionality.
    """

    def __init__(self, db_path: Path | None = None):
        """
        Initialize database.

        Args:
            db_path: Path to database_v1.jsonl. If None, uses default submodule path.
        """
        self.db_path = db_path or DEFAULT_OPRA_PATH
        self._vendors: dict[str, dict] = {}
        self._products: dict[str, dict] = {}
        self._eq_profiles: dict[str, dict] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load database if not already loaded."""
        if self._loaded:
            return

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"OPRA database not found at {self.db_path}. "
                "Run 'git submodule update --init' to fetch OPRA data."
            )

        with open(self.db_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    entry_type = entry.get("type")
                    entry_id = entry.get("id")
                    data = entry.get("data", {})

                    if entry_type == "vendor":
                        self._vendors[entry_id] = data
                    elif entry_type == "product":
                        self._products[entry_id] = data
                    elif entry_type == "eq":
                        self._eq_profiles[entry_id] = data
                except json.JSONDecodeError:
                    continue

        self._loaded = True

    @property
    def vendor_count(self) -> int:
        """Number of vendors in database."""
        self._ensure_loaded()
        return len(self._vendors)

    @property
    def product_count(self) -> int:
        """Number of products in database."""
        self._ensure_loaded()
        return len(self._products)

    @property
    def eq_profile_count(self) -> int:
        """Number of EQ profiles in database."""
        self._ensure_loaded()
        return len(self._eq_profiles)

    def get_vendors(self) -> list[dict]:
        """Get all vendors sorted by name."""
        self._ensure_loaded()
        vendors = [{"id": vid, **vdata} for vid, vdata in self._vendors.items()]
        vendors.sort(key=lambda v: v.get("name", "").lower())
        return vendors

    def get_products_by_vendor(self, vendor_id: str) -> list[dict]:
        """Get all products for a vendor."""
        self._ensure_loaded()
        products = []
        for pid, pdata in self._products.items():
            if pdata.get("vendor_id") == vendor_id:
                eq_profiles = self._get_eq_profiles_for_product(pid)
                products.append(
                    {
                        "id": pid,
                        "eq_profiles": eq_profiles,
                        **pdata,
                    }
                )
        products.sort(key=lambda p: p.get("name", "").lower())
        return products

    def _get_eq_profiles_for_product(self, product_id: str) -> list[dict]:
        """Get all EQ profiles for a product."""
        profiles = []
        for eq_id, eq_data in self._eq_profiles.items():
            if eq_data.get("product_id") == product_id:
                profiles.append({"id": eq_id, **eq_data})
        return profiles

    def search(self, query: str, limit: int = 50) -> list[dict]:
        """
        Search products by name (case-insensitive).

        Returns products with embedded vendor info and EQ profiles.
        """
        self._ensure_loaded()
        query_lower = query.lower()
        results = []

        for pid, pdata in self._products.items():
            product_name = pdata.get("name", "")
            vendor_id = pdata.get("vendor_id", "")
            vendor_data = self._vendors.get(vendor_id, {})
            vendor_name = vendor_data.get("name", "")

            # Search in product name and vendor name
            if (
                query_lower in product_name.lower()
                or query_lower in vendor_name.lower()
            ):
                eq_profiles = self._get_eq_profiles_for_product(pid)
                if eq_profiles:  # Only include products with EQ profiles
                    results.append(
                        {
                            "id": pid,
                            "name": product_name,
                            "type": pdata.get("type", ""),
                            "vendor": {
                                "id": vendor_id,
                                "name": vendor_name,
                            },
                            "eq_profiles": eq_profiles,
                        }
                    )

        # Sort by relevance (exact match first, then by name)
        def sort_key(item: dict) -> tuple:
            name = item.get("name", "").lower()
            vendor = item["vendor"].get("name", "").lower()
            exact_match = query_lower == name
            starts_with = name.startswith(query_lower) or vendor.startswith(query_lower)
            return (not exact_match, not starts_with, vendor, name)

        results.sort(key=sort_key)
        return results[:limit]

    def get_eq_profile(self, eq_id: str) -> dict | None:
        """Get a specific EQ profile by ID."""
        self._ensure_loaded()
        eq_data = self._eq_profiles.get(eq_id)
        if eq_data is None:
            return None
        return {"id": eq_id, **eq_data}

    def get_product(self, product_id: str) -> dict | None:
        """Get a specific product by ID."""
        self._ensure_loaded()
        pdata = self._products.get(product_id)
        if pdata is None:
            return None

        vendor_id = pdata.get("vendor_id", "")
        vendor_data = self._vendors.get(vendor_id, {})
        eq_profiles = self._get_eq_profiles_for_product(product_id)

        return {
            "id": product_id,
            "name": pdata.get("name", ""),
            "type": pdata.get("type", ""),
            "vendor": {
                "id": vendor_id,
                "name": vendor_data.get("name", ""),
            },
            "eq_profiles": eq_profiles,
        }


# Module-level singleton for convenience
_db_instance: OpraDatabase | None = None


def get_database() -> OpraDatabase:
    """Get shared database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = OpraDatabase()
    return _db_instance


if __name__ == "__main__":
    # Demo usage
    import sys

    db = OpraDatabase()
    print("OPRA Database loaded:")
    print(f"  Vendors: {db.vendor_count}")
    print(f"  Products: {db.product_count}")
    print(f"  EQ Profiles: {db.eq_profile_count}")

    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(f"\nSearch results for '{query}':")
        results = db.search(query, limit=10)
        for r in results:
            print(
                f"  {r['vendor']['name']} {r['name']} ({len(r['eq_profiles'])} profiles)"
            )
            for eq in r["eq_profiles"][:2]:
                print(f"    - {eq.get('author', 'unknown')}: {eq.get('details', '')}")

                # Convert to APO format
                apo = convert_opra_to_apo(eq)
                print("      APO format preview:")
                for line in apo.to_apo_format().split("\n")[:3]:
                    print(f"        {line}")
