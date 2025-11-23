"""OPRA database endpoints."""

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Add scripts directory to path for OPRA module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from opra import (  # noqa: E402
    apply_modern_target_correction,
    convert_opra_to_apo,
    get_database as get_opra_database,
)

from ..constants import EQ_PROFILES_DIR
from ..models import ApiResponse
from ..services import load_config, save_config

router = APIRouter(prefix="/opra", tags=["opra"])


@router.get("/stats")
async def opra_stats():
    """Get OPRA database statistics."""
    try:
        db = get_opra_database()
        return {
            "vendors": db.vendor_count,
            "products": db.product_count,
            "eq_profiles": db.eq_profile_count,
            "license": "CC BY-SA 4.0",
            "attribution": "OPRA Project (https://github.com/opra-project/OPRA)",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/vendors")
async def opra_vendors():
    """List all vendors in OPRA database."""
    try:
        db = get_opra_database()
        vendors = db.get_vendors()
        return {"vendors": vendors, "count": len(vendors)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/search")
async def opra_search(q: str = "", limit: int = 50):
    """
    Search headphones by name.
    Returns products with embedded vendor info and EQ profiles.
    """
    try:
        db = get_opra_database()
        results = db.search(q, limit=limit)
        return {"results": results, "count": len(results), "query": q}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/products/{product_id}")
async def opra_product(product_id: str):
    """Get a specific product with its EQ profiles."""
    try:
        db = get_opra_database()
        product = db.get_product(product_id)
        if product is None:
            raise HTTPException(
                status_code=404, detail=f"Product '{product_id}' not found"
            )
        return product
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/eq/{eq_id}")
async def opra_eq_profile(eq_id: str, apply_correction: bool = False):
    """
    Get a specific EQ profile with APO format preview.

    Args:
        eq_id: EQ profile ID
        apply_correction: If True, apply Modern Target (KB5000_7) correction
    """
    try:
        db = get_opra_database()
        eq_data = db.get_eq_profile(eq_id)
        if eq_data is None:
            raise HTTPException(
                status_code=404, detail=f"EQ profile '{eq_id}' not found"
            )

        # Convert to APO format
        apo_profile = convert_opra_to_apo(eq_data)

        # Apply Modern Target correction if requested
        if apply_correction:
            apo_profile = apply_modern_target_correction(apo_profile)

        return {
            "id": eq_id,
            "name": eq_data.get("name", ""),
            "author": eq_data.get("author", ""),
            "details": apo_profile.details,  # Includes correction info if applied
            "parameters": eq_data.get("parameters", {}),
            "apo_format": apo_profile.to_apo_format(),
            "modern_target_applied": apply_correction,
            "attribution": {
                "license": "CC BY-SA 4.0",
                "source": "OPRA Project",
                "author": eq_data.get("author", "unknown"),
            },
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/apply/{eq_id}", response_model=ApiResponse)
async def opra_apply_eq(eq_id: str, apply_correction: bool = False):
    """
    Apply an OPRA EQ profile.
    Converts to APO format, saves to data/EQ, and activates.

    Args:
        eq_id: EQ profile ID
        apply_correction: If True, apply Modern Target (KB5000_7) correction
    """
    try:
        db = get_opra_database()
        eq_data = db.get_eq_profile(eq_id)
        if eq_data is None:
            raise HTTPException(
                status_code=404, detail=f"EQ profile '{eq_id}' not found"
            )

        # Convert to APO format
        apo_profile = convert_opra_to_apo(eq_data)

        # Apply Modern Target correction if requested
        if apply_correction:
            apo_profile = apply_modern_target_correction(apo_profile)

        apo_content = apo_profile.to_apo_format()

        # Add attribution comment
        author = eq_data.get("author", "unknown")
        header = f"# OPRA: {eq_data.get('name', eq_id)}\n"
        header += f"# Author: {author}\n"
        header += f"# Details: {apo_profile.details}\n"
        if apply_correction:
            header += "# Modern Target: KB5000_7 correction applied\n"
        header += "# License: CC BY-SA 4.0\n"
        header += "# Source: https://github.com/opra-project/OPRA\n\n"
        apo_content = header + apo_content

        # Save to EQ profiles directory
        EQ_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        # Use safe filename: replace problematic chars
        safe_name = eq_id.replace("/", "_").replace("\\", "_")
        suffix = "_kb5000_7" if apply_correction else ""
        profile_path = EQ_PROFILES_DIR / f"opra_{safe_name}{suffix}.txt"
        profile_path.write_text(apo_content)

        # Update config to use this profile
        config = load_config()
        config.eq_profile = profile_path.stem
        save_config(config)

        return ApiResponse(
            success=True,
            message=f"Applied '{eq_data.get('name', eq_id)}'",
            data={
                "profile_name": profile_path.stem,
                "modern_target_applied": apply_correction,
                "author": author,
            },
            restart_required=True,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
