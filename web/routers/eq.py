"""EQ profile management endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from ..constants import EQ_PROFILES_DIR
from ..models import (
    ApiResponse,
    EqActiveResponse,
    EqProfileInfo,
    EqProfilesResponse,
    EqValidationResponse,
)
from ..services import (
    is_safe_profile_name,
    load_config,
    parse_eq_profile_content,
    read_and_validate_upload,
    save_config,
)

router = APIRouter(prefix="/eq", tags=["eq"])


def validate_profile_name(name: str) -> None:
    """
    Validate profile name to prevent path traversal attacks.

    Raises HTTPException 400 if name contains unsafe characters.
    """
    if not is_safe_profile_name(name):
        raise HTTPException(
            status_code=400,
            detail="Invalid profile name. Cannot start with '.' or contain '..'.",
        )


@router.get("/profiles", response_model=EqProfilesResponse)
async def list_eq_profiles():
    """
    List available EQ profiles in data/EQ directory.
    Returns extended info including file size and profile type.
    """
    profiles = []
    if EQ_PROFILES_DIR.exists():
        for f in EQ_PROFILES_DIR.iterdir():
            if f.is_file() and f.suffix == ".txt":
                # Determine profile type by checking content
                profile_type = "custom"
                filter_count = 0
                try:
                    content = f.read_text(encoding="utf-8")
                    if "# OPRA:" in content:
                        profile_type = "opra"
                    # Count filter lines
                    filter_count = sum(
                        1
                        for line in content.split("\n")
                        if line.strip().startswith("Filter ")
                    )
                except IOError:
                    pass

                stat = f.stat()
                profiles.append(
                    EqProfileInfo(
                        name=f.stem,
                        filename=f.name,
                        path=str(f),
                        size=stat.st_size,
                        modified=stat.st_mtime,
                        type=profile_type,
                        filter_count=filter_count,
                    )
                )
    # Sort by modified time (newest first)
    profiles.sort(key=lambda x: x.modified, reverse=True)
    return EqProfilesResponse(profiles=profiles)


@router.post("/validate", response_model=EqValidationResponse)
async def validate_eq_profile(file: UploadFile):
    """
    Validate an EQ profile file before importing.
    Checks format, parameter ranges, and security constraints.
    Does not save the file.
    """
    _, safe_filename, validation = await read_and_validate_upload(file)

    # Check if file already exists
    dest_path = EQ_PROFILES_DIR / safe_filename
    file_exists = dest_path.exists()

    return EqValidationResponse(
        valid=validation["valid"],
        errors=validation["errors"],
        warnings=validation["warnings"],
        preamp_db=validation["preamp_db"],
        filter_count=validation["filter_count"],
        filename=safe_filename,
        file_exists=file_exists,
        size_bytes=validation["size_bytes"],
        recommended_preamp_db=validation["recommended_preamp_db"],
    )


@router.post("/import", response_model=ApiResponse)
async def import_eq_profile(file: UploadFile, overwrite: bool = False):
    """
    Import an EQ profile file (AutoEq/Equalizer APO format).

    Security features:
    - File size limit (1MB)
    - Filename sanitization (path traversal prevention)
    - Content validation (format, parameter ranges)
    - Overwrite protection (requires explicit flag)
    """
    content, safe_filename, validation = await read_and_validate_upload(file)

    # Reject invalid content
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid EQ profile: {'; '.join(validation['errors'])}",
        )

    # Ensure EQ profiles directory exists
    EQ_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing file
    dest_path = EQ_PROFILES_DIR / safe_filename
    if dest_path.exists() and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Profile '{safe_filename}' already exists. Set overwrite=true to replace.",
        )

    # Save file
    try:
        dest_path.write_text(content, encoding="utf-8")
        return ApiResponse(
            success=True,
            message=f"Profile '{safe_filename}' imported successfully",
            data={
                "path": str(dest_path),
                "filename": safe_filename,
                "filter_count": validation["filter_count"],
                "preamp_db": validation["preamp_db"],
                "warnings": validation["warnings"],
            },
        )
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")


@router.post("/activate/{name}", response_model=ApiResponse)
async def activate_eq_profile(name: str):
    """Activate an EQ profile by name."""
    # Validate profile name (prevent path traversal)
    validate_profile_name(name)

    # Find profile file
    profile_path = EQ_PROFILES_DIR / f"{name}.txt"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")

    # Update config
    config = load_config()
    config.eq_enabled = True
    config.eq_profile = name
    config.eq_profile_path = str(profile_path)
    if save_config(config):
        return ApiResponse(
            success=True,
            message=f"Profile '{name}' activated",
            restart_required=True,
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to save config")


@router.post("/deactivate", response_model=ApiResponse)
async def deactivate_eq():
    """Deactivate current EQ profile."""
    config = load_config()
    config.eq_enabled = False
    config.eq_profile = None
    config.eq_profile_path = None
    if save_config(config):
        return ApiResponse(
            success=True, message="EQ deactivated", restart_required=True
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to save config")


@router.delete("/profiles/{name}", response_model=ApiResponse)
async def delete_eq_profile(name: str):
    """Delete an EQ profile."""
    # Validate profile name (prevent path traversal)
    validate_profile_name(name)

    profile_path = EQ_PROFILES_DIR / f"{name}.txt"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")

    # Check if currently active
    config = load_config()
    if config.eq_profile == name:
        config.eq_enabled = False
        config.eq_profile = None
        config.eq_profile_path = None
        save_config(config)

    try:
        profile_path.unlink()
        return ApiResponse(success=True, message=f"Profile '{name}' deleted")
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")


@router.get("/active", response_model=EqActiveResponse)
async def get_active_eq():
    """Get the currently active EQ profile with parsed content."""
    config = load_config()

    if not config.eq_enabled or not config.eq_profile or not config.eq_profile_path:
        return EqActiveResponse(active=False)

    # Defense-in-depth: validate profile name from config
    # This catches cases where config was tampered with or contains unsafe values
    if not is_safe_profile_name(config.eq_profile):
        return EqActiveResponse(
            active=True,
            name=config.eq_profile,
            error="Invalid profile name in config",
        )

    profile_path = Path(config.eq_profile_path)
    if not profile_path.exists():
        return EqActiveResponse(
            active=True,
            name=config.eq_profile,
            error="Profile file not found",
        )

    parsed = parse_eq_profile_content(profile_path)

    return EqActiveResponse(
        active=True,
        name=config.eq_profile,
        source_type=parsed["source_type"],
        has_modern_target=parsed["has_modern_target"],
        opra_info=parsed["opra_info"],
        opra_filters=parsed["opra_filters"],
        original_filters=parsed["original_filters"],
    )
