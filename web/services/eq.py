"""EQ profile parsing, validation, and file handling."""

import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from ..constants import (
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
    SAFE_FILENAME_PATTERN,
    SAFE_PROFILE_NAME_PATTERN,
)

# ============================================================================
# Filter Type Definitions
# ============================================================================

# Equalizer APO filter type parameter requirements
# Reference: https://sourceforge.net/p/equalizerapo/wiki/Configuration%20reference/
FILTER_TYPE_PARAMS = {
    # Peaking filters - require Fc, Gain, Q
    "PK": {"fc": True, "gain": True, "q": True},
    "MODAL": {"fc": True, "gain": True, "q": True},
    "PEQ": {"fc": True, "gain": True, "q": True},
    # Low-pass filters
    "LP": {"fc": True, "gain": False, "q": False},  # Basic low-pass, no Q
    "LPQ": {"fc": True, "gain": False, "q": False},  # Low-pass with optional Q
    # High-pass filters
    "HP": {"fc": True, "gain": False, "q": False},  # Basic high-pass, no Q
    "HPQ": {"fc": True, "gain": False, "q": False},  # High-pass with optional Q
    # Band-pass filter
    "BP": {"fc": True, "gain": False, "q": False},
    # Notch filter
    "NO": {"fc": True, "gain": False, "q": False},
    # All-pass filter
    "AP": {"fc": True, "gain": True, "q": True},
    # Shelf filters (generic)
    "LS": {"fc": True, "gain": True, "q": True},  # Low-shelf
    "HS": {"fc": True, "gain": True, "q": True},  # High-shelf
    # Shelf filters with specific Q characteristics
    "LSC": {"fc": True, "gain": True, "q": False},  # Low-shelf constant Q
    "HSC": {"fc": True, "gain": True, "q": False},  # High-shelf constant Q
    "LSQ": {"fc": True, "gain": True, "q": True},  # Low-shelf with Q
    "HSQ": {"fc": True, "gain": True, "q": True},  # High-shelf with Q
    # Fixed-slope shelf filters
    "LS 6DB": {"fc": True, "gain": True, "q": False},  # 6dB/oct low-shelf
    "LS 12DB": {"fc": True, "gain": True, "q": False},  # 12dB/oct low-shelf
    "HS 6DB": {"fc": True, "gain": True, "q": False},  # 6dB/oct high-shelf
    "HS 12DB": {"fc": True, "gain": True, "q": False},  # 12dB/oct high-shelf
}


def is_safe_profile_name(name: str | None) -> bool:
    """
    Check if profile name is safe (no path traversal risk).

    Args:
        name: Profile name to validate (can be None)

    Returns:
        True if name is safe, False otherwise
    """
    if not name:
        return True  # None/empty is valid (means no profile)

    # Check against safe pattern
    if not SAFE_PROFILE_NAME_PATTERN.match(name):
        return False

    # Reject path traversal patterns
    if ".." in name or name.startswith("."):
        return False

    return True


def sanitize_filename(filename: str) -> str | None:
    """
    Sanitize filename for security.

    Returns sanitized filename if valid, None if invalid.
    Prevents path traversal attacks and ensures safe characters.
    """
    if not filename:
        return None

    # Normalize path separators (handle Windows paths on Unix)
    # Replace backslashes with forward slashes before extracting basename
    normalized = filename.replace("\\", "/")

    # Remove any path components (prevent path traversal)
    # Use split to handle both Unix and Windows-style paths
    basename = normalized.split("/")[-1]

    # Check against safe pattern
    if not SAFE_FILENAME_PATTERN.match(basename):
        return None

    # Additional check: no double dots (extra protection)
    if ".." in basename:
        return None

    return basename


def _parse_filter_line(line: str) -> dict[str, Any] | None:
    """
    Parse a single filter line with flexible parameter matching.

    Supports:
    - ON/OFF state
    - Optional Gain and Q parameters
    - BW (bandwidth) as alternative to Q
    - Oct (octaves) as alternative to Q

    Returns dict with parsed values or None if parsing fails.
    """
    # Basic pattern: Filter N: [ON|OFF] TYPE Fc FREQ [Hz] (N optional)
    base_pattern = r"Filter\s*(\d+)?\s*:\s+(ON|OFF)\s+(.+?)\s+Fc\s+([\d.]+)\s*(?:Hz)?"

    match = re.match(base_pattern, line, re.IGNORECASE)
    if not match:
        return None

    result = {
        "filter_num": int(match.group(1)) if match.group(1) else None,
        "enabled": match.group(2).upper() == "ON",
        "filter_type": match.group(3).strip().upper(),
        "frequency": float(match.group(4)),
        "gain": None,
        "q": None,
        "bw": None,
        "oct": None,
    }

    # Extract remaining part after frequency
    remainder = line[match.end() :].strip()

    # Try to extract Gain
    gain_match = re.search(r"Gain\s+([-+]?\d+\.?\d*)\s*dB", remainder, re.IGNORECASE)
    if gain_match:
        result["gain"] = float(gain_match.group(1))

    # Try to extract Q
    q_match = re.search(r"Q\s+([\d.]+)", remainder, re.IGNORECASE)
    if q_match:
        result["q"] = float(q_match.group(1))

    # Try to extract BW (bandwidth)
    bw_match = re.search(r"BW\s+([\d.]+)\s*(?:Hz)?", remainder, re.IGNORECASE)
    if bw_match:
        result["bw"] = float(bw_match.group(1))

    # Try to extract Oct (octaves)
    oct_match = re.search(r"BW\s+oct\s+([\d.]+)", remainder, re.IGNORECASE)
    if oct_match:
        result["oct"] = float(oct_match.group(1))

    return result


def validate_eq_profile_content(content: str) -> dict[str, Any]:
    """
    Validate EQ profile content for correctness and safety.

    Enhanced validation supporting:
    - All Equalizer APO filter types
    - ON/OFF filter states
    - Optional Gain and Q parameters (filter-type dependent)
    - BW and Oct alternatives to Q

    Returns:
        dict with keys:
        - valid: bool
        - errors: list[str] (validation errors)
        - warnings: list[str] (non-fatal issues)
        - preamp_db: float | None
        - filter_count: int
        - recommended_preamp_db: float
    """
    errors: list[str] = []
    warnings: list[str] = []
    preamp_db: float | None = None
    filter_count = 0
    max_positive_gain = 0.0
    recommended_preamp_db = 0.0

    if not content or not content.strip():
        return {
            "valid": False,
            "errors": ["Empty file"],
            "warnings": [],
            "preamp_db": None,
            "filter_count": 0,
            "recommended_preamp_db": 0.0,
        }

    lines = content.strip().split("\n")

    # Check for Preamp line
    preamp_found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Preamp:"):
            preamp_found = True
            # Parse preamp value
            preamp_match = re.search(
                r"Preamp:\s*([-+]?\d+\.?\d*)\s*[dD][bB]?", stripped
            )
            if preamp_match:
                try:
                    preamp_db = float(preamp_match.group(1))
                    if preamp_db < PREAMP_MIN_DB or preamp_db > PREAMP_MAX_DB:
                        errors.append(
                            f"Preamp {preamp_db}dB out of range "
                            f"({PREAMP_MIN_DB}dB to {PREAMP_MAX_DB}dB)"
                        )
                except ValueError:
                    errors.append(f"Invalid Preamp value: {stripped}")
            else:
                warnings.append(f"Could not parse Preamp value: {stripped}")
            break

    if not preamp_found:
        errors.append("Missing 'Preamp:' line")

    # Parse and validate filter lines
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # Skip comments and empty lines
        if not stripped or stripped.startswith("#"):
            continue

        # Skip Preamp line (already processed)
        if lower.startswith("preamp:"):
            continue

        # Check if it's a Filter line
        if lower.startswith("filter ") or lower.startswith("filter:"):
            filter_count += 1
            parsed = _parse_filter_line(stripped)

            if not parsed:
                # Truncate long lines for readability
                display_line = stripped[:50] + "..." if len(stripped) > 50 else stripped
                warnings.append(f"Could not parse filter line: {display_line}")
                continue

            filter_num = parsed["filter_num"]
            filter_label = filter_num if filter_num is not None else filter_count
            filter_type = parsed["filter_type"]
            freq = parsed["frequency"]
            gain = parsed["gain"]
            q = parsed["q"]

            # Validate filter type
            if filter_type not in FILTER_TYPE_PARAMS:
                warnings.append(f"Filter {filter_label}: Unknown type '{filter_type}'")
            else:
                # Check parameter requirements for this filter type
                params = FILTER_TYPE_PARAMS[filter_type]

                # Check Gain requirement
                if params["gain"] and gain is None:
                    errors.append(
                        f"Filter {filter_label}: Type '{filter_type}' requires Gain parameter"
                    )

                # Check Q requirement (or BW/Oct alternatives)
                if (
                    params["q"]
                    and q is None
                    and parsed["bw"] is None
                    and parsed["oct"] is None
                ):
                    errors.append(
                        f"Filter {filter_label}: Type '{filter_type}' requires Q (or BW/Oct) parameter"
                    )

            # Validate frequency
            if freq < FREQ_MIN_HZ or freq > FREQ_MAX_HZ:
                errors.append(
                    f"Filter {filter_label}: Frequency {freq}Hz out of range "
                    f"({FREQ_MIN_HZ}Hz to {FREQ_MAX_HZ}Hz)"
                )

            # Validate gain if present
            if gain is not None:
                if gain < GAIN_MIN_DB or gain > GAIN_MAX_DB:
                    errors.append(
                        f"Filter {filter_label}: Gain {gain}dB out of range "
                        f"({GAIN_MIN_DB}dB to {GAIN_MAX_DB}dB)"
                    )
                elif parsed["enabled"] and gain > max_positive_gain:
                    max_positive_gain = gain

            # Validate Q if present
            if q is not None:
                if q < Q_MIN or q > Q_MAX:
                    errors.append(
                        f"Filter {filter_label}: Q {q} out of range ({Q_MIN} to {Q_MAX})"
                    )

    # Check filter count limit
    if filter_count > MAX_EQ_FILTERS:
        errors.append(
            f"Too many filters ({filter_count}). Maximum allowed: {MAX_EQ_FILTERS}"
        )

    if filter_count == 0 and preamp_found:
        warnings.append("No filter lines found (only Preamp)")

    if max_positive_gain > 0:
        recommended_preamp_db = -max_positive_gain
        if preamp_db is not None and preamp_db > recommended_preamp_db:
            warnings.append(
                (
                    f"Preamp {preamp_db}dB may clip (max boost +{max_positive_gain}dB). "
                    f"Recommended Preamp: {recommended_preamp_db}dB or lower."
                )
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "preamp_db": preamp_db,
        "filter_count": filter_count,
        "recommended_preamp_db": recommended_preamp_db,
    }


async def read_and_validate_upload(file: UploadFile) -> tuple[str, str, dict]:
    """
    Common validation logic for EQ profile upload.

    Args:
        file: FastAPI UploadFile object

    Returns:
        Tuple of (content, safe_filename, validation_result)

    Raises:
        HTTPException on validation failure
    """
    # Check file extension
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    # Sanitize filename (security: prevent path traversal)
    safe_filename = sanitize_filename(file.filename)
    if not safe_filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Use only letters, numbers, underscores, hyphens, and dots.",
        )

    # Read file content with size limit
    try:
        content_bytes = await file.read()
        if len(content_bytes) > MAX_EQ_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_EQ_FILE_SIZE // (1024 * 1024)}MB",
            )
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")

    # Validate content
    validation = validate_eq_profile_content(content)
    validation["size_bytes"] = len(content_bytes)

    return content, safe_filename, validation


def parse_eq_profile_content(file_path: Path) -> dict[str, Any]:
    """
    Parse EQ profile file and return structured content.

    Distinguishes between:
    - OPRA profiles (with # OPRA: header)
    - Custom profiles (user uploaded)

    For OPRA profiles with Modern Target correction, separates
    OPRA filters from original additions.

    Returns:
        dict with keys:
        - source_type: "opra" | "custom"
        - has_modern_target: bool
        - opra_info: dict | None (author, license, source for OPRA)
        - opra_filters: list[str]
        - original_filters: list[str]
        - raw_content: str
        - error: str (only if error occurred)
    """
    if not file_path.exists():
        return {"error": "File not found"}

    try:
        content = file_path.read_text(encoding="utf-8")
    except IOError as e:
        return {"error": f"Failed to read file: {e}"}

    lines = content.strip().split("\n")

    # Detect profile type
    is_opra = any(line.startswith("# OPRA:") for line in lines)
    has_modern_target = any("Modern Target" in line for line in lines)

    # Parse header info for OPRA profiles
    opra_info: dict[str, str] = {}
    if is_opra:
        for line in lines:
            if line.startswith("# OPRA:"):
                opra_info["product"] = line.replace("# OPRA:", "").strip()
            elif line.startswith("# Author:"):
                opra_info["author"] = line.replace("# Author:", "").strip()
            elif line.startswith("# License:"):
                opra_info["license"] = line.replace("# License:", "").strip()
            elif line.startswith("# Source:"):
                opra_info["source"] = line.replace("# Source:", "").strip()
            elif line.startswith("# Details:"):
                opra_info["details"] = line.replace("# Details:", "").strip()

    # Extract filter lines (Preamp and Filter N:)
    filter_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Preamp:") or stripped.startswith("Filter "):
            filter_lines.append(stripped)

    # For OPRA with Modern Target, separate original addition
    # KB5000_7 correction: Fc 5366 Hz, Gain 2.8 dB, Q 1.5
    opra_filters: list[str] = []
    original_filters: list[str] = []

    if is_opra and has_modern_target:
        # Detection patterns for KB5000_7 correction filter
        fc_pattern = "Fc 5366"
        gain_pattern = "Gain 2.8"
        q_pattern = "Q 1.5"

        for line in filter_lines:
            # Check if this is the KB5000_7 correction filter
            if fc_pattern in line and gain_pattern in line and q_pattern in line:
                original_filters.append(line)
            else:
                opra_filters.append(line)
    else:
        opra_filters = filter_lines

    return {
        "source_type": "opra" if is_opra else "custom",
        "has_modern_target": has_modern_target,
        "opra_info": opra_info if is_opra else None,
        "opra_filters": opra_filters,
        "original_filters": original_filters,
        "raw_content": content,
    }
