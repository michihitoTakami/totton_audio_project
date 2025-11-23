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


def validate_eq_profile_content(content: str) -> dict[str, Any]:
    """
    Validate EQ profile content for correctness and safety.

    Returns:
        dict with keys:
        - valid: bool
        - errors: list[str] (validation errors)
        - warnings: list[str] (non-fatal issues)
        - preamp_db: float | None
        - filter_count: int
    """
    errors: list[str] = []
    warnings: list[str] = []
    preamp_db: float | None = None
    filter_count = 0

    if not content or not content.strip():
        return {
            "valid": False,
            "errors": ["Empty file"],
            "warnings": [],
            "preamp_db": None,
            "filter_count": 0,
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
    filter_pattern = re.compile(
        r"Filter\s+(\d+):\s+ON\s+(\w+)\s+Fc\s+([\d.]+)\s*Hz?\s+"
        r"Gain\s+([-+]?\d+\.?\d*)\s*dB\s+Q\s+([\d.]+)",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith("#"):
            continue

        # Skip Preamp line (already processed)
        if stripped.startswith("Preamp:"):
            continue

        # Check if it's a Filter line
        if stripped.startswith("Filter "):
            filter_count += 1
            match = filter_pattern.match(stripped)
            if match:
                filter_num = int(match.group(1))
                filter_type = match.group(2).upper()
                freq = float(match.group(3))
                gain = float(match.group(4))
                q = float(match.group(5))

                # Validate filter type
                valid_types = {"PK", "LS", "HS", "LP", "HP", "LSC", "HSC", "LSQ", "HSQ"}
                if filter_type not in valid_types:
                    warnings.append(
                        f"Filter {filter_num}: Unknown type '{filter_type}'"
                    )

                # Validate frequency
                if freq < FREQ_MIN_HZ or freq > FREQ_MAX_HZ:
                    errors.append(
                        f"Filter {filter_num}: Frequency {freq}Hz out of range "
                        f"({FREQ_MIN_HZ}Hz to {FREQ_MAX_HZ}Hz)"
                    )

                # Validate gain
                if gain < GAIN_MIN_DB or gain > GAIN_MAX_DB:
                    errors.append(
                        f"Filter {filter_num}: Gain {gain}dB out of range "
                        f"({GAIN_MIN_DB}dB to {GAIN_MAX_DB}dB)"
                    )

                # Validate Q
                if q < Q_MIN or q > Q_MAX:
                    errors.append(
                        f"Filter {filter_num}: Q {q} out of range ({Q_MIN} to {Q_MAX})"
                    )
            else:
                # Truncate long lines for readability
                display_line = stripped[:50] + "..." if len(stripped) > 50 else stripped
                warnings.append(f"Could not parse filter line: {display_line}")

    # Check filter count limit
    if filter_count > MAX_EQ_FILTERS:
        errors.append(
            f"Too many filters ({filter_count}). Maximum allowed: {MAX_EQ_FILTERS}"
        )

    if filter_count == 0 and preamp_found:
        warnings.append("No filter lines found (only Preamp)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "preamp_db": preamp_db,
        "filter_count": filter_count,
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
