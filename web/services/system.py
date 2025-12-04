"""System-related service functions."""

import logging
from pathlib import Path
from typing import Optional

from .config import load_config

logger = logging.getLogger(__name__)

# Maximum log file size to read (100MB)
MAX_LOG_FILE_SIZE = 100 * 1024 * 1024


def read_system_logs(
    lines: int = 100,
    offset: int = 0,
    level_filter: Optional[str] = None,
) -> dict:
    """
    Read system logs from daemon log file.

    Args:
        lines: Number of lines to return (default: 100, max: 1000)
        offset: Number of lines to skip from the beginning (default: 0)
        level_filter: Filter by log level (e.g., "debug", "info", "warning", "error")

    Returns:
        Dictionary containing:
            - logs: List of log lines
            - total_lines: Total number of lines in the file
            - file_path: Path to the log file
            - file_size: Size of the log file in bytes

    Raises:
        ValueError: If log file is too large or path is invalid
        FileNotFoundError: If log file doesn't exist
    """
    # Load config to get log file path
    config = load_config()
    log_path_str = config.model_dump().get("logging", {}).get("filePath")

    if not log_path_str:
        log_path_str = "/var/log/gpu_upsampler/daemon.log"

    log_path = Path(log_path_str)

    # Security: Ensure log file path is absolute
    if not log_path.is_absolute():
        raise ValueError("Log file path must be absolute")

    # Check if file exists
    if not log_path.exists():
        logger.warning(f"Log file not found: {log_path}")
        return {
            "logs": [],
            "total_lines": 0,
            "file_path": str(log_path),
            "file_size": 0,
        }

    # Check file size
    file_size = log_path.stat().st_size
    if file_size > MAX_LOG_FILE_SIZE:
        raise ValueError(
            f"Log file too large: {file_size} bytes (max: {MAX_LOG_FILE_SIZE})"
        )

    # Read all lines from file
    try:
        with open(log_path, "r", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read log file: {e}")
        raise

    # Filter by level if specified
    if level_filter:
        level_upper = level_filter.upper()
        filtered_lines = [line for line in all_lines if level_upper in line.upper()]
    else:
        filtered_lines = all_lines

    total = len(filtered_lines)

    # Apply pagination
    start = offset
    end = start + min(lines, 1000)  # Cap at 1000 lines per request
    page_lines = filtered_lines[start:end]

    return {
        "logs": [line.rstrip("\n") for line in page_lines],
        "total_lines": total,
        "file_path": str(log_path),
        "file_size": file_size,
    }
