"""Web security utilities (admin API token enforcement)."""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, Request, status


def require_admin_token(
    request: Request, x_admin_token: str | None = Header(default=None)
) -> str:
    """
    Enforce admin token for protected endpoints.

    - Token is provided via `X-Admin-Token` header.
    - If `MAGICBOX_ADMIN_TOKEN` is not set, the check is skipped (backward compat).
    - Invalid or missing token returns 401.
    """
    expected = os.getenv("MAGICBOX_ADMIN_TOKEN")
    if not expected:
        return ""

    provided = x_admin_token or request.cookies.get("admin_token")
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing admin token"
        )
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token"
        )
    return provided


__all__ = ["require_admin_token"]
