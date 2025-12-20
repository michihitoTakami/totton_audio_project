"""Admin authentication helpers for management endpoints."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException

ADMIN_TOKEN_ENV = "MAGICBOX_ADMIN_TOKEN"


def get_admin_token() -> str | None:
    """Return configured admin token from environment, if present."""
    token = os.getenv(ADMIN_TOKEN_ENV, "").strip()
    return token or None


def require_admin_token(x_admin_token: str | None = Header(None)) -> None:
    """Validate X-Admin-Token header against configured token."""
    configured = get_admin_token()
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="Admin token is not configured",
        )
    if not x_admin_token or x_admin_token != configured:
        raise HTTPException(status_code=401, detail="Unauthorized")
