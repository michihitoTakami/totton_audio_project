"""Admin authentication helpers for management endpoints."""

from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

ADMIN_USER_ENV = "MAGICBOX_ADMIN_USER"
ADMIN_PASSWORD_ENV = "MAGICBOX_ADMIN_PASSWORD"  # pragma: allowlist secret

_security = HTTPBasic(auto_error=False)


def _get_admin_credentials() -> tuple[str, str] | None:
    """Return configured admin credentials from environment, if present."""
    user = os.getenv(ADMIN_USER_ENV, "").strip()
    password = os.getenv(ADMIN_PASSWORD_ENV, "").strip()
    if not user or not password:
        return None
    return user, password


def require_admin_basic(
    credentials: HTTPBasicCredentials | None = Depends(_security),
) -> None:
    """Validate HTTP Basic credentials against configured admin user."""
    configured = _get_admin_credentials()
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="Admin credentials are not configured",
        )

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )

    expected_user, expected_password = configured
    user_ok = secrets.compare_digest(credentials.username, expected_user)
    pass_ok = secrets.compare_digest(credentials.password, expected_password)
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Unauthorized")
