"""Client for Raspberry Pi Control API (USB-I2S bridge)."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


class PiClientError(Exception):
    """Base class for Pi client errors."""


class PiConnectionError(PiClientError):
    """Raised when the Pi API is unreachable."""


class PiResponseError(PiClientError):
    """Raised when the Pi API returns a bad response."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


DEFAULT_PI_API_BASE = os.getenv("MAGICBOX_PI_API_BASE", "http://192.168.55.2:8081")
DEFAULT_PI_API_TIMEOUT_MS = int(os.getenv("MAGICBOX_PI_API_TIMEOUT_MS", "2000"))


class PiClient:
    """Thin synchronous HTTP client for the Pi control API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        resolved_base = (base_url or DEFAULT_PI_API_BASE).strip()
        if not resolved_base:
            raise PiResponseError("Pi API base URL is empty", status_code=503)
        self.base_url = resolved_base.rstrip("/")
        resolved_timeout = timeout_ms or DEFAULT_PI_API_TIMEOUT_MS
        self.timeout_sec = max(0.1, resolved_timeout / 1000.0)

    def status(self) -> dict[str, Any]:
        return self._request_json("GET", "/raspi/api/v1/status")

    def get_config(self) -> dict[str, Any]:
        return self._request_json("GET", "/raspi/api/v1/config")

    def update_config(
        self, payload: dict[str, Any], *, apply: bool = True
    ) -> dict[str, Any]:
        query = "true" if apply else "false"
        return self._request_json(
            "PUT",
            f"/raspi/api/v1/config?apply={query}",
            payload=payload,
        )

    def restart_bridge(self) -> dict[str, Any]:
        return self._request_json("POST", "/raspi/api/v1/actions/restart")

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, method=method, data=data, headers=headers)

        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = ""
            if exc.fp is not None:
                try:
                    body = exc.fp.read().decode("utf-8")
                except Exception:
                    body = ""
            message = body.strip() or exc.reason or "Pi API error"
            raise PiResponseError(message, status_code=exc.code) from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise PiConnectionError(str(reason)) from exc

        if not body:
            return {}
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PiResponseError("Invalid JSON from Pi API") from exc
        if not isinstance(data, dict):
            raise PiResponseError("Unexpected Pi API payload")
        return data


def get_pi_client() -> PiClient:
    return PiClient()


__all__ = [
    "PiClient",
    "PiClientError",
    "PiConnectionError",
    "PiResponseError",
    "get_pi_client",
]
