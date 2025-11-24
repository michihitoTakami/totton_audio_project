"""Custom exception handlers for unified error responses.

Implements RFC 9457 Problem Details for HTTP APIs.
All error responses use application/problem+json content type.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .error_codes import ErrorCategory, get_error_mapping
from .models import ErrorResponse, InnerError
from .services.daemon_client import DaemonError

# RFC 9457 content type
PROBLEM_JSON_MEDIA_TYPE = "application/problem+json"


def _error_code_to_type(error_code: str) -> str:
    """Convert error code to URI type for RFC 9457.

    Example: DAC_RATE_NOT_SUPPORTED -> /errors/dac-rate-not-supported
    """
    slug = error_code.lower().replace("_", "-")
    return f"/errors/{slug}"


def _create_error_response(
    status_code: int,
    detail: str,
    error_code: str,
    category: str | None = None,
    title: str | None = None,
    inner_error: InnerError | None = None,
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        type=_error_code_to_type(error_code),
        title=title,
        status=status_code,
        detail=detail,
        error_code=error_code,
        category=category,
        inner_error=inner_error,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on the FastAPI app."""

    @app.exception_handler(DaemonError)
    async def daemon_error_handler(request: Request, exc: DaemonError) -> JSONResponse:
        """Handle DaemonError from C++ Audio Engine.

        Maps error codes to HTTP status codes and creates RFC 9457 responses.
        """
        mapping = get_error_mapping(exc.error_code)

        # Build inner_error model if present
        inner_error = None
        if exc.inner_error:
            inner_error = InnerError(
                cpp_code=exc.inner_error.get("cpp_code"),
                cpp_message=exc.inner_error.get("cpp_message"),
                alsa_errno=exc.inner_error.get("alsa_errno"),
                alsa_func=exc.inner_error.get("alsa_func"),
                cuda_error=exc.inner_error.get("cuda_error"),
                zmq_errno=exc.inner_error.get("zmq_errno"),
            )

        response = _create_error_response(
            status_code=mapping.http_status,
            detail=exc.message,
            error_code=exc.error_code,
            category=mapping.category.value,
            title=mapping.title,
            inner_error=inner_error,
        )

        return JSONResponse(
            status_code=mapping.http_status,
            content=response.model_dump(exclude_none=True),
            media_type=PROBLEM_JSON_MEDIA_TYPE,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions with RFC 9457 format.

        Preserves structured details (dict) if provided, otherwise uses string.
        """
        detail = exc.detail if isinstance(exc.detail, (str, dict)) else str(exc.detail)
        error_code = f"HTTP_{exc.status_code}"

        response = _create_error_response(
            status_code=exc.status_code,
            detail=detail if isinstance(detail, str) else str(detail),
            error_code=error_code,
            title=_http_status_to_title(exc.status_code),
        )

        # If detail is a dict, use it directly
        content = response.model_dump(exclude_none=True)
        if isinstance(detail, dict):
            content["detail"] = detail

        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            media_type=PROBLEM_JSON_MEDIA_TYPE,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle validation errors with RFC 9457 format."""
        errors = []
        for error in exc.errors():
            loc = ".".join(str(x) for x in error["loc"])
            errors.append(f"{loc}: {error['msg']}")

        response = _create_error_response(
            status_code=422,
            detail="; ".join(errors),
            error_code="VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION.value,
            title="Validation Error",
        )

        return JSONResponse(
            status_code=422,
            content=response.model_dump(exclude_none=True),
            media_type=PROBLEM_JSON_MEDIA_TYPE,
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions with RFC 9457 format."""
        response = _create_error_response(
            status_code=500,
            detail="Internal server error",
            error_code="INTERNAL_ERROR",
            category=ErrorCategory.INTERNAL.value,
            title="Internal Server Error",
        )

        return JSONResponse(
            status_code=500,
            content=response.model_dump(exclude_none=True),
            media_type=PROBLEM_JSON_MEDIA_TYPE,
        )


def _http_status_to_title(status_code: int) -> str:
    """Convert HTTP status code to human-readable title."""
    titles = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        409: "Conflict",
        422: "Unprocessable Entity",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
    }
    return titles.get(status_code, f"HTTP {status_code}")
