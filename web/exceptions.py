"""Custom exception handlers for unified error responses."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import ErrorResponse


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on the FastAPI app."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions with unified format."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                detail=str(exc.detail),
                error_code=f"HTTP_{exc.status_code}",
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle validation errors with unified format."""
        # Format validation errors nicely
        errors = []
        for error in exc.errors():
            loc = ".".join(str(x) for x in error["loc"])
            errors.append(f"{loc}: {error['msg']}")

        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                detail="; ".join(errors),
                error_code="VALIDATION_ERROR",
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail="Internal server error",
                error_code="INTERNAL_ERROR",
            ).model_dump(),
        )
