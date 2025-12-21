"""
GPU Upsampler Web Control API
FastAPI-based control interface for the GPU audio upsampler daemon.
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .exceptions import register_exception_handlers
from .i18n import normalize_lang
from .models import ApiResponse
from .routers import (
    crossfeed_router,
    dac_router,
    daemon_router,
    delimiter_router,
    eq_router,
    i2s_router,
    opra_router,
    opra_sync_router,
    output_mode_router,
    partitioned_router,
    pi_router,
    status_router,
)
from .templates.pages import (
    render_dashboard,
    render_eq_settings,
    render_pi_settings,
    render_system,
)

# OpenAPI tag descriptions
tags_metadata = [
    {
        "name": "status",
        "description": "System status and settings management",
    },
    {
        "name": "daemon",
        "description": "Audio daemon lifecycle and ZeroMQ communication",
    },
    {
        "name": "eq",
        "description": "EQ profile management (import, activate, delete)",
    },
    {
        "name": "opra",
        "description": "OPRA headphone database integration (CC BY-SA 4.0)",
    },
    {
        "name": "dac",
        "description": "DAC capability detection and sample rate filtering",
    },
    {
        "name": "crossfeed",
        "description": "Crossfeed (HRTF-based headphone virtualization) control",
    },
    {
        "name": "delimiter",
        "description": "De-limiter runtime control and status",
    },
    {
        "name": "output",
        "description": "Output mode selection and device preferences",
    },
    {
        "name": "pi",
        "description": "Raspberry Pi USB-I2S bridge proxy control",
    },
    {
        "name": "legacy",
        "description": "Deprecated endpoints - use alternatives instead",
    },
]


_logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    """環境変数を真偽値として解釈する."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_rtp_manager(app: FastAPI):
    """DIのオーバーライドを考慮してRTPマネージャを取得."""
    from .services.rtp_input import get_rtp_receiver_manager

    override = app.dependency_overrides.get(get_rtp_receiver_manager)
    if override:
        return override()
    return get_rtp_receiver_manager()


def create_app(*, enable_rtp: bool | None = None) -> FastAPI:
    """
    FastAPIアプリを生成する。

    - RTP はデフォルト無効（I2Sメイン運用のため）。
    - `MAGICBOX_ENABLE_RTP=true`（または enable_rtp=True）でのみ API 露出/自動起動対象になる。
    """
    resolved_enable_rtp = (
        _env_flag("MAGICBOX_ENABLE_RTP", False) if enable_rtp is None else enable_rtp
    )

    effective_tags = list(tags_metadata)
    if resolved_enable_rtp:
        effective_tags.extend(
            [
                {
                    "name": "rtp",
                    "description": "RTP ZeroMQブリッジのステータス/制御",
                },
                {
                    "name": "rtp-input",
                    "description": "RTP入力のステータス取得と制御 (GStreamer)",
                },
            ]
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle."""
        if resolved_enable_rtp:
            rtp_manager = _resolve_rtp_manager(app)
            rtp_autostart = _env_flag("MAGICBOX_RTP_AUTOSTART", False)
            if rtp_autostart:
                try:
                    await rtp_manager.start()
                    _logger.info("RTP input autostarted (MAGICBOX_RTP_AUTOSTART=true)")
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("RTP autostart failed: %s", exc)
            else:
                _logger.info("RTP autostart disabled (MAGICBOX_ENABLE_RTP=true)")

        yield
        if resolved_enable_rtp:
            try:
                rtp_manager = _resolve_rtp_manager(app)
                await rtp_manager.shutdown()
            except Exception as exc:  # noqa: BLE001
                _logger.warning("RTP shutdown encountered an error: %s", exc)

    app = FastAPI(
        lifespan=lifespan,
        title="GPU Upsampler Control",
        description="""
## GPU Audio Upsampler Web API

Control interface for the GPU-accelerated audio upsampler daemon.

### Features
- **Daemon Control**: Start/stop/restart the audio processing daemon
- **EQ Management**: Import and manage EQ profiles (AutoEq/Equalizer APO format)
- **OPRA Integration**: Access the OPRA headphone EQ database
- **Real-time Stats**: WebSocket endpoint for live statistics

### Authentication
Most endpoints do not require authentication (local network only).
    """,
        version="1.0.0",
        openapi_tags=effective_tags,
    )

    # Register exception handlers for unified error responses
    register_exception_handlers(app)

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include routers
    app.include_router(status_router)
    app.include_router(i2s_router)
    app.include_router(daemon_router)
    app.include_router(eq_router)
    app.include_router(opra_router)
    app.include_router(opra_sync_router)
    app.include_router(dac_router)
    app.include_router(crossfeed_router)
    app.include_router(delimiter_router)
    app.include_router(partitioned_router)
    app.include_router(output_mode_router)
    app.include_router(pi_router)
    if resolved_enable_rtp:
        from .routers.rtp import router as rtp_router
        from .routers.rtp_input import router as rtp_input_router

        app.include_router(rtp_router)
        app.include_router(rtp_input_router)

    return app


app = create_app()


# Legacy restart endpoint (forwards to daemon restart)
@app.post(
    "/restart",
    response_model=ApiResponse,
    tags=["legacy"],
    deprecated=True,
    summary="Restart daemon (DEPRECATED)",
    description="**Deprecated**: Use `POST /daemon/restart` instead.",
)
async def restart():
    """
    Legacy restart endpoint - forwards to daemon restart.

    **Deprecated**: Use `POST /daemon/restart` instead.
    """
    from .routers.daemon import daemon_restart

    return await daemon_restart()


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request, lang: str | None = None):
    """Serve the Dashboard page."""
    resolved_lang = _resolve_lang(request, lang)
    return _render_with_lang(
        content=render_dashboard(lang=resolved_lang), lang=resolved_lang
    )


@app.get("/eq", response_class=HTMLResponse)
async def eq_page(request: Request, lang: str | None = None):
    """Serve the EQ Settings page."""
    resolved_lang = _resolve_lang(request, lang)
    return _render_with_lang(
        content=render_eq_settings(lang=resolved_lang), lang=resolved_lang
    )


@app.get("/system", response_class=HTMLResponse)
async def system_page(request: Request, lang: str | None = None):
    """Serve the System page."""
    resolved_lang = _resolve_lang(request, lang)
    return _render_with_lang(
        content=render_system(lang=resolved_lang), lang=resolved_lang
    )


@app.get("/pi", response_class=HTMLResponse)
async def pi_page(request: Request, lang: str | None = None):
    """Serve the Pi Settings page."""
    resolved_lang = _resolve_lang(request, lang)
    return _render_with_lang(
        content=render_pi_settings(lang=resolved_lang), lang=resolved_lang
    )


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=11881)


# ============================================================================
# Helpers
# ============================================================================


def _resolve_lang(request: Request, lang: str | None) -> str:
    """
    Resolve the language by query param -> cookie -> default.
    """
    if lang:
        return normalize_lang(lang)
    cookie_lang = request.cookies.get("lang")
    if cookie_lang:
        return normalize_lang(cookie_lang)
    return "en"


def _render_with_lang(content: str, lang: str) -> HTMLResponse:
    """
    Wrap rendered HTML with a cookie that persists language choice.
    """
    response = HTMLResponse(content=content)
    response.set_cookie(
        "lang",
        lang,
        max_age=60 * 60 * 24 * 365,
        path="/",
        httponly=False,
        samesite="lax",
    )
    return response
