"""
GPU Upsampler Web Control API
FastAPI-based control interface for the GPU audio upsampler daemon.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .exceptions import register_exception_handlers
from .i18n import normalize_lang
from .models import ApiResponse, TcpInputStatusResponse, TcpInputTelemetry
from .routers import (
    crossfeed_router,
    dac_router,
    daemon_router,
    eq_router,
    opra_router,
    output_mode_router,
    partitioned_router,
    status_router,
    tcp_input_router,
    rtp_router,
    rtp_input_router,
)
from .services import get_daemon_client
from .services.tcp_input import (
    TcpTelemetryPoller,
    TcpTelemetryStore,
    parse_tcp_telemetry,
)
from .templates.pages import (
    render_dashboard,
    render_eq_settings,
    render_system,
    render_tcp_input,
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
        "name": "output",
        "description": "Output mode selection and device preferences",
    },
    {
        "name": "legacy",
        "description": "Deprecated endpoints - use alternatives instead",
    },
    {
        "name": "tcp-input",
        "description": "TCP入力のステータス取得と制御 (ZeroMQ経由)",
    },
    {
        "name": "rtp",
        "description": "RTP ZeroMQブリッジのステータス/制御",
    },
    {
        "name": "rtp-input",
        "description": "RTP入力のステータス取得と制御 (GStreamer)",
    },
]


_tcp_telemetry_store = TcpTelemetryStore()


async def _fetch_tcp_telemetry() -> TcpInputTelemetry | None:
    """ZeroMQ経由でTCPテレメトリを取得."""
    with get_daemon_client(timeout_ms=1500) as client:
        response = client.tcp_input_status()
    if response.success:
        data = response.data
        if isinstance(data, TcpInputStatusResponse):
            return data.telemetry
        if isinstance(data, TcpInputTelemetry):
            return data
        if isinstance(data, dict):
            telemetry_payload = data.get("telemetry", data)
            if isinstance(telemetry_payload, dict):
                return parse_tcp_telemetry(telemetry_payload)
        return None
    if response.error:
        # 例外はポーラー側で握りつぶし、storeにエラーを記録させる
        raise response.error
    return None


_tcp_telemetry_poller = TcpTelemetryPoller(
    fetcher=_fetch_tcp_telemetry, store=_tcp_telemetry_store
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    await _tcp_telemetry_poller.start()
    yield
    # Shutdown
    await _tcp_telemetry_poller.stop()


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
No authentication required (local network only).
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
)

# Register exception handlers for unified error responses
register_exception_handlers(app)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Include routers
app.include_router(status_router)
app.include_router(daemon_router)
app.include_router(eq_router)
app.include_router(opra_router)
app.include_router(dac_router)
app.include_router(crossfeed_router)
app.include_router(partitioned_router)
app.include_router(output_mode_router)
app.include_router(tcp_input_router)
app.include_router(rtp_router)
app.include_router(rtp_input_router)


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


@app.get("/tcp-input", response_class=HTMLResponse)
async def tcp_input_page(request: Request, lang: str | None = None):
    """Serve the TCP Input page."""
    resolved_lang = _resolve_lang(request, lang)
    return _render_with_lang(
        content=render_tcp_input(lang=resolved_lang), lang=resolved_lang
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
