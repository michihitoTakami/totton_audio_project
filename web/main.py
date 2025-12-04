"""
GPU Upsampler Web Control API
FastAPI-based control interface for the GPU audio upsampler daemon.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .exceptions import register_exception_handlers
from .models import ApiResponse
from .routers import (
    crossfeed_router,
    dac_router,
    daemon_router,
    eq_router,
    input_mode_router,
    opra_router,
    output_mode_router,
    partitioned_router,
    rtp_router,
    status_router,
)
from .services import telemetry_poller
from .templates import get_admin_html, get_rtp_sessions_html
from .templates.pages import (
    render_dashboard,
    render_eq_settings,
    render_crossfeed,
    render_rtp_management,
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
        "name": "rtp",
        "description": "RTP session lifecycle management and telemetry",
    },
    {
        "name": "input-mode",
        "description": "Switch between PipeWire and RTP input sources",
    },
    {
        "name": "output",
        "description": "Output mode selection and device preferences",
    },
    {
        "name": "legacy",
        "description": "Deprecated endpoints - use alternatives instead",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage RTP telemetry poller lifecycle."""
    # Startup: start telemetry polling
    await telemetry_poller.start()
    yield
    # Shutdown: stop telemetry polling
    await telemetry_poller.stop()


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
app.include_router(rtp_router)
app.include_router(partitioned_router)
app.include_router(input_mode_router)
app.include_router(output_mode_router)


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
async def dashboard_page(lang: str = "en"):
    """Serve the Dashboard page."""
    return render_dashboard(lang=lang)


@app.get("/eq", response_class=HTMLResponse)
async def eq_page(lang: str = "en"):
    """Serve the EQ Settings page."""
    return render_eq_settings(lang=lang)


@app.get("/crossfeed", response_class=HTMLResponse)
async def crossfeed_page(lang: str = "en"):
    """Serve the Crossfeed page."""
    return render_crossfeed(lang=lang)


@app.get("/rtp", response_class=HTMLResponse)
async def rtp_management_page(lang: str = "en"):
    """Serve the RTP Management page."""
    return render_rtp_management(lang=lang)


@app.get("/system", response_class=HTMLResponse)
async def system_page(lang: str = "en"):
    """Serve the System page."""
    return render_system(lang=lang)


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve the admin dashboard (legacy)."""
    return get_admin_html()


@app.get("/rtp-sessions", response_class=HTMLResponse)
async def rtp_sessions_page():
    """Serve the RTP session management page (legacy)."""
    return get_rtp_sessions_html()


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=11881)
