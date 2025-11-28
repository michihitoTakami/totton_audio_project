"""
GPU Upsampler Web Control API
FastAPI-based control interface for the GPU audio upsampler daemon.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .exceptions import register_exception_handlers
from .models import ApiResponse
from .routers import (
    crossfeed_router,
    dac_router,
    daemon_router,
    eq_router,
    opra_router,
    partitioned_router,
    rtp_router,
    status_router,
)
from .templates import get_admin_html, get_rtp_sessions_html

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
        "name": "legacy",
        "description": "Deprecated endpoints - use alternatives instead",
    },
]

app = FastAPI(
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

# Include routers
app.include_router(status_router)
app.include_router(daemon_router)
app.include_router(eq_router)
app.include_router(opra_router)
app.include_router(dac_router)
app.include_router(crossfeed_router)
app.include_router(rtp_router)
app.include_router(partitioned_router)


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


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve the admin dashboard."""
    return get_admin_html()


@app.get("/rtp", response_class=HTMLResponse)
async def rtp_page():
    """Serve the RTP session management page."""
    return get_rtp_sessions_html()


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=11881)
