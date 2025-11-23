"""
GPU Upsampler Web Control API
FastAPI-based control interface for the GPU audio upsampler daemon.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .routers import daemon_router, eq_router, opra_router, status_router
from .templates import get_admin_html

app = FastAPI(
    title="GPU Upsampler Control",
    description="Web API for GPU Audio Upsampler daemon control",
    version="1.0.0",
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(status_router)
app.include_router(daemon_router)
app.include_router(eq_router)
app.include_router(opra_router)


# Legacy restart endpoint (forwards to daemon restart)
@app.post("/restart")
async def restart():
    """Legacy restart endpoint - forwards to daemon restart."""
    from .routers.daemon import daemon_restart

    return await daemon_restart()


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve the admin dashboard"""
    return get_admin_html()


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=11881)
