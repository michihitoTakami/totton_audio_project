"""Router for UI mockup pages."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, FileResponse

from web.templates.mockup import (
    get_eq_html,
    get_headphones_html,
    get_rtp_html,
    get_status_html,
    get_system_html,
)

router = APIRouter(prefix="/mockup", tags=["mockup"])


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Return dashboard mockup page (static HTML)"""
    dashboard_path = (
        Path(__file__).parent.parent / "static" / "mockup" / "dashboard.html"
    )
    return FileResponse(dashboard_path)


@router.get("/headphones", response_class=HTMLResponse)
async def get_headphones():
    """Return headphones selection mockup page"""
    return get_headphones_html()


@router.get("/eq", response_class=HTMLResponse)
async def get_eq():
    """Return EQ settings mockup page"""
    return get_eq_html()


@router.get("/system", response_class=HTMLResponse)
async def get_system():
    """Return system settings mockup page"""
    return get_system_html()


@router.get("/status", response_class=HTMLResponse)
async def get_status():
    """Return status monitoring mockup page"""
    return get_status_html()


@router.get("/rtp", response_class=HTMLResponse)
async def get_rtp():
    """Return RTP settings mockup page"""
    return get_rtp_html()
