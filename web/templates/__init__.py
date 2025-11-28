"""HTML templates for the web UI."""

from .admin import get_admin_html
from .rtp import get_rtp_sessions_html
from .user import get_embedded_html

__all__ = ["get_embedded_html", "get_admin_html", "get_rtp_sessions_html"]
