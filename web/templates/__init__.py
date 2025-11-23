"""HTML templates for the web UI."""

from .admin import get_admin_html
from .user import get_embedded_html

__all__ = ["get_embedded_html", "get_admin_html"]
