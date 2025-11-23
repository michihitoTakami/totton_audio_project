"""API routers for the GPU Upsampler Web API."""

from .daemon import router as daemon_router
from .eq import router as eq_router
from .opra import router as opra_router
from .status import router as status_router

__all__ = [
    "daemon_router",
    "eq_router",
    "opra_router",
    "status_router",
]
