"""API routers for the GPU Upsampler Web API."""

from .crossfeed import router as crossfeed_router
from .dac import router as dac_router
from .daemon import router as daemon_router
from .eq import router as eq_router
from .input_mode import router as input_mode_router
from .opra import router as opra_router
from .partitioned import router as partitioned_router
from .status import router as status_router
from .rtp import router as rtp_router
from .output_mode import router as output_mode_router

__all__ = [
    "crossfeed_router",
    "dac_router",
    "daemon_router",
    "eq_router",
    "opra_router",
    "input_mode_router",
    "output_mode_router",
    "partitioned_router",
    "status_router",
    "rtp_router",
]
