"""API routers for the GPU Upsampler Web API."""

from .crossfeed import router as crossfeed_router
from .dac import router as dac_router
from .daemon import router as daemon_router
from .delimiter import router as delimiter_router
from .eq import router as eq_router
from .opra import router as opra_router
from .opra_sync import router as opra_sync_router
from .partitioned import router as partitioned_router
from .status import router as status_router
from .output_mode import router as output_mode_router
from .i2s import router as i2s_router
from .pi import router as pi_router

__all__ = [
    "crossfeed_router",
    "dac_router",
    "daemon_router",
    "delimiter_router",
    "eq_router",
    "i2s_router",
    "opra_router",
    "opra_sync_router",
    "pi_router",
    "output_mode_router",
    "partitioned_router",
    "status_router",
]
