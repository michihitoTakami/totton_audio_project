"""DAC capability API endpoints."""

from fastapi import APIRouter, HTTPException, Query

from ..models import (
    ApiResponse,
    DacCapabilityInfo,
    DacCapabilityResponse,
    DacDeviceInfo,
    DacDevicesResponse,
    DacMaxRatioResponse,
    DacDaemonState,
    DacSelectRequest,
    DacSupportedRatesResponse,
)
from ..services.alsa import get_alsa_devices
from ..services.dac import (
    get_max_upsample_ratio,
    get_supported_output_rates,
    is_safe_device_name,
    scan_dac_capability,
)
from ..services.daemon_client import get_daemon_client
from ..services.config import load_config, save_config

router = APIRouter(prefix="/dac", tags=["dac"])


@router.get("/state", response_model=DacDaemonState)
async def get_dac_state() -> DacDaemonState:
    """
    Fetch current DAC runtime state from the daemon.
    """
    with get_daemon_client() as client:
        result = client.dac_status()
        if not result.success:
            raise result.error
        data = result.data
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="Invalid daemon response")
        return DacDaemonState(**data)


@router.get("/capabilities", response_model=DacCapabilityResponse)
async def get_dac_capabilities(
    device: str = Query(default="default", description="ALSA device name (e.g., hw:0)"),
) -> DacCapabilityResponse:
    """
    Get DAC capabilities for the specified device.

    Returns supported sample rates, channel count, and other hardware info.
    Results are cached for 60 seconds.
    """
    if not is_safe_device_name(device):
        raise HTTPException(status_code=400, detail="Invalid device name format")

    cap = scan_dac_capability(device)
    return DacCapabilityResponse(
        device_name=cap.device_name,
        min_sample_rate=cap.min_sample_rate,
        max_sample_rate=cap.max_sample_rate,
        supported_rates=cap.supported_rates,
        max_channels=cap.max_channels,
        is_valid=cap.is_valid,
        error_message=cap.error_message,
    )


@router.get("/devices", response_model=DacDevicesResponse)
async def list_dac_devices() -> DacDevicesResponse:
    """
    List available DAC devices with their capabilities.

    Returns device list with basic info and supported rates.
    Capability scan is skipped for 'default' device.
    """
    devices = get_alsa_devices()
    result: list[DacDeviceInfo] = []

    for dev in devices:
        capabilities = None

        # Skip capability scan for 'default' as it may not support hw params query
        if dev["id"] != "default" and is_safe_device_name(dev["id"]):
            cap = scan_dac_capability(dev["id"])
            if cap.is_valid:
                capabilities = DacCapabilityInfo(
                    min_sample_rate=cap.min_sample_rate,
                    max_sample_rate=cap.max_sample_rate,
                    supported_rates=cap.supported_rates,
                    max_channels=cap.max_channels,
                )

        result.append(
            DacDeviceInfo(
                id=dev["id"],
                name=dev["name"],
                description=dev["description"],
                capabilities=capabilities,
            )
        )

    return DacDevicesResponse(devices=result)


@router.get("/supported-rates", response_model=DacSupportedRatesResponse)
async def get_supported_rates(
    device: str = Query(default="default", description="ALSA device name"),
    family: str = Query(default="44k", description="Rate family: '44k' or '48k'"),
) -> DacSupportedRatesResponse:
    """
    Get supported output rates for a specific rate family.

    Useful for filtering UI options based on DAC capabilities.
    Only rates that belong to the specified family (44.1k multiples or 48k multiples)
    and are supported by the DAC are returned.
    """
    if not is_safe_device_name(device):
        raise HTTPException(status_code=400, detail="Invalid device name format")

    if family not in ("44k", "48k"):
        raise HTTPException(status_code=400, detail="family must be '44k' or '48k'")

    rates = get_supported_output_rates(device, family)
    return DacSupportedRatesResponse(
        device=device,
        family=family,
        supported_rates=rates,
    )


@router.get("/max-ratio", response_model=DacMaxRatioResponse)
async def get_max_ratio(
    device: str = Query(default="default", description="ALSA device name"),
    input_rate: int = Query(default=44100, description="Input sample rate"),
) -> DacMaxRatioResponse:
    """
    Get maximum upsampling ratio for a given input rate.

    Returns the highest ratio (1, 2, 4, 8, 16) that the DAC supports
    for the specified input sample rate.
    """
    if not is_safe_device_name(device):
        raise HTTPException(status_code=400, detail="Invalid device name format")

    if input_rate <= 0:
        raise HTTPException(status_code=400, detail="input_rate must be positive")

    ratio = get_max_upsample_ratio(device, input_rate)
    max_output_rate = input_rate * ratio if ratio > 0 else 0

    return DacMaxRatioResponse(
        device=device,
        input_rate=input_rate,
        max_ratio=ratio,
        max_output_rate=max_output_rate,
    )


@router.get("/validate-config", response_model=ApiResponse)
async def validate_dac_config(
    device: str = Query(..., description="ALSA device name"),
    input_rate: int = Query(..., description="Input sample rate"),
    output_rate: int = Query(..., description="Output sample rate"),
) -> ApiResponse:
    """
    Validate if a DAC configuration is supported.

    Checks if the DAC supports the specified output rate and if the
    upsampling ratio is valid (1, 2, 4, 8, or 16).
    """
    if not is_safe_device_name(device):
        raise HTTPException(status_code=400, detail="Invalid device name format")

    if input_rate <= 0:
        raise HTTPException(status_code=400, detail="input_rate must be positive")

    if output_rate <= 0:
        raise HTTPException(status_code=400, detail="output_rate must be positive")

    cap = scan_dac_capability(device)

    if not cap.is_valid:
        return ApiResponse(
            success=False,
            message=f"Cannot scan device: {cap.error_message}",
            data={"device": device, "is_supported": False},
        )

    is_supported = output_rate in cap.supported_rates

    # Also check if the ratio makes sense
    ratio = output_rate // input_rate
    valid_ratios = [1, 2, 4, 8, 16]
    ratio_valid = ratio in valid_ratios and output_rate == input_rate * ratio

    return ApiResponse(
        success=True,
        message="Configuration validated",
        data={
            "device": device,
            "input_rate": input_rate,
            "output_rate": output_rate,
            "is_supported": is_supported and ratio_valid,
            "rate_supported": is_supported,
            "ratio_valid": ratio_valid,
            "suggested_rates": cap.supported_rates,
        },
    )


@router.post("/select", response_model=ApiResponse)
async def select_runtime_dac(request: DacSelectRequest) -> ApiResponse:
    """
    Ask the daemon to switch to a specific ALSA device.

    Optionally persist the selection to config.json.
    """
    if not is_safe_device_name(request.device):
        raise HTTPException(status_code=400, detail="Invalid device name format")

    with get_daemon_client() as client:
        result = client.dac_select(request.device)
        if not result.success:
            raise result.error
        data = result.data if isinstance(result.data, dict) else None

    if request.persist:
        settings = load_config()
        settings.alsa_device = request.device
        if not save_config(settings):
            raise HTTPException(status_code=500, detail="Failed to persist config")

    return ApiResponse(success=True, message="DAC selection updated", data=data)


@router.post("/rescan", response_model=ApiResponse)
async def rescan_runtime_dac() -> ApiResponse:
    """
    Trigger a DAC rescan on the daemon.
    """
    with get_daemon_client() as client:
        result = client.dac_rescan()
        if not result.success:
            raise result.error
        data = result.data if isinstance(result.data, dict) else None
    return ApiResponse(success=True, message="Rescan requested", data=data)
