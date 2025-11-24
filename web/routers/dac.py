"""DAC capability API endpoints."""

from fastapi import APIRouter, HTTPException, Query

from ..models import ApiResponse, DacCapabilityResponse
from ..services.alsa import get_alsa_devices
from ..services.dac import (
    get_max_upsample_ratio,
    get_supported_output_rates,
    scan_dac_capability,
)

router = APIRouter(prefix="/dac", tags=["dac"])


@router.get("/capabilities", response_model=DacCapabilityResponse)
async def get_dac_capabilities(
    device: str = Query(default="default", description="ALSA device name (e.g., hw:0)"),
):
    """
    Get DAC capabilities for the specified device.

    Returns supported sample rates, channel count, and other hardware info.
    """
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


@router.get("/devices")
async def list_dac_devices():
    """
    List available DAC devices with their capabilities.

    Returns device list with basic info and supported rates.
    """
    devices = get_alsa_devices()
    result = []

    for dev in devices:
        device_info = {
            "id": dev["id"],
            "name": dev["name"],
            "description": dev["description"],
            "capabilities": None,
        }

        # Skip capability scan for 'default' as it may not support hw params query
        if dev["id"] != "default":
            cap = scan_dac_capability(dev["id"])
            if cap.is_valid:
                device_info["capabilities"] = {
                    "min_sample_rate": cap.min_sample_rate,
                    "max_sample_rate": cap.max_sample_rate,
                    "supported_rates": cap.supported_rates,
                    "max_channels": cap.max_channels,
                }

        result.append(device_info)

    return {"devices": result}


@router.get("/supported-rates")
async def get_supported_rates(
    device: str = Query(default="default", description="ALSA device name"),
    family: str = Query(default="44k", description="Rate family: '44k' or '48k'"),
):
    """
    Get supported output rates for a specific rate family.

    Useful for filtering UI options based on DAC capabilities.
    """
    if family not in ("44k", "48k"):
        raise HTTPException(status_code=400, detail="family must be '44k' or '48k'")

    rates = get_supported_output_rates(device, family)
    return {
        "device": device,
        "family": family,
        "supported_rates": rates,
    }


@router.get("/max-ratio")
async def get_max_ratio(
    device: str = Query(default="default", description="ALSA device name"),
    input_rate: int = Query(default=44100, description="Input sample rate"),
):
    """
    Get maximum upsampling ratio for a given input rate.

    Returns the highest ratio (1, 2, 4, 8, 16) that the DAC supports.
    """
    if input_rate <= 0:
        raise HTTPException(status_code=400, detail="input_rate must be positive")

    ratio = get_max_upsample_ratio(device, input_rate)
    max_output_rate = input_rate * ratio if ratio > 0 else 0

    return {
        "device": device,
        "input_rate": input_rate,
        "max_ratio": ratio,
        "max_output_rate": max_output_rate,
    }


@router.get("/validate-config")
async def validate_dac_config(
    device: str = Query(..., description="ALSA device name"),
    input_rate: int = Query(..., description="Input sample rate"),
    output_rate: int = Query(..., description="Output sample rate"),
):
    """
    Validate if a DAC configuration is supported.

    Checks if the DAC supports the specified output rate.
    """
    cap = scan_dac_capability(device)

    if not cap.is_valid:
        return ApiResponse(
            success=False,
            message=f"Cannot scan device: {cap.error_message}",
            data={"device": device, "is_supported": False},
        )

    is_supported = output_rate in cap.supported_rates

    # Also check if the ratio makes sense
    ratio = output_rate // input_rate if input_rate > 0 else 0
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
