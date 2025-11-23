"""Pydantic models for the GPU Upsampler Web API."""

from typing import Any, Optional

from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings model."""

    alsa_device: str = "default"
    upsample_ratio: int = 8
    eq_profile: Optional[str] = None
    input_rate: int = 44100
    output_rate: int = 352800


class SettingsUpdate(BaseModel):
    """Settings update request model."""

    alsa_device: Optional[str] = None
    upsample_ratio: Optional[int] = None
    eq_profile: Optional[str] = None
    input_rate: Optional[int] = None
    output_rate: Optional[int] = None


class Status(BaseModel):
    """System status response model."""

    settings: Settings
    pipewire_connected: bool = False
    alsa_connected: bool = False
    clip_count: int = 0
    total_samples: int = 0
    clip_rate: float = 0.0
    daemon_running: bool = False
    eq_active: bool = False
    input_rate: int = 0
    output_rate: int = 0


class RewireRequest(BaseModel):
    """Request model for rewiring PipeWire connections."""

    source_node: str
    target_node: str


class EqProfile(BaseModel):
    """EQ profile model."""

    name: str
    path: str
    filters: list[dict[str, Any]] = []


class ApiResponse(BaseModel):
    """Standard API response model."""

    success: bool
    message: str
    data: Optional[dict[str, Any]] = None
    restart_required: bool = False
