#!/usr/bin/env python3
"""
Utilities for reasoning about partitioned convolution plans.

This module mirrors the logic in `ConvolutionEngine::buildPartitionPlan` so that
Python tooling (scripts + tests) can analyze the fast/tail split when low-latency
mode is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import List

import numpy as np

__all__ = [
    "PartitionConfig",
    "PartitionDescriptor",
    "PartitionPlan",
    "build_partition_plan",
    "partition_energy_summary",
    "estimate_settling_samples",
    "load_partition_config",
]


def _next_pow2(value: int) -> int:
    """Return the next power-of-two >= value."""
    if value <= 0:
        return 1
    return 1 << ((value - 1).bit_length())


@dataclass
class PartitionConfig:
    """Runtime tunables for partitioned convolution."""

    enabled: bool = True
    fast_partition_taps: int = 32_768
    min_partition_taps: int = 32_768
    max_partitions: int = 4
    tail_fft_multiple: int = 2

    def merge_overrides(self, **overrides: int | bool) -> "PartitionConfig":
        """Return a copy with fields overridden."""
        data = asdict(self)
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in data:
                raise KeyError(f"Unknown partition config key: {key}")
            data[key] = value
        return PartitionConfig(**data)


@dataclass
class PartitionDescriptor:
    """Single partition characteristics."""

    taps: int
    fft_size: int
    valid_output: int
    realtime: bool


@dataclass
class PartitionPlan:
    """Full plan for a filter."""

    total_taps: int = 0
    realtime_taps: int = 0
    partitions: List[PartitionDescriptor] = field(default_factory=list)
    enabled: bool = False

    def describe(self) -> str:
        if not self.enabled or not self.partitions:
            return "disabled"
        parts = []
        for idx, part in enumerate(self.partitions):
            role = "fast" if part.realtime else "tail"
            parts.append(
                f"{role}#{idx}: taps={part.taps}, fft={part.fft_size}, valid={part.valid_output}"
            )
        return " | ".join(parts)


def _make_descriptor(
    taps: int, realtime: bool, tail_fft_multiple: int
) -> PartitionDescriptor:
    fft_multiple = 2 if realtime else max(2, tail_fft_multiple)
    fft_target = max(taps * fft_multiple, taps + 1)
    fft_size = _next_pow2(fft_target)
    valid_output = max(1, fft_size - taps + 1)
    return PartitionDescriptor(
        taps=taps, fft_size=fft_size, valid_output=valid_output, realtime=realtime
    )


def build_partition_plan(total_taps: int, config: PartitionConfig) -> PartitionPlan:
    """Python port of ConvolutionEngine::buildPartitionPlan."""
    plan = PartitionPlan(total_taps=total_taps)
    if not config.enabled or total_taps <= 0:
        return plan

    plan.enabled = True
    min_taps = max(1024, int(config.min_partition_taps))
    max_partitions = max(1, int(config.max_partitions))

    remaining = total_taps
    fast_lower = min(min_taps, total_taps)
    fast_upper = max(fast_lower, total_taps)
    fast_taps = int(np.clip(config.fast_partition_taps, fast_lower, fast_upper))

    plan.partitions.append(_make_descriptor(fast_taps, True, config.tail_fft_multiple))
    plan.realtime_taps = plan.partitions[0].taps
    remaining -= fast_taps
    if remaining <= 0:
        return plan

    previous_taps = fast_taps
    partitions_used = 1
    while remaining > 0 and partitions_used < max_partitions:
        suggested = previous_taps * 2
        taps = min(max(suggested, min_taps), remaining)

        # Last slot takes all remaining taps
        if partitions_used == max_partitions - 1:
            taps = remaining

        plan.partitions.append(_make_descriptor(taps, False, config.tail_fft_multiple))
        remaining -= taps
        previous_taps = taps
        partitions_used += 1

    if remaining > 0:
        plan.partitions.append(
            _make_descriptor(remaining, False, config.tail_fft_multiple)
        )

    plan.realtime_taps = plan.partitions[0].taps
    return plan


def _partition_segments(coeffs: np.ndarray, plan: PartitionPlan) -> List[np.ndarray]:
    cursor = 0
    segments: List[np.ndarray] = []
    for desc in plan.partitions:
        next_cursor = min(cursor + desc.taps, coeffs.size)
        segment = coeffs[cursor:next_cursor]
        segments.append(segment)
        cursor = next_cursor
    if cursor < coeffs.size:
        # Append leftover taps to the last partition to keep energy accounting accurate.
        if segments:
            segments[-1] = np.concatenate([segments[-1], coeffs[cursor:]])
        else:
            segments.append(coeffs[cursor:])
    return segments


def partition_energy_summary(coeffs: np.ndarray, plan: PartitionPlan) -> list[dict]:
    """Return energy per partition expressed as a percentage of total energy."""
    if not plan.enabled or not plan.partitions:
        return []

    coeffs64 = coeffs.astype(np.float64, copy=False)
    total_energy = float(np.sum(coeffs64**2))
    if total_energy == 0:
        total_energy = 1.0

    segments = _partition_segments(coeffs64, plan)
    summary = []
    for idx, (desc, segment) in enumerate(zip(plan.partitions, segments)):
        energy = float(np.sum(segment**2))
        energy_pct = (energy / total_energy) * 100.0
        summary.append(
            {
                "index": idx,
                "role": "fast" if desc.realtime else "tail",
                "taps": int(desc.taps),
                "fft_size": int(desc.fft_size),
                "valid_output": int(desc.valid_output),
                "energy": energy,
                "energy_pct": energy_pct,
            }
        )
    return summary


def estimate_settling_samples(plan: PartitionPlan) -> tuple[int, int]:
    """
    Estimate how many output samples are produced before:
      * only the fast partition has contributed (fast window)
      * all partitions produce at least one block (settling window)
    """
    if not plan.enabled or not plan.partitions:
        return (0, 0)
    fast_window = plan.partitions[0].valid_output
    total_window = sum(part.valid_output for part in plan.partitions)
    return fast_window, total_window


def load_partition_config(
    path: str | Path | None, *, base: PartitionConfig | None = None
) -> PartitionConfig:
    """Load partition config from config.json (GPU daemon format)."""
    if base is None:
        base = PartitionConfig()
    if path is None:
        return base

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    section = raw.get("partitionedConvolution", {})
    overrides = {
        "enabled": section.get("enabled", base.enabled),
        "fast_partition_taps": section.get(
            "fastPartitionTaps", base.fast_partition_taps
        ),
        "min_partition_taps": section.get("minPartitionTaps", base.min_partition_taps),
        "max_partitions": section.get("maxPartitions", base.max_partitions),
        "tail_fft_multiple": section.get("tailFftMultiple", base.tail_fft_multiple),
    }
    return PartitionConfig(**overrides)
