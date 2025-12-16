"""Unit tests for scripts.analysis.partition_analysis helpers."""

from pathlib import Path
import json

import numpy as np

from scripts.analysis.partition_analysis import (
    PartitionConfig,
    build_partition_plan,
    estimate_settling_samples,
    load_partition_config,
    partition_energy_summary,
)


def test_build_partition_plan_matches_cpp_defaults():
    config = PartitionConfig(
        enabled=True,
        fast_partition_taps=32_768,
        min_partition_taps=32_768,
        max_partitions=4,
        tail_fft_multiple=2,
    )
    plan = build_partition_plan(640_000, config)
    assert plan.enabled
    taps = [part.taps for part in plan.partitions]
    assert taps == [32_768, 65_536, 131_072, 410_624]
    assert plan.partitions[0].fft_size == 65_536
    assert plan.partitions[0].valid_output == 32_769


def test_partition_energy_summary_tracks_segments():
    coeffs = np.ones(16, dtype=np.float32)
    config = PartitionConfig(
        enabled=True, fast_partition_taps=4, min_partition_taps=4, max_partitions=3
    )
    plan = build_partition_plan(len(coeffs), config)
    summary = partition_energy_summary(coeffs, plan)
    assert len(summary) == len(plan.partitions)
    total_pct = sum(entry["energy_pct"] for entry in summary)
    assert abs(total_pct - 100.0) < 1e-6


def test_estimate_settling_samples_returns_valid_windows():
    config = PartitionConfig(
        enabled=True, fast_partition_taps=8, min_partition_taps=4, max_partitions=2
    )
    plan = build_partition_plan(32, config)
    fast_window, settling_window = estimate_settling_samples(plan)
    assert fast_window > 0
    assert settling_window >= fast_window


def test_load_partition_config_reads_json(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg = {
        "partitionedConvolution": {
            "enabled": True,
            "fastPartitionTaps": 4096,
            "minPartitionTaps": 2048,
            "maxPartitions": 5,
            "tailFftMultiple": 6,
        }
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = load_partition_config(cfg_path)
    assert loaded.enabled is True
    assert loaded.fast_partition_taps == 4096
    assert loaded.min_partition_taps == 2048
    assert loaded.max_partitions == 5
    assert loaded.tail_fft_multiple == 6
