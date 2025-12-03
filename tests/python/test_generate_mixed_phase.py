"""
Unit tests for scripts/generate_mixed_phase.py
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_filter import FilterConfig, FilterDesigner  # noqa: E402
from generate_mixed_phase import (  # noqa: E402
    MixedPhaseCombiner,
    MixedPhaseFilterGenerator,
    MixedPhaseSettings,
    MixedPhaseSolverSettings,
)


class TestMixedPhaseSettings:
    def test_auto_delay_ms(self):
        """Auto delay should equal one period of crossover frequency."""
        settings = MixedPhaseSettings(crossover_hz=90.0, transition_hz=20.0, delay_ms=None)
        assert pytest.approx(settings.delay_ms, rel=1e-6) == pytest.approx(1000.0 / 90.0)


class TestMixedPhaseCombiner:
    def test_group_delay_alignment(self):
        """Synthesis should keep group delay error within a small bound."""
        config = FilterConfig(
            n_taps=4096,
            input_rate=44100,
            upsample_ratio=4,
            passband_end=20000,
            stopband_start=22050,
            kaiser_beta=10,
        )
        settings = MixedPhaseSettings(crossover_hz=100.0, transition_hz=30.0)
        solver = MixedPhaseSolverSettings(
            iterations=20, phase_blend=0.3, fft_oversample=2, gd_smooth_hz=100.0
        )
        designer = FilterDesigner(config)
        h_linear = designer.design_linear_phase()
        h_min = designer.convert_to_minimum_phase(h_linear)

        combiner = MixedPhaseCombiner(config, settings, solver)
        h_mixed, diagnostics = combiner.synthesize(h_linear, h_min)

        assert h_mixed.shape[0] == config.n_taps
        assert diagnostics["target_delay_samples"] > 0
        # Allow generous tolerance (1% of sampling rate) due to coarse FFT grid
        assert diagnostics["max_group_delay_error_samples"] < config.output_rate * 0.01


class TestMixedPhaseFilterGenerator:
    def test_metadata_contains_mixed_phase_block(self, monkeypatch):
        """Generator metadata should include mixed-phase diagnostics."""
        config = FilterConfig(
            n_taps=2048,
            input_rate=44100,
            upsample_ratio=2,
            output_prefix="test_mixed_phase",
        )
        settings = MixedPhaseSettings()
        solver = MixedPhaseSolverSettings(iterations=5, fft_oversample=1, phase_blend=0.2)
        generator = MixedPhaseFilterGenerator(config, settings, solver)

        # Avoid writing files / plots in tests
        monkeypatch.setattr(generator.plotter, "plot", lambda *args, **kwargs: None)
        captured: dict[str, Any] = {}

        def fake_export(_h: np.ndarray, metadata: dict[str, Any], _skip_header: bool) -> str:
            captured["metadata"] = metadata
            return metadata["output_basename"]

        monkeypatch.setattr(generator.exporter, "export", fake_export)

        base_name, taps = generator.generate(filter_name="unit-test")
        assert base_name == "test_mixed_phase"
        assert taps == config.n_taps
        assert "mixed_phase" in captured["metadata"]["validation_results"]

    def test_default_basename_uses_hybrid_suffix(self, monkeypatch):
        """When no prefix is supplied, generator should use *_hybrid_phase."""
        config = FilterConfig(n_taps=1024, input_rate=44100, upsample_ratio=2)
        settings = MixedPhaseSettings()
        solver = MixedPhaseSolverSettings(iterations=5, fft_oversample=1)
        generator = MixedPhaseFilterGenerator(config, settings, solver)

        monkeypatch.setattr(generator.plotter, "plot", lambda *args, **kwargs: None)
        monkeypatch.setattr(generator.exporter, "export", lambda *_a, **_k: generator.config.base_name)

        base_name, taps = generator.generate(filter_name="unit-test-hybrid")
        assert base_name.endswith("_hybrid_phase")
        assert taps == config.n_taps

