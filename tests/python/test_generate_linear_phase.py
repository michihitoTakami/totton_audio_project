"""Unit tests for scripts/generate_linear_phase.py"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure scripts directory is importable
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_filter import FilterConfig  # noqa: E402
import generate_linear_phase as linear_phase_module  # noqa: E402
from generate_linear_phase import LinearPhaseDesigner, generate_single_filter  # noqa: E402


class TestLinearPhaseDesigner:
    def test_design_is_symmetric(self):
        config = FilterConfig(
            n_taps=101,
            input_rate=44100,
            upsample_ratio=16,
            passband_end=20000,
            stopband_start=22050,
            kaiser_beta=14,
            phase_suffix="linear_phase",
        )
        designer = LinearPhaseDesigner(config)
        h = designer.design()

        assert len(h) == config.n_taps
        assert np.allclose(h, h[::-1], atol=1e-10)
        assert np.sum(h) > 0


class TestLinearPhaseGenerator:
    def test_generate_single_filter_returns_expected_name(self, monkeypatch):
        args = argparse.Namespace(
            taps=1024,
            input_rate=44100,
            upsample_ratio=16,
            passband_end=20000,
            stopband_start=22050,
            stopband_attenuation=40,
            kaiser_beta=14,
            output_prefix=None,
        )

        def fake_export(self, h, metadata, skip_header=False):
            return self.config.base_name

        monkeypatch.setattr(linear_phase_module.FilterExporter, "export", fake_export)
        monkeypatch.setattr(
            linear_phase_module.FilterPlotter,
            "plot",
            lambda self, *args, **kwargs: None,
        )

        base_name, actual_taps = generate_single_filter(args)

        config = FilterConfig(
            n_taps=args.taps,
            input_rate=args.input_rate,
            upsample_ratio=args.upsample_ratio,
            passband_end=args.passband_end,
            stopband_start=args.stopband_start,
            kaiser_beta=args.kaiser_beta,
            phase_suffix="linear_phase",
        )

        assert base_name == config.base_name
        assert actual_taps in {args.taps, args.taps + 1}
