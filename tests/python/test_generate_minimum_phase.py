"""Unit tests for scripts/filters/generate_minimum_phase.py"""

import argparse

import numpy as np

from scripts.filters.generate_filter import FilterConfig
import scripts.filters.generate_minimum_phase as min_phase_module
from scripts.filters.generate_minimum_phase import (
    MinimumPhaseDesigner,
    generate_single_filter,
)


class TestMinimumPhaseDesigner:
    def test_design_returns_minimum_phase(self):
        config = FilterConfig(
            n_taps=101,
            input_rate=44100,
            upsample_ratio=16,
            passband_end=20000,
            stopband_start=22050,
            kaiser_beta=14,
        )
        designer = MinimumPhaseDesigner(config)
        h_min, h_linear = designer.design()

        assert len(h_min) == config.n_taps
        assert len(h_linear) >= config.n_taps
        peak_idx = int(np.argmax(np.abs(h_min)))
        assert peak_idx < len(h_min) * 0.2
        energy_first = np.sum(h_min[: len(h_min) // 2] ** 2)
        energy_second = np.sum(h_min[len(h_min) // 2 :] ** 2)
        assert energy_first > energy_second


class TestMinimumPhaseGenerator:
    def test_generate_single_filter_uses_config(self, monkeypatch):
        args = argparse.Namespace(
            taps=1024,
            input_rate=44100,
            upsample_ratio=16,
            passband_end=20000,
            stopband_start=22050,
            stopband_attenuation=40,
            kaiser_beta=14,
            minimum_phase_method="homomorphic",
            output_prefix=None,
        )

        def fake_export(self, h, metadata, skip_header=False):
            return self.config.base_name

        monkeypatch.setattr(min_phase_module.FilterExporter, "export", fake_export)
        monkeypatch.setattr(
            min_phase_module.FilterPlotter, "plot", lambda self, *args, **kwargs: None
        )

        base_name, actual_taps = generate_single_filter(args)

        config = FilterConfig(
            n_taps=args.taps,
            input_rate=args.input_rate,
            upsample_ratio=args.upsample_ratio,
            passband_end=args.passband_end,
            stopband_start=args.stopband_start,
            kaiser_beta=args.kaiser_beta,
        )

        assert base_name == config.base_name
        assert actual_taps == args.taps
