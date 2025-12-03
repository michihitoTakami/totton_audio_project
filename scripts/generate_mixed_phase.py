#!/usr/bin/env python3
"""
GPU Audio Upsampler - Mixed Phase Filter Generator (Approach 2)

アプローチ2に基づき、振幅特性を固定したまま群遅延を最適化する
FIR混合位相フィルタを生成する。100 Hz以下は最小位相に準拠し、
それ以上の帯域は一定群遅延（約10 ms）へ滑らかに遷移させる。
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from types import SimpleNamespace

import numpy as np

from generate_filter import (
    FilterConfig,
    FilterDesigner,
    FilterExporter,
    FilterPlotter,
    FilterValidator,
    MinimumPhaseMethod,
    MULTI_RATE_CONFIGS,
    calculate_safe_gain,
    generate_multi_rate_header,
    normalize_coefficients,
    print_safe_gain_recommendation,
    validate_tap_count,
)


@dataclass
class MixedPhaseSettings:
    """設定値（クロスオーバー/遷移幅/整列遅延）"""

    crossover_hz: float = 300.0
    transition_hz: float = 400.0
    delay_ms: float | None = None

    def __post_init__(self) -> None:
        if self.crossover_hz <= 0:
            raise ValueError("crossover_hz must be > 0")
        if self.transition_hz <= 0:
            raise ValueError("transition_hz must be > 0")
        if self.delay_ms is None:
            self.delay_ms = 1000.0 / self.crossover_hz
        if self.delay_ms <= 0:
            raise ValueError("delay_ms must be > 0")

    @property
    def delay_seconds(self) -> float:
        return self.delay_ms / 1000.0

    def describe(self) -> dict[str, float]:
        return {
            "crossover_hz": float(self.crossover_hz),
            "transition_hz": float(self.transition_hz),
            "delay_ms": float(self.delay_ms),
        }


@dataclass
class MixedPhaseSolverSettings:
    """位相最適化用の調整パラメータ"""

    iterations: int = 200
    min_iterations: int = 50
    phase_blend: float = 0.2
    amplitude_relax: float = 0.0
    fft_oversample: int = 4
    gd_smooth_hz: float = 200.0
    tail_taps: int = 2048
    tolerance: float = 1e-4


class MixedPhaseCombiner:
    """線形位相＋最小位相フィルタから混合位相を合成する"""

    def __init__(
        self,
        config: FilterConfig,
        settings: MixedPhaseSettings,
        solver: MixedPhaseSolverSettings,
    ) -> None:
        self.config = config
        self.settings = settings
        self.solver = solver

    def synthesize(
        self, h_linear: np.ndarray, h_min: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        n_fft = self.config.n_taps * max(1, self.solver.fft_oversample)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.config.output_rate)
        omega = 2.0 * np.pi * freqs
        h_min_use = h_min[: self.config.n_taps]
        h_linear_use = h_linear[: self.config.n_taps]

        H_min = np.fft.rfft(h_min_use, n_fft)
        H_linear = np.fft.rfft(h_linear_use, n_fft)

        magnitude_target = np.maximum(np.abs(H_linear), 1e-12)
        magnitude_current = magnitude_target.copy()
        phase_min = np.unwrap(np.angle(H_min))

        gd_min_sec = -np.gradient(phase_min, omega, edge_order=2)
        gd_min_sec = np.clip(gd_min_sec, 0.0, None)

        gd_target_sec = self._build_target_group_delay(freqs, gd_min_sec)
        gd_target_sec = self._smooth_group_delay(
            freqs, gd_target_sec, self.solver.gd_smooth_hz
        )
        phase_target = self._integrate_phase(omega, gd_target_sec)
        phase_target += phase_min[0] - phase_target[0]

        phase_current = phase_target.copy()
        control_mask = freqs <= self.config.passband_end
        padded = np.zeros(n_fft, dtype=np.float64)
        prev_phase = phase_current.copy()

        max_iters = max(1, self.solver.iterations)
        for idx in range(max_iters):
            spectrum = magnitude_current * np.exp(1j * phase_current)
            h_time = np.fft.irfft(spectrum, n_fft).real
            h_time = h_time[: self.config.n_taps]
            padded[:] = 0.0
            padded[: self.config.n_taps] = h_time
            spectrum_time = np.fft.rfft(padded, n_fft)

            phase_proj = np.angle(spectrum_time)
            phase_current = phase_proj
            if np.any(control_mask):
                delta = self._wrap_phase(phase_proj[control_mask] - phase_target[control_mask])
                phase_current[control_mask] = (
                    phase_proj[control_mask] - self.solver.phase_blend * delta
                )

            if self.solver.amplitude_relax > 0.0:
                mag_measured = np.abs(spectrum_time)
                magnitude_current = (
                    (1 - self.solver.amplitude_relax) * magnitude_current
                    + self.solver.amplitude_relax * mag_measured
                )

            if idx >= max(self.solver.min_iterations, 1):
                phase_delta = np.max(
                    np.abs(self._wrap_phase(phase_current - prev_phase))
                )
                prev_phase = phase_current.copy()
                if phase_delta < self.solver.tolerance:
                    break
            else:
                prev_phase = phase_current.copy()

        H_target = magnitude_current * np.exp(1j * phase_current)
        h_time = np.fft.irfft(H_target, n_fft).real[: self.config.n_taps]
        h_time = self._apply_time_taper(h_time)

        diagnostics = self._analyze(h_time, omega, gd_target_sec, freqs)
        return h_time, diagnostics

    def _build_target_group_delay(
        self, freqs: np.ndarray, gd_min_sec: np.ndarray
    ) -> np.ndarray:
        target = gd_min_sec.copy()
        start = max(0.0, self.settings.crossover_hz - self.settings.transition_hz / 2.0)
        end = self.settings.crossover_hz + self.settings.transition_hz / 2.0

        delay_const = self.settings.delay_seconds
        low_mask = freqs <= start
        high_mask = freqs >= end
        transition_mask = (~low_mask) & (~high_mask)

        target[high_mask] = delay_const
        if np.any(transition_mask):
            phase = (freqs[transition_mask] - start) / max(end - start, 1e-9)
            smooth = 0.5 - 0.5 * np.cos(np.pi * np.clip(phase, 0.0, 1.0))
            target[transition_mask] = (
                (1.0 - smooth) * gd_min_sec[transition_mask] + smooth * delay_const
            )

        target[low_mask] = gd_min_sec[low_mask]
        return np.clip(target, 0.0, None)

    def _smooth_group_delay(
        self, freqs: np.ndarray, gd_sec: np.ndarray, bandwidth_hz: float
    ) -> np.ndarray:
        if bandwidth_hz <= 0 or len(freqs) < 2:
            return gd_sec
        step = freqs[1] - freqs[0]
        window = max(3, int(bandwidth_hz / max(step, 1e-9)))
        if window % 2 == 0:
            window += 1
        if window >= len(gd_sec):
            return np.full_like(gd_sec, gd_sec.mean())
        kernel = np.hanning(window)
        kernel /= np.sum(kernel)
        return np.convolve(gd_sec, kernel, mode="same")

    @staticmethod
    def _wrap_phase(delta: np.ndarray) -> np.ndarray:
        return (delta + np.pi) % (2 * np.pi) - np.pi

    def _apply_time_taper(self, h: np.ndarray) -> np.ndarray:
        tail = min(self.solver.tail_taps, h.size)
        if tail <= 0:
            return h
        fade = 0.5 * (1 + np.cos(np.linspace(0, np.pi, tail, dtype=h.dtype)))
        h = h.copy()
        h[-tail:] *= fade
        return h

    def _integrate_phase(self, omega: np.ndarray, gd_target_sec: np.ndarray) -> np.ndarray:
        phase = np.zeros_like(omega)
        delta_omega = np.diff(omega)
        avg_tau = 0.5 * (gd_target_sec[1:] + gd_target_sec[:-1])
        phase[1:] = -np.cumsum(avg_tau * delta_omega)
        return phase

    def _analyze(
        self, h_time: np.ndarray, omega: np.ndarray, gd_target_sec: np.ndarray, freqs: np.ndarray
    ) -> dict[str, Any]:
        n_fft = (len(omega) - 1) * 2
        h_padded = np.pad(h_time, (0, max(0, n_fft - len(h_time))))
        H_actual = np.fft.rfft(h_padded, n_fft)
        phase_actual = np.unwrap(np.angle(H_actual[: len(gd_target_sec)]))
        gd_actual_sec = -np.gradient(phase_actual, omega, edge_order=2)

        error_samples = (gd_actual_sec - gd_target_sec) * self.config.output_rate
        passband_mask = freqs <= self.config.passband_end
        if np.any(passband_mask):
            max_error = float(np.max(np.abs(error_samples[passband_mask])))
        else:
            max_error = float(np.max(np.abs(error_samples)))

        sample_points = [
            20.0,
            self.settings.crossover_hz,
            min(self.config.passband_end, 2000.0),
        ]
        analysis = []
        for freq in sample_points:
            idx = int(np.argmin(np.abs(freqs - freq)))
            analysis.append(
                {
                    "frequency_hz": float(freqs[idx]),
                    "target_delay_samples": float(
                        gd_target_sec[idx] * self.config.output_rate
                    ),
                    "actual_delay_samples": float(
                        gd_actual_sec[idx] * self.config.output_rate
                    ),
                }
            )

        peak_idx = int(np.argmax(np.abs(h_time)))

        return {
            "crossover_hz": self.settings.crossover_hz,
            "transition_hz": self.settings.transition_hz,
            "target_delay_ms": self.settings.delay_ms,
            "target_delay_samples": float(
                self.settings.delay_seconds * self.config.output_rate
            ),
            "max_group_delay_error_samples": max_error,
            "peak_sample_index": peak_idx,
            "analysis": analysis,
        }


class MixedPhaseFilterGenerator:
    """混合位相フィルタ生成のオーケストレーション"""

    def __init__(
        self,
        config: FilterConfig,
        settings: MixedPhaseSettings,
        solver: MixedPhaseSolverSettings,
    ) -> None:
        self.config = config
        self.settings = settings
        self.solver = solver
        if not self.config.output_prefix:
            hybrid_name = (
                f"filter_{self.config.family}_{self.config.upsample_ratio}x_"
                f"{self.config.taps_label}_hybrid_phase"
            )
            self.config.output_prefix = hybrid_name
        if not hasattr(self.config, "phase_type"):
            self.config.phase_type = SimpleNamespace(value="mixed")
        self.designer = FilterDesigner(config)
        self.validator = FilterValidator(config)
        self.exporter = FilterExporter(config)
        self.plotter = FilterPlotter(config)

    def generate(
        self,
        filter_name: str | None = None,
        skip_header: bool = False,
    ) -> tuple[str, int]:
        validate_tap_count(self.config.n_taps, self.config.upsample_ratio)

        h_linear = self.designer.design_linear_phase()
        h_min = self.designer.convert_to_minimum_phase(h_linear)
        combiner = MixedPhaseCombiner(self.config, self.settings, self.solver)
        h_mixed, diagnostics = combiner.synthesize(h_linear, h_min)

        h_final, normalization = normalize_coefficients(
            h_mixed,
            target_dc_gain=self.config.target_dc_gain,
            dc_gain_factor=self.config.dc_gain_factor,
        )

        validation = self.validator.validate(h_final)
        validation["mixed_phase"] = diagnostics
        validation["normalization"] = normalization

        self.plotter.plot(h_final, h_linear, filter_name)

        metadata = self._create_metadata(validation)
        base_name = self.exporter.export(h_final, metadata, skip_header)
        self._print_report(validation, normalization, base_name)
        return base_name, len(h_final)

    def _create_metadata(self, validation: dict[str, Any]) -> dict[str, Any]:
        meta = {
            "generation_mode": "mixed_phase",
            "mixed_phase_settings": self.settings.describe(),
            "generation_date": datetime.now().isoformat(),
            "sample_rate_input": self.config.input_rate,
            "sample_rate_output": self.config.output_rate,
            "upsample_ratio": self.config.upsample_ratio,
            "passband_end_hz": self.config.passband_end,
            "stopband_start_hz": self.config.stopband_start,
            "target_stopband_attenuation_db": self.config.stopband_attenuation_db,
            "kaiser_beta": self.config.kaiser_beta,
            "minimum_phase_method": self.config.minimum_phase_method.value,
            "target_dc_gain": self.config.target_dc_gain,
            "output_basename": self.config.base_name,
            "validation_results": validation,
        }
        return meta

    def _print_report(
        self,
        validation: dict[str, Any],
        normalization: dict[str, Any],
        base_name: str,
    ) -> None:
        print("\n" + "=" * 70)
        print(f"完了 - {validation['actual_taps']:,}タップ混合位相フィルタ")
        print("=" * 70)
        print(
            f"阻止帯域減衰: {validation['stopband_attenuation_db']:.1f} dB "
            f"(目標 {self.config.stopband_attenuation_db} dB)"
        )
        print(
            "係数正規化: "
            f"目標DC={normalization['target_dc_gain']:.6f}, "
            f"結果DC={normalization['normalized_dc_gain']:.6f}"
        )
        mixed_info = validation.get("mixed_phase", {})
        if mixed_info:
            print(
                f"混合位相: crossover={mixed_info.get('crossover_hz', 0):.1f} Hz, "
                f"delay={mixed_info.get('target_delay_ms', 0):.2f} ms"
            )
            print(
                "  最大群遅延誤差: "
                f"{mixed_info.get('max_group_delay_error_samples', 0):.2f} samples"
            )
        print(f"係数ファイル: data/coefficients/{base_name}.bin")


def generate_single_filter(
    args: argparse.Namespace, filter_name: str | None = None, skip_header: bool = False
) -> tuple[str, int]:
    stopband_start = args.stopband_start if args.stopband_start else args.input_rate // 2
    config = FilterConfig(
        n_taps=args.taps,
        input_rate=args.input_rate,
        upsample_ratio=args.upsample_ratio,
        passband_end=args.passband_end,
        stopband_start=stopband_start,
        stopband_attenuation_db=args.stopband_attenuation,
        kaiser_beta=args.kaiser_beta,
        minimum_phase_method=MinimumPhaseMethod(args.minimum_phase_method),
        output_prefix=args.output_prefix,
    )
    settings = MixedPhaseSettings(
        crossover_hz=args.crossover_hz,
        transition_hz=args.transition_hz,
        delay_ms=args.delay_ms,
    )
    solver = MixedPhaseSolverSettings(
        iterations=args.iterations,
        min_iterations=args.min_iterations,
        phase_blend=args.phase_blend,
        amplitude_relax=args.amplitude_relax,
        fft_oversample=args.fft_oversample,
        gd_smooth_hz=args.gd_smooth_hz,
        tail_taps=args.tail_taps,
        tolerance=args.tolerance,
    )
    generator = MixedPhaseFilterGenerator(config, settings, solver)
    return generator.generate(filter_name, skip_header)


def _mixed_phase_worker(
    worker_args: tuple[str, dict[str, Any], dict[str, Any]]
) -> tuple[str, str, int, dict[str, Any], str | None]:
    name, cfg, args_dict = worker_args
    try:
        config = FilterConfig(
            n_taps=args_dict["taps"],
            input_rate=cfg["input_rate"],
            upsample_ratio=cfg["ratio"],
            passband_end=args_dict["passband_end"],
            stopband_start=cfg["stopband"],
            stopband_attenuation_db=args_dict["stopband_attenuation"],
            kaiser_beta=args_dict["kaiser_beta"],
            minimum_phase_method=MinimumPhaseMethod(args_dict["minimum_phase_method"]),
        )
        settings = MixedPhaseSettings(
            crossover_hz=args_dict["crossover_hz"],
            transition_hz=args_dict["transition_hz"],
            delay_ms=args_dict["delay_ms"],
        )
        solver = MixedPhaseSolverSettings(
            iterations=args_dict["iterations"],
            min_iterations=args_dict["min_iterations"],
            phase_blend=args_dict["phase_blend"],
            amplitude_relax=args_dict["amplitude_relax"],
            fft_oversample=args_dict["fft_oversample"],
            gd_smooth_hz=args_dict["gd_smooth_hz"],
            tail_taps=args_dict["tail_taps"],
            tolerance=args_dict["tolerance"],
        )
        generator = MixedPhaseFilterGenerator(config, settings, solver)
        base_name, actual_taps = generator.generate(filter_name=name, skip_header=True)
        return (name, base_name, actual_taps, cfg, None)
    except Exception as exc:  # pragma: no cover - error path
        return (name, "", 0, cfg, str(exc))


def generate_all_filters(args: argparse.Namespace) -> None:
    if args.family == "44k":
        configs = {k: v for k, v in MULTI_RATE_CONFIGS.items() if k.startswith("44k")}
    elif args.family == "48k":
        configs = {k: v for k, v in MULTI_RATE_CONFIGS.items() if k.startswith("48k")}
    else:
        configs = MULTI_RATE_CONFIGS

    worker_count = args.workers or os.cpu_count()

    print("=" * 70)
    print(f"Mixed Phase Multi-Rate Generation - {len(configs)} filters")
    print(
        f"Settings: crossover={args.crossover_hz} Hz, transition={args.transition_hz} Hz, "
        f"delay={'auto' if args.delay_ms is None else args.delay_ms} ms"
    )
    if args.parallel:
        print(f"Parallel Mode: {worker_count} workers")
    print("=" * 70)

    args_dict = {
        "taps": args.taps,
        "passband_end": args.passband_end,
        "stopband_attenuation": args.stopband_attenuation,
        "kaiser_beta": args.kaiser_beta,
        "minimum_phase_method": args.minimum_phase_method,
        "crossover_hz": args.crossover_hz,
        "transition_hz": args.transition_hz,
        "delay_ms": args.delay_ms,
        "iterations": args.iterations,
        "min_iterations": args.min_iterations,
        "phase_blend": args.phase_blend,
        "amplitude_relax": args.amplitude_relax,
        "fft_oversample": args.fft_oversample,
        "gd_smooth_hz": args.gd_smooth_hz,
        "tail_taps": args.tail_taps,
        "tolerance": args.tolerance,
    }

    results: list[tuple[str, str]] = []
    filter_infos: list[tuple[str, str, int, dict[str, Any]]] = []

    if args.parallel and len(configs) > 1:
        worker_args = [(name, cfg, args_dict) for name, cfg in configs.items()]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for name, base_name, actual_taps, cfg, error in executor.map(
                _mixed_phase_worker, worker_args
            ):
                if error:
                    print(f"  ❌ {name}: {error}")
                    results.append((name, "Failed"))
                else:
                    print(f"  ✅ {name}: completed")
                    results.append((name, "Success"))
                    filter_infos.append((name, base_name, actual_taps, cfg))
    else:
        for name, cfg in configs.items():
            print("\n" + "=" * 70)
            print(f"Generating {name}...")
            print("=" * 70)
            local_args = argparse.Namespace(**vars(args))
            local_args.input_rate = cfg["input_rate"]
            local_args.upsample_ratio = cfg["ratio"]
            local_args.stopband_start = cfg["stopband"]
            local_args.output_prefix = None

            try:
                base_name, actual_taps = generate_single_filter(
                    local_args, filter_name=name, skip_header=True
                )
                results.append((name, "Success"))
                filter_infos.append((name, base_name, actual_taps, cfg))
            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append((name, "Failed"))

    if filter_infos:
        generate_multi_rate_header(filter_infos)
        safe_gain = calculate_safe_gain(filter_infos)
        print_safe_gain_recommendation(safe_gain)

    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    for name, status in results:
        print(f"  {name}: {status}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FIR mixed-phase filters with group delay control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single 44.1kHz → 705.6kHz mixed-phase filter
  %(prog)s --input-rate 44100 --upsample-ratio 16

  # Generate all rate families with default settings
  %(prog)s --generate-all

  # Generate with custom crossover/transition
  %(prog)s --generate-all --crossover-hz 90 --transition-hz 25
""",
    )
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Generate all 8 filter configurations (44k/48k × 16x/8x/4x/2x)",
    )
    parser.add_argument(
        "--family",
        type=str,
        choices=["44k", "48k", "all"],
        default="all",
        help="Rate family to generate (only with --generate-all). Default: all",
    )
    parser.add_argument(
        "--input-rate",
        type=int,
        default=44100,
        help="Input sample rate (Hz). Default: 44100",
    )
    parser.add_argument(
        "--upsample-ratio",
        type=int,
        default=16,
        help="Upsampling ratio. Default: 16",
    )
    parser.add_argument(
        "--taps",
        type=int,
        default=640_000,
        help="Number of filter taps. Default: 640000",
    )
    parser.add_argument(
        "--passband-end",
        type=int,
        default=20000,
        help="Passband end frequency (Hz). Default: 20000",
    )
    parser.add_argument(
        "--stopband-start",
        type=int,
        default=None,
        help="Stopband start frequency (Hz). Default: auto (input Nyquist)",
    )
    parser.add_argument(
        "--stopband-attenuation",
        type=int,
        default=160,
        help="Target stopband attenuation (dB). Default: 160",
    )
    parser.add_argument(
        "--kaiser-beta",
        type=float,
        default=28.0,
        help="Kaiser window beta. Default: 28",
    )
    parser.add_argument(
        "--minimum-phase-method",
        type=str,
        choices=["homomorphic", "hilbert"],
        default="homomorphic",
        help="Minimum phase conversion method. Default: homomorphic",
    )
    parser.add_argument(
        "--crossover-hz",
        type=float,
        default=100.0,
        help="Frequency where group delay transitions to constant delay. Default: 100",
    )
    parser.add_argument(
        "--transition-hz",
        type=float,
        default=400.0,
        help="Transition width around the crossover frequency. Default: 400",
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=None,
        help="Absolute delay applied above crossover (ms). Default: auto (1000/crossover)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of solver iterations for phase optimization. Default: 200",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=50,
        help="Minimum iterations before convergence check. Default: 50",
    )
    parser.add_argument(
        "--phase-blend",
        type=float,
        default=0.2,
        help="Phase projection blend factor (0-1). Default: 0.2",
    )
    parser.add_argument(
        "--amplitude-relax",
        type=float,
        default=0.0,
        help="Blend factor for relaxing magnitude constraint (0 disables).",
    )
    parser.add_argument(
        "--fft-oversample",
        type=int,
        default=4,
        help="FFT oversample multiplier for solver. Default: 4",
    )
    parser.add_argument(
        "--gd-smooth-hz",
        type=float,
        default=200.0,
        help="Group delay smoothing bandwidth in Hz. Default: 200",
    )
    parser.add_argument(
        "--tail-taps",
        type=int,
        default=2048,
        help="Number of tail samples to cosine-fade for windowing. Default: 2048",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Phase convergence tolerance (radians). Default: 1e-4",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file basename (without extension). Default: auto",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for --generate-all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel mode. Default: CPU count",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generate_all:
        generate_all_filters(args)
    else:
        print("=" * 70)
        print("GPU Audio Upsampler - Mixed Phase Filter Generation")
        print("=" * 70)
        generate_single_filter(args)


if __name__ == "__main__":
    main()

