#!/usr/bin/env python3
"""
GPU Audio Upsampler - Mixed Phase Filter Generator (Phase EQ approach)

従来の最小位相FIR (640k tap) をベースとしつつ、100〜500 Hz で群遅延を
約3 msに滑らかに遷移させる混合位相フィルタを生成する。

手順:
1. generate_filter.py と同じ設定で線形位相→最小位相FIRを生成 (h_min)
2. h_min の群遅延を測定してターゲット総合群遅延 τ_total を構築
3. 必要な位相EQ群遅延 τ_eq = τ_total - τ_min を計算
4. 短尺 all-pass 近似FIR (h_eq) を周波数サンプリングLSで設計
5. h_mixed = h_min ⊛ h_eq を先頭側優先でトリムし、正規化・検証・出力
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

try:  # optional GPU backend
    import cupy as cp
    HAS_CUPY = True
except Exception:  # pragma: no cover - optional dependency
    cp = None
    HAS_CUPY = False

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
    """位相EQに関するパラメータ"""

    eq_taps: int = 4_096
    eq_delay_ms: float = 3.0
    eq_low_hz: float = 100.0
    eq_high_hz: float = 500.0
    eq_max_freq: float = 20_000.0
    eq_low_cut_hz: float = 60.0
    target_smooth_hz: float = 60.0
    eq_iterations: int = 50_000
    eq_tolerance: float = 1e-6
    eq_step_size: float = 0.004
    eq_oversample: int = 2
    analysis_fft_exp: int = 22
    weight_sub: float = 0.01
    weight_low: float = 0.1
    weight_transition: float = 0.4
    weight_high: float = 1.0
    use_gpu: bool = False

    def validate(self, fs: int) -> None:
        nyquist = fs / 2.0
        if self.eq_taps <= 0:
            raise ValueError("eq_taps must be > 0")
        if not 0.0 < self.eq_low_hz < self.eq_high_hz:
            raise ValueError("0 < eq_low_hz < eq_high_hz must hold")
        if self.eq_high_hz >= self.eq_max_freq:
            raise ValueError("eq_max_freq must be greater than eq_high_hz")
        if self.eq_max_freq >= nyquist:
            raise ValueError("eq_max_freq must be below Nyquist")
        if self.eq_low_cut_hz < 0.0 or self.eq_low_cut_hz >= self.eq_low_hz:
            raise ValueError("eq_low_cut_hz must be between 0 and eq_low_hz")
        if self.target_smooth_hz < 0.0:
            raise ValueError("target_smooth_hz must be non-negative")
        if self.eq_iterations <= 0:
            raise ValueError("eq_iterations must be positive")
        if self.eq_tolerance <= 0.0:
            raise ValueError("eq_tolerance must be positive")
        if self.eq_step_size <= 0.0:
            raise ValueError("eq_step_size must be positive")
        if self.eq_oversample < 2:
            raise ValueError("eq_oversample must be >= 2 for stability")
        if self.analysis_fft_exp < 18:
            raise ValueError("analysis_fft_exp must be >= 18 (≈256k FFT)")

    def delay_samples(self, fs: int) -> float:
        return float(self.eq_delay_ms * 1e-3 * fs)


@dataclass
class GroupDelayAnalysis:
    freqs: np.ndarray
    omega: np.ndarray
    tau_min: np.ndarray
    tau_total: np.ndarray
    tau_eq: np.ndarray
    phase_eq: np.ndarray
    n_fft: int
    const_delay_samples: float


def next_pow_two(value: int) -> int:
    if value <= 0:
        return 1
    return 1 << (value - 1).bit_length()


def smooth_curve(freqs: np.ndarray, data: np.ndarray, width_hz: float) -> np.ndarray:
    if width_hz <= 0.0 or len(freqs) < 3:
        return data
    step = freqs[1] - freqs[0]
    window = max(3, int(round(width_hz / max(step, 1e-9))))
    if window % 2 == 0:
        window += 1
    if window >= len(data):
        return np.full_like(data, float(np.mean(data)))
    kernel = np.hanning(window)
    kernel /= np.sum(kernel)
    return np.convolve(data, kernel, mode="same")


def integrate_phase(omega: np.ndarray, tau_samples: np.ndarray) -> np.ndarray:
    phase = np.zeros_like(omega)
    delta = np.diff(omega)
    avg_tau = 0.5 * (tau_samples[1:] + tau_samples[:-1])
    phase[1:] = -np.cumsum(avg_tau * delta)
    phase -= phase[0]
    return phase


def measure_group_delay(
    h: np.ndarray, fs: int, n_fft: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_fft = max(n_fft, next_pow_two(len(h) * 2))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    omega = 2.0 * np.pi * freqs / fs
    H = np.fft.rfft(h.astype(np.float64), n=n_fft)
    phase = np.unwrap(np.angle(H))
    tau = -np.gradient(phase, omega, edge_order=2)
    return freqs, omega, tau


def build_total_delay_target(
    freqs: np.ndarray, tau_min: np.ndarray, settings: MixedPhaseSettings, fs: int
) -> np.ndarray:
    tau_total = tau_min.copy()
    const_delay = settings.delay_samples(fs)
    low_mask = freqs <= settings.eq_low_hz
    high_mask = freqs >= settings.eq_high_hz
    mid_mask = (~low_mask) & (~high_mask)

    tau_total[high_mask] = const_delay
    if np.any(mid_mask):
        phase = (freqs[mid_mask] - settings.eq_low_hz) / (
            settings.eq_high_hz - settings.eq_low_hz
        )
        smooth = 0.5 - 0.5 * np.cos(np.pi * np.clip(phase, 0.0, 1.0))
        tau_total[mid_mask] = (
            (1.0 - smooth) * tau_min[mid_mask] + smooth * const_delay
        )
    tau_total[low_mask] = tau_min[low_mask]
    return smooth_curve(freqs, tau_total, settings.target_smooth_hz)


def analyze_base_filter(
    h_min: np.ndarray, fs: int, settings: MixedPhaseSettings
) -> GroupDelayAnalysis:
    fft_min = 1 << settings.analysis_fft_exp
    n_fft = max(fft_min, next_pow_two(len(h_min) * 4))
    freqs, omega, tau_min = measure_group_delay(h_min, fs, n_fft)
    tau_min = np.clip(tau_min, 0.0, None)
    tau_total = build_total_delay_target(freqs, tau_min, settings, fs)
    tau_eq = tau_total - tau_min
    tau_eq[freqs < settings.eq_low_cut_hz] = 0.0
    phase_eq = integrate_phase(omega, tau_eq)
    return GroupDelayAnalysis(
        freqs=freqs,
        omega=omega,
        tau_min=tau_min,
        tau_total=tau_total,
        tau_eq=tau_eq,
        phase_eq=phase_eq,
        n_fft=n_fft,
        const_delay_samples=settings.delay_samples(fs),
    )


def build_weight_profile(freqs: np.ndarray, settings: MixedPhaseSettings) -> np.ndarray:
    weights = np.zeros_like(freqs)
    weights[freqs <= 20.0] = settings.weight_sub
    low_mask = (freqs > 20.0) & (freqs <= settings.eq_low_hz)
    trans_mask = (freqs > settings.eq_low_hz) & (freqs <= settings.eq_high_hz)
    high_mask = (freqs > settings.eq_high_hz) & (freqs <= settings.eq_max_freq)

    weights[low_mask] = settings.weight_low
    weights[trans_mask] = settings.weight_transition
    weights[high_mask] = settings.weight_high
    return weights


class PhaseEqDesigner:
    """周波数サンプリング型最小二乗で位相EQ FIRを求める"""

    def __init__(self, fs: int, settings: MixedPhaseSettings) -> None:
        self.fs = fs
        self.settings = settings
        self.n_fft = next_pow_two(settings.eq_taps * settings.eq_oversample)
        if self.n_fft <= settings.eq_taps:
            self.n_fft = next_pow_two(settings.eq_taps * settings.eq_oversample * 2)
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / fs)
        self.use_gpu = bool(getattr(settings, "use_gpu", False) and HAS_CUPY)
        if getattr(settings, "use_gpu", False) and not HAS_CUPY:
            print("[PhaseEqDesigner] CuPy not available, falling back to CPU (NumPy) backend.")

    def design(
        self, phase_target: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Design a phase EQ FIR using weighted least-squares optimization.
        Uses CuPy/CUDA backend if enabled and available.
        """
        if phase_target.shape != self.freqs.shape:
            raise ValueError("phase_target shape mismatch")
        if weights.shape != self.freqs.shape:
            raise ValueError("weights shape mismatch")

        # backend selection
        xp = cp if self.use_gpu else np

        taps = self.settings.eq_taps
        # host copies for safety, then move to backend
        phase_host = phase_target.astype(np.float64, copy=False)
        weights_host = weights.astype(np.float64, copy=False)
        target_host = np.exp(1j * phase_host)

        phase_x = xp.asarray(phase_host) if self.use_gpu else phase_host
        weights_x = xp.asarray(weights_host) if self.use_gpu else weights_host
        target_x = xp.asarray(target_host) if self.use_gpu else target_host

        h = xp.zeros(taps, dtype=xp.float64)
        h[0] = 1.0
        padded = xp.zeros(self.n_fft, dtype=xp.float64)

        history: list[float] = []
        threshold = max(float(np.max(weights_host)), 1e-9)

        def to_scalar(val: Any) -> float:
            if self.use_gpu:
                # CuPy scalar to Python float
                return float(val.get())  # type: ignore[attr-defined]
            return float(val)

        iterations_used = self.settings.eq_iterations
        for idx in range(1, self.settings.eq_iterations + 1):
            # forward FFT
            padded[:taps] = h
            if self.n_fft > taps:
                padded[taps:] = 0.0
            H = xp.fft.rfft(padded, n=self.n_fft)

            # weighted complex error
            error = weights_x * (H - target_x)
            loss_val = xp.sqrt(xp.mean(xp.abs(error) ** 2))
            loss = to_scalar(loss_val)
            history.append(loss)

            # gradient via IFFT
            grad = 2.0 * xp.fft.irfft(error, n=self.n_fft).real[:taps]
            h -= self.settings.eq_step_size * grad

            # very light DC normalization & decay on higher taps to avoid drift
            dc = to_scalar(xp.sum(h))
            if abs(dc) > 1e-12:
                h /= dc
            if taps > 1:
                h[1:] *= 0.999

            if loss < self.settings.eq_tolerance * threshold:
                iterations_used = idx
                break
        else:
            iterations_used = self.settings.eq_iterations

        # bring coefficients back to CPU (NumPy)
        h_host = cp.asnumpy(h) if self.use_gpu else np.asarray(h)

        diagnostics = {
            "eq_taps": taps,
            "design_fft": self.n_fft,
            "iterations": iterations_used,
            "final_loss": history[-1] if history else None,
            "step_size": self.settings.eq_step_size,
            "tolerance": self.settings.eq_tolerance,
            "loss_history": history[: min(len(history), 32)],
            "max_abs_coefficient": float(np.max(np.abs(h_host))),
            "backend": "cupy" if self.use_gpu else "numpy",
        }
        return h_host, diagnostics


def summarize_group_delay(
    analysis: GroupDelayAnalysis,
    tau_actual: np.ndarray,
    fs: int,
    band: tuple[float, float] = (20.0, 20_000.0),
) -> dict[str, Any]:
    freqs = analysis.freqs
    tau_target = analysis.tau_total
    diff = tau_actual - tau_target
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        mask = slice(None)
    max_error = float(np.max(np.abs(diff[mask])))
    rms_error = float(np.sqrt(np.mean(diff[mask] ** 2)))

    sample_freqs = [100.0, 500.0, 1000.0, 5000.0, 10_000.0]
    sample_stats = []
    for freq in sample_freqs:
        idx = int(np.argmin(np.abs(freqs - freq)))
        sample_stats.append(
            {
                "frequency_hz": float(freqs[idx]),
                "target_delay_ms": float(tau_target[idx] / fs * 1000.0),
                "actual_delay_ms": float(tau_actual[idx] / fs * 1000.0),
                "error_samples": float(diff[idx]),
            }
        )

    return {
        "band_hz": list(band),
        "max_group_delay_error_samples": max_error,
        "rms_group_delay_error_samples": rms_error,
        "target_const_delay_ms": float(analysis.const_delay_samples / fs * 1000.0),
        "analysis_fft": analysis.n_fft,
        "sample_points": sample_stats,
    }


class MixedPhaseGenerator:
    """混合位相フィルタ生成のオーケストレーション"""

    def __init__(self, config: FilterConfig, settings: MixedPhaseSettings) -> None:
        self.config = config
        self.settings = settings
        self.designer = FilterDesigner(config)
        self.validator = FilterValidator(config)
        self.exporter = FilterExporter(config)
        self.plotter = FilterPlotter(config)

    def generate(
        self, filter_name: str | None = None, skip_header: bool = False
    ) -> tuple[str, int]:
        fs = self.config.output_rate
        validate_tap_count(self.config.n_taps, self.config.upsample_ratio)
        self.settings.validate(fs)

        print("\n=== Phase 1: 基準最小位相FIR生成 ===")
        h_linear = self.designer.design_linear_phase()
        h_min = self.designer.convert_to_minimum_phase(h_linear)

        print("\n=== Phase 2: 群遅延ターゲット解析 ===")
        analysis = analyze_base_filter(h_min, fs, self.settings)

        print("\n=== Phase 3: 位相EQ FIR設計 ===")
        eq_designer = PhaseEqDesigner(fs, self.settings)
        phase_interp = np.interp(
            eq_designer.freqs,
            analysis.freqs,
            analysis.phase_eq,
            left=0.0,
            right=float(analysis.phase_eq[-1]),
        )
        weights = build_weight_profile(eq_designer.freqs, self.settings)
        weights[eq_designer.freqs > self.settings.eq_max_freq] = 0.0
        h_eq, eq_info = eq_designer.design(phase_interp, weights)

        print("\n=== Phase 4: 畳み込み・トリミング ===")
        h_total = signal.fftconvolve(
            h_min.astype(np.float64), h_eq.astype(np.float64)
        )
        h_mixed = h_total[: len(h_min)]

        h_final, normalization = normalize_coefficients(
            h_mixed,
            target_dc_gain=self.config.target_dc_gain,
            dc_gain_factor=self.config.dc_gain_factor,
        )

        validation = self.validator.validate(h_final)
        freqs_actual, _, tau_actual = measure_group_delay(
            h_final, fs, analysis.n_fft
        )
        if not np.allclose(freqs_actual, analysis.freqs):
            tau_actual = np.interp(analysis.freqs, freqs_actual, tau_actual)
        metrics = summarize_group_delay(analysis, tau_actual, fs)

        validation["group_delay_metrics"] = metrics
        validation["phase_eq"] = {
            **eq_info,
            "eq_delay_ms": self.settings.eq_delay_ms,
            "eq_low_hz": self.settings.eq_low_hz,
            "eq_high_hz": self.settings.eq_high_hz,
            "eq_max_freq": self.settings.eq_max_freq,
        }
        validation["normalization"] = normalization

        print("\n=== Phase 5: 可視化 ===")
        self.plotter.plot(h_final, h_linear, filter_name)

        metadata = self._create_metadata(validation, metrics, eq_info)
        base_name = self.exporter.export(h_final, metadata, skip_header)
        self._print_report(base_name, validation)
        return base_name, len(h_final)

    def _create_metadata(
        self,
        validation: dict[str, Any],
        metrics: dict[str, Any],
        eq_info: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "generation_date": datetime.now().isoformat(),
            "generation_mode": "mixed_phase",
            "mixed_phase_settings": asdict(self.settings),
            "sample_rate_input": self.config.input_rate,
            "sample_rate_output": self.config.output_rate,
            "upsample_ratio": self.config.upsample_ratio,
            "n_taps_specified": self.config.n_taps,
            "passband_end_hz": self.config.passband_end,
            "stopband_start_hz": self.config.stopband_start,
            "target_stopband_attenuation_db": self.config.stopband_attenuation_db,
            "kaiser_beta": self.config.kaiser_beta,
            "minimum_phase_method": self.config.minimum_phase_method.value,
            "target_dc_gain": self.config.target_dc_gain,
            "output_basename": self.config.base_name,
            "validation_results": validation,
            "group_delay_metrics": metrics,
            "phase_eq_diagnostics": eq_info,
        }

    def _print_report(self, base_name: str, validation: dict[str, Any]) -> None:
        metrics = validation.get("group_delay_metrics", {})
        print("\n" + "=" * 70)
        print(f"完了 - {validation.get('actual_taps', self.config.n_taps):,}タップ混合位相FIR")
        print("=" * 70)
        print(
            f"阻止帯域減衰: {validation['stopband_attenuation_db']:.1f} dB "
            f"(目標 {self.config.stopband_attenuation_db} dB)"
        )
        if metrics:
            print(
                "群遅延誤差: "
                f"max={metrics.get('max_group_delay_error_samples', 0.0):.3f} samples, "
                f"rms={metrics.get('rms_group_delay_error_samples', 0.0):.3f} samples"
            )
        print(f"係数ファイル: data/coefficients/{base_name}.bin")


def build_filter_config(args: argparse.Namespace) -> FilterConfig:
    stopband = args.stopband_start if args.stopband_start else args.input_rate // 2
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        family = "44k" if args.input_rate % 44100 == 0 else "48k"
        taps_label = "2m" if args.taps == 640_000 else str(args.taps)
        output_prefix = f"filter_{family}_{args.upsample_ratio}x_{taps_label}_hybrid_phase"
    return FilterConfig(
        n_taps=args.taps,
        input_rate=args.input_rate,
        upsample_ratio=args.upsample_ratio,
        passband_end=args.passband_end,
        stopband_start=stopband,
        stopband_attenuation_db=args.stopband_attenuation,
        kaiser_beta=args.kaiser_beta,
        minimum_phase_method=MinimumPhaseMethod(args.minimum_phase_method),
        output_prefix=output_prefix,
    )


def build_mixed_settings(args: argparse.Namespace) -> MixedPhaseSettings:
    return MixedPhaseSettings(
        eq_taps=args.eq_taps,
        eq_delay_ms=args.eq_delay_ms,
        eq_low_hz=args.eq_low_hz,
        eq_high_hz=args.eq_high_hz,
        eq_max_freq=args.eq_max_freq,
        eq_low_cut_hz=args.eq_low_cut_hz,
        target_smooth_hz=args.eq_target_smooth_hz,
        eq_iterations=args.eq_iterations,
        eq_tolerance=args.eq_tolerance,
        eq_step_size=args.eq_step_size,
        eq_oversample=args.eq_oversample,
        analysis_fft_exp=args.analysis_fft_exp,
        use_gpu=getattr(args, "eq_use_gpu", False),
    )


def generate_single_filter(
    args: argparse.Namespace, filter_name: str | None = None, skip_header: bool = False
) -> tuple[str, int]:
    config = build_filter_config(args)
    settings = build_mixed_settings(args)
    generator = MixedPhaseGenerator(config, settings)
    return generator.generate(filter_name, skip_header)


def generate_all_filters(args: argparse.Namespace) -> None:
    if args.family == "44k":
        configs = {k: v for k, v in MULTI_RATE_CONFIGS.items() if k.startswith("44k")}
    elif args.family == "48k":
        configs = {k: v for k, v in MULTI_RATE_CONFIGS.items() if k.startswith("48k")}
    else:
        configs = MULTI_RATE_CONFIGS

    results: list[tuple[str, str]] = []
    filter_infos: list[tuple[str, str, int, dict[str, Any]]] = []
    print("=" * 70)
    print(f"Mixed Phase Multi-Rate Generation - {len(configs)} filters")
    print(
        f"Group delay target: low={args.eq_low_hz} Hz, high={args.eq_high_hz} Hz, "
        f"delay={args.eq_delay_ms} ms"
    )
    print("=" * 70)

    for name, cfg in configs.items():
        print("\n" + "-" * 70)
        print(f"Generating {name} ({cfg['input_rate']} Hz × {cfg['ratio']}x)")
        print("-" * 70)
        local_args = copy.deepcopy(args)
        local_args.input_rate = cfg["input_rate"]
        local_args.upsample_ratio = cfg["ratio"]
        local_args.stopband_start = cfg["stopband"]
        local_args.output_prefix = None
        try:
            base_name, actual_taps = generate_single_filter(
                local_args, filter_name=name, skip_header=True
            )
            filter_infos.append((name, base_name, actual_taps, cfg))
            results.append((name, "Success"))
        except Exception as exc:  # pragma: no cover
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
        description="Generate mixed-phase FIR filters with controlled group delay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generate-all", action="store_true", help="Generate all rate families.")
    parser.add_argument(
        "--family",
        type=str,
        choices=["44k", "48k", "all"],
        default="all",
        help="Rate family to use with --generate-all.",
    )
    parser.add_argument("--input-rate", type=int, default=44100, help="Input sample rate (Hz).")
    parser.add_argument("--upsample-ratio", type=int, default=16, help="Upsampling ratio.")
    parser.add_argument("--taps", type=int, default=640_000, help="Target tap count.")
    parser.add_argument(
        "--passband-end", type=int, default=20000, help="Passband end frequency (Hz)."
    )
    parser.add_argument(
        "--stopband-start",
        type=int,
        default=None,
        help="Stopband start frequency (Hz). Default: input Nyquist.",
    )
    parser.add_argument(
        "--stopband-attenuation",
        type=int,
        default=160,
        help="Target stopband attenuation (dB).",
    )
    parser.add_argument("--kaiser-beta", type=float, default=28.0, help="Kaiser window beta.")
    parser.add_argument(
        "--minimum-phase-method",
        type=str,
        choices=["homomorphic", "hilbert"],
        default="homomorphic",
        help="Minimum phase conversion method.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output basename (otherwise auto-generated).",
    )
    # Mixed-phase specific parameters
    parser.add_argument("--eq-taps", type=int, default=4_096, help="Phase EQ FIR length.")
    parser.add_argument("--eq-delay-ms", type=float, default=3.0, help="Target constant delay (ms).")
    parser.add_argument("--eq-low-hz", type=float, default=100.0, help="Delay ramp start frequency.")
    parser.add_argument(
        "--eq-high-hz", type=float, default=500.0, help="Delay ramp end frequency."
    )
    parser.add_argument(
        "--eq-max-freq",
        type=float,
        default=20_000.0,
        help="Upper frequency where phase EQ is enforced.",
    )
    parser.add_argument(
        "--eq-low-cut-hz",
        type=float,
        default=60.0,
        help="Frequencies below this keep τ_eq = 0.",
    )
    parser.add_argument(
        "--eq-target-smooth-hz",
        type=float,
        default=60.0,
        help="Hann smoothing bandwidth for τ_total.",
    )
    parser.add_argument(
        "--eq-iterations", type=int, default=50_000, help="Phase EQ solver iterations."
    )
    parser.add_argument(
        "--eq-tolerance",
        type=float,
        default=1e-6,
        help="Phase EQ convergence tolerance (relative).",
    )
    parser.add_argument(
        "--eq-step-size",
        type=float,
        default=0.004,
        help="Gradient descent step size for phase EQ.",
    )
    parser.add_argument(
        "--eq-oversample",
        type=int,
        default=2,
        help="FFT oversample factor for EQ design.",
    )
    parser.add_argument(
        "--analysis-fft-exp",
        type=int,
        default=22,
        help="Log2 FFT size used for group delay measurement (e.g., 22 → 4M points).",
    )
    parser.add_argument(
        "--eq-use-gpu",
        action="store_true",
        help="Use CuPy/CUDA backend for phase EQ optimization if available.",
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


