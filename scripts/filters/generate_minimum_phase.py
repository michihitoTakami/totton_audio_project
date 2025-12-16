#!/usr/bin/env python3
"""Minimum phase filter generator CLI."""

from __future__ import annotations

import argparse
import copy
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
from scipy import signal

from .generate_filter import (
    FilterConfig,
    FilterExporter,
    FilterPlotter,
    FilterValidator,
    MinimumPhaseMethod,
    MULTI_RATE_CONFIGS,
    validate_tap_count,
    normalize_coefficients,
    generate_multi_rate_header,
    calculate_safe_gain,
    print_safe_gain_recommendation,
)

try:
    import cupy as cp
    from cupyx.scipy import fft as cp_fft

    CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional GPU support
    cp = None
    cp_fft = None
    CUPY_AVAILABLE = False


class MinimumPhaseDesigner:
    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def design_linear_phase(self) -> np.ndarray:
        cutoff = (self.config.passband_end + self.config.stopband_start) / 2
        nyquist = self.config.output_rate / 2
        normalized_cutoff = cutoff / nyquist
        numtaps = (
            self.config.n_taps
            if self.config.n_taps % 2 == 1
            else self.config.n_taps + 1
        )
        print("基準線形位相FIRフィルタ設計中...")
        print(f"  タップ数: {numtaps}")
        print(f"  カットオフ周波数: {cutoff} Hz (正規化: {normalized_cutoff:.6f})")
        print(f"  Kaiser β: {self.config.kaiser_beta}")
        return signal.firwin(
            numtaps=numtaps,
            cutoff=normalized_cutoff,
            window=("kaiser", self.config.kaiser_beta),
            fs=1.0,
            scale=True,
        )

    def convert_to_minimum_phase(self, h_linear: np.ndarray) -> np.ndarray:
        print("\n最小位相変換中...")
        n_fft = 2 ** int(np.ceil(np.log2(len(h_linear) * 8)))
        print(f"  FFTサイズ: {n_fft:,}")
        if (
            CUPY_AVAILABLE
            and self.config.minimum_phase_method == MinimumPhaseMethod.HOMOMORPHIC
        ):
            print("  🚀 GPU高速化（CuPy）を使用")
            h_min = self._convert_to_minimum_phase_gpu(h_linear, n_fft)
        else:
            if not CUPY_AVAILABLE:
                print("  ⚠️ CuPyが利用できません。CPU版を使用")
            else:
                print(
                    f"  CPU版を使用（method={self.config.minimum_phase_method.value}）"
                )
            h_min = signal.minimum_phase(
                h_linear,
                method=self.config.minimum_phase_method.value,
                n_fft=n_fft,
            )
        if len(h_min) > self.config.n_taps:
            h_min = h_min[: self.config.n_taps]
        elif len(h_min) < self.config.n_taps:
            h_min = np.pad(h_min, (0, self.config.n_taps - len(h_min)))
        print(f"  最小位相係数タップ数: {len(h_min)}")
        return h_min

    def _convert_to_minimum_phase_gpu(
        self, h_linear: np.ndarray, n_fft: int
    ) -> np.ndarray:
        import time

        start = time.time()
        h_gpu = cp.asarray(h_linear, dtype=cp.float64)
        h_padded = cp.zeros(n_fft, dtype=cp.float64)
        h_padded[: len(h_linear)] = h_gpu
        H = cp_fft.fft(h_padded)
        eps = cp.finfo(cp.float64).eps
        log_H = cp.log(cp.maximum(cp.abs(H), eps))
        cepstrum = cp_fft.ifft(log_H).real
        causal = cp.zeros_like(cepstrum)
        causal[0] = cepstrum[0]
        if n_fft % 2 == 0:
            causal[1 : n_fft // 2] = 2 * cepstrum[1 : n_fft // 2]
            causal[n_fft // 2] = cepstrum[n_fft // 2]
        else:
            causal[1 : (n_fft + 1) // 2] = 2 * cepstrum[1 : (n_fft + 1) // 2]
        H_min = cp.exp(cp_fft.fft(causal))
        h_min = cp.asnumpy(cp_fft.ifft(H_min).real[: (len(h_linear) + 1) // 2])
        elapsed = time.time() - start
        print(f"  GPU処理時間: {elapsed:.2f}秒")
        return h_min

    def design(self) -> tuple[np.ndarray, np.ndarray]:
        h_linear = self.design_linear_phase()
        h_min = self.convert_to_minimum_phase(h_linear)
        return h_min, h_linear


class MinimumPhaseGenerator:
    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self.designer = MinimumPhaseDesigner(config)
        self.validator = FilterValidator(config)
        self.exporter = FilterExporter(config)
        self.plotter = FilterPlotter(config)

    def generate(
        self, filter_name: str | None = None, skip_header: bool = False
    ) -> tuple[str, int]:
        validate_tap_count(self.config.n_taps, self.config.upsample_ratio)
        h_final, h_linear = self.designer.design()
        h_final, normalization = normalize_coefficients(
            h_final,
            target_dc_gain=self.config.target_dc_gain,
            dc_gain_factor=self.config.dc_gain_factor,
        )
        validation = self.validator.validate(h_final)
        validation["normalization"] = normalization
        self.plotter.plot(h_final, h_linear, filter_name)
        metadata = self._create_metadata(validation)
        base_name = self.exporter.export(h_final, metadata, skip_header)
        self._print_report(validation, normalization, base_name)
        return base_name, validation.get("actual_taps", len(h_final))

    def _create_metadata(self, validation: dict[str, Any]) -> dict[str, Any]:
        return {
            "generation_date": datetime.now().isoformat(),
            "n_taps_specified": self.config.n_taps,
            "n_taps_actual": validation.get("actual_taps", self.config.final_taps),
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

    def _print_report(
        self, validation: dict[str, Any], normalization: dict[str, Any], base_name: str
    ) -> None:
        actual_taps = validation.get("actual_taps", self.config.final_taps)
        print("\n" + "=" * 70)
        if actual_taps != self.config.n_taps:
            print(
                f"完了 - {self.config.n_taps:,}→{actual_taps:,}タップフィルタ（パディング）"
            )
        else:
            print(f"完了 - {actual_taps:,}タップフィルタ")
        print("=" * 70)
        print(f"阻止帯域減衰: {validation['stopband_attenuation_db']:.1f} dB")
        spec_status = "合格" if validation["meets_stopband_spec"] else "不合格"
        print(f"  {spec_status} (目標: {self.config.stopband_attenuation_db} dB以上)")
        print(
            "係数正規化: "
            f"目標DC={normalization['target_dc_gain']:.6f}, "
            f"結果DC={normalization['normalized_dc_gain']:.6f}"
        )
        max_coef = normalization.get("max_coefficient_amplitude", 0.0)
        print(f"最大係数振幅: {max_coef:.6f}")
        if max_coef > 1.0:
            print("  ⚠️ CUDA側で補正が必要（#260参照）")
        print(
            f"係数ファイル: data/coefficients/{base_name}.bin ({actual_taps:,} coeffs)"
        )
        print("検証プロット: plots/analysis/")
        print("=" * 70)


def _create_config_from_args(args: argparse.Namespace) -> FilterConfig:
    return FilterConfig(
        n_taps=args.taps,
        input_rate=args.input_rate,
        upsample_ratio=args.upsample_ratio,
        passband_end=args.passband_end,
        stopband_start=args.stopband_start,
        stopband_attenuation_db=args.stopband_attenuation,
        kaiser_beta=args.kaiser_beta,
        minimum_phase_method=MinimumPhaseMethod(args.minimum_phase_method),
        output_prefix=args.output_prefix,
    )


def generate_single_filter(
    args: argparse.Namespace, filter_name: str | None = None, skip_header: bool = False
) -> tuple[str, int]:
    config = _create_config_from_args(args)
    generator = MinimumPhaseGenerator(config)
    return generator.generate(filter_name=filter_name, skip_header=skip_header)


def _generate_filter_worker(
    worker_args: tuple[str, dict[str, Any], dict[str, Any]],
) -> tuple[str, str, int, dict[str, Any], str | None]:
    name, cfg, arg_values = worker_args
    try:
        config = FilterConfig(
            n_taps=arg_values["taps"],
            input_rate=cfg["input_rate"],
            upsample_ratio=cfg["ratio"],
            passband_end=arg_values["passband_end"],
            stopband_start=cfg["stopband"],
            stopband_attenuation_db=arg_values["stopband_attenuation"],
            kaiser_beta=arg_values["kaiser_beta"],
            minimum_phase_method=MinimumPhaseMethod(arg_values["minimum_phase_method"]),
            output_prefix=None,
        )
        generator = MinimumPhaseGenerator(config)
        base_name, actual_taps = generator.generate(filter_name=name, skip_header=True)
        return (name, base_name, actual_taps, cfg, None)
    except Exception as exc:  # pragma: no cover - propagate failures
        return (name, "", 0, cfg, str(exc))


def generate_all_filters(args: argparse.Namespace) -> None:
    if args.family == "44k":
        configs = {k: v for k, v in MULTI_RATE_CONFIGS.items() if k.startswith("44k")}
    elif args.family == "48k":
        configs = {k: v for k, v in MULTI_RATE_CONFIGS.items() if k.startswith("48k")}
    else:
        configs = MULTI_RATE_CONFIGS

    total = len(configs)
    print("=" * 70)
    print(f"Multi-Rate Filter Generation - {total} filters")
    if args.parallel:
        workers = args.workers or os.cpu_count()
        print(f"Parallel Mode: {workers} workers")
    print("=" * 70)
    for name, cfg in configs.items():
        output_rate = cfg["input_rate"] * cfg["ratio"]
        print(f"  {name}: {cfg['input_rate']}Hz × {cfg['ratio']}x → {output_rate}Hz")

    if args.output_prefix:
        print("\n注意: --output-prefix は --generate-all 時は無視されます")
    print()

    arg_values = {
        "taps": args.taps,
        "passband_end": args.passband_end,
        "stopband_attenuation": args.stopband_attenuation,
        "kaiser_beta": args.kaiser_beta,
        "minimum_phase_method": args.minimum_phase_method,
    }

    results: list[tuple[str, str]] = []
    filter_infos: list[tuple[str, str, int, dict[str, Any]]] = []

    if args.parallel and total > 1:
        workers = args.workers or os.cpu_count()
        worker_args = [(name, cfg, arg_values) for name, cfg in configs.items()]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for name, base_name, actual_taps, cfg, error in executor.map(
                _generate_filter_worker, worker_args
            ):
                if error:
                    results.append((name, f"Failed: {error}"))
                    print(f"  ❌ {name}: {error}")
                else:
                    results.append((name, "Success"))
                    filter_infos.append((name, base_name, actual_taps, cfg))
                    print(f"  ✅ {name}: completed")
    else:
        for index, (name, cfg) in enumerate(configs.items(), start=1):
            print("\n" + "=" * 70)
            print(f"[{index}/{total}] Generating {name}...")
            print("=" * 70)
            local_args = copy.copy(args)
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
                results.append((name, f"Failed: {exc}"))
                print(f"ERROR: {exc}")

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
    success_count = sum(1 for _, status in results if status == "Success")
    print(f"\nCompleted: {success_count}/{total} filters generated successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate minimum-phase FIR filter coefficients.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Generate all 8 filter configurations",
    )
    parser.add_argument(
        "--family",
        type=str,
        choices=["44k", "48k", "all"],
        default="all",
        help="Menu family when --generate-all is used",
    )
    parser.add_argument(
        "--input-rate", type=int, default=44100, help="Input sample rate (Hz)"
    )
    parser.add_argument("--upsample-ratio", type=int, default=16, help="Upsample ratio")
    parser.add_argument(
        "--taps", type=int, default=640_000, help="Number of filter taps"
    )
    parser.add_argument(
        "--passband-end", type=int, default=20000, help="Passband end frequency"
    )
    parser.add_argument(
        "--stopband-start",
        type=int,
        default=None,
        help="Stopband start frequency (default: input Nyquist)",
    )
    parser.add_argument(
        "--stopband-attenuation",
        type=int,
        default=160,
        help="Stopband attenuation target (dB)",
    )
    parser.add_argument(
        "--kaiser-beta", type=float, default=28.0, help="Kaiser window β"
    )
    parser.add_argument(
        "--minimum-phase-method",
        type=str,
        choices=[m.value for m in MinimumPhaseMethod],
        default=MinimumPhaseMethod.HOMOMORPHIC.value,
        help="Minimum phase conversion method",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Custom basename (single-run only)",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Enable parallel generation"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Worker count for parallel mode"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generate_all:
        generate_all_filters(args)
    else:
        print("=" * 70)
        print("GPU Audio Upsampler - Minimum Phase Filter Generator")
        print("=" * 70)
        generate_single_filter(args)


if __name__ == "__main__":
    main()
