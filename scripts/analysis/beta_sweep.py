#!/usr/bin/env python3
"""
Beta sweep simulator for Kaiser-window FIR filters.

目的:
- カイザー窓βをスイープし、Float64設計値とFloat32量子化後の特性差を数値化
- ストップバンド減衰、通過帯域リップル、係数量子化ノイズなどを比較
- βを上げても改善が頭打ちになるポイントを推定

使い方:
    uv run python scripts/analysis/beta_sweep.py --taps 131072 --betas 10 15 20 25 30

    # 44.1kHz→16xアップサンプル、最小位相化も評価
    uv run python scripts/analysis/beta_sweep.py --phase minimum --taps 2000000 \\
        --beta-min 12 --beta-max 32 --beta-step 2

出力:
- βごとの統計テーブル
- --output-csv/--plot 指定時はCSVとプロットも保存
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import signal


EPS = 1e-20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kaiser β sweep simulator (Float64 vs Float32 characteristics)"
    )
    parser.add_argument(
        "--input-rate", type=int, default=44100, help="入力サンプルレート"
    )
    parser.add_argument(
        "--upsample-ratio", type=int, default=16, help="アップサンプリング比率"
    )
    parser.add_argument(
        "--passband-end",
        type=float,
        default=20000.0,
        help="通過帯域終端 [Hz]",
    )
    parser.add_argument(
        "--stopband-start",
        type=float,
        default=None,
        help="阻止帯域開始 [Hz]（未指定時は入力Nyquist）",
    )
    parser.add_argument("--taps", type=int, default=131072, help="FIRタップ数")
    parser.add_argument(
        "--phase",
        choices=("linear", "minimum"),
        default="linear",
        help="位相タイプ（最小位相はβごとにminimum_phase変換）",
    )
    parser.add_argument(
        "--minimum-phase-method",
        choices=("homomorphic", "hilbert"),
        default="homomorphic",
        help="scipy.signal.minimum_phase に渡す手法",
    )
    parser.add_argument(
        "--freq-points",
        type=int,
        default=32768,
        help="周波数応答評価ポイント数 (signal.freqz の worN)",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        help="評価するβ値の明示リスト（指定時は min/max/step を無視）",
    )
    parser.add_argument(
        "--beta-min", type=float, default=10.0, help="βスイープ下限（betas未指定時）"
    )
    parser.add_argument(
        "--beta-max", type=float, default=40.0, help="βスイープ上限（betas未指定時）"
    )
    parser.add_argument(
        "--beta-step", type=float, default=1.0, help="βスイープ刻み（betas未指定時）"
    )
    parser.add_argument(
        "--target-dc-gain",
        type=float,
        default=None,
        help="DCゲインのターゲット。未指定時はアップサンプル比に合わせる",
    )
    parser.add_argument(
        "--target-stopband-db",
        type=float,
        default=160.0,
        help="目標とする阻止帯域減衰[dB]（比較用）",
    )
    parser.add_argument(
        "--plateau-threshold-db",
        type=float,
        default=1.0,
        help="隣接β間の改善幅がこの値未満になったら頭打ちとみなす",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="結果をCSVで保存するパス",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="β vs 阻止帯域/差分グラフを plots/analysis/beta_sweep.png に保存",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("plots/analysis/beta_sweep.png"),
        help="--plot時の保存先",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class SweepConfig:
    input_rate: int
    upsample_ratio: int
    passband_end: float
    stopband_start: float
    taps: int
    phase: str
    minimum_phase_method: str
    freq_points: int
    target_dc_gain: float

    @property
    def output_rate(self) -> int:
        return self.input_rate * self.upsample_ratio


@dataclass
class BetaResult:
    beta: float
    stopband64_db: float
    stopband32_db: float
    passband_ripple64_db: float
    passband_ripple32_db: float
    stopband_loss_db: float
    max_response_diff_db: float
    coeff_snr_db: float
    zero_ratio: float

    @property
    def effective_bits(self) -> float:
        return self.coeff_snr_db / 6.02


def build_beta_list(args: argparse.Namespace) -> list[float]:
    if args.betas:
        betas = sorted({float(b) for b in args.betas if b >= 0})
    else:
        if args.beta_step <= 0:
            raise ValueError("--beta-step は正である必要があります")
        count = int(np.floor((args.beta_max - args.beta_min) / args.beta_step)) + 1
        betas = [
            round(args.beta_min + i * args.beta_step, 6)
            for i in range(max(count, 1))
            if args.beta_min + i * args.beta_step <= args.beta_max + 1e-6
        ]
    if not betas:
        raise ValueError(
            "βリストが空です。--betas か min/max/step を確認してください。"
        )
    return betas


def normalize_dc_gain(h: np.ndarray, target: float) -> np.ndarray:
    dc = np.sum(h)
    if abs(dc) < 1e-20:
        raise ValueError("DCゲインが0に近いため正規化できません。")
    return h * (target / dc)


def design_filter(beta: float, cfg: SweepConfig) -> np.ndarray:
    cutoff = (cfg.passband_end + cfg.stopband_start) / 2.0
    nyquist = cfg.output_rate / 2.0
    normalized_cutoff = cutoff / nyquist

    if cfg.phase == "minimum" and cfg.taps % 2 == 0:
        design_taps = cfg.taps + 1
    else:
        design_taps = cfg.taps

    h_linear = signal.firwin(
        numtaps=design_taps,
        cutoff=normalized_cutoff,
        window=("kaiser", beta),
        fs=1.0,
        scale=True,
    )

    if cfg.phase == "minimum":
        n_fft = 2 ** int(np.ceil(np.log2(len(h_linear) * 8)))
        h_min = signal.minimum_phase(
            h_linear, method=cfg.minimum_phase_method, n_fft=n_fft
        )
        if len(h_min) >= cfg.taps:
            h = h_min[: cfg.taps]
        else:
            h = np.pad(h_min, (0, cfg.taps - len(h_min)))
    else:
        h = h_linear[: cfg.taps]

    return h


def compute_response(h: np.ndarray, cfg: SweepConfig) -> tuple[np.ndarray, np.ndarray]:
    w, H = signal.freqz(h, worN=cfg.freq_points, fs=cfg.output_rate)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(H), EPS))
    return w, magnitude_db


def summarize_response(
    freqs: np.ndarray, magnitude_db: np.ndarray, cfg: SweepConfig
) -> tuple[float, float]:
    passband_mask = freqs <= cfg.passband_end
    stopband_mask = freqs >= cfg.stopband_start

    passband_ripple = (
        magnitude_db[passband_mask].max() - magnitude_db[passband_mask].min()
    )
    stopband_min = magnitude_db[stopband_mask].min()
    return float(passband_ripple), float(stopband_min)


def evaluate_beta(beta: float, cfg: SweepConfig) -> BetaResult:
    h64 = design_filter(beta, cfg)
    h64 = normalize_dc_gain(h64, cfg.target_dc_gain)

    h32 = h64.astype(np.float32).astype(np.float64)

    freqs, resp64 = compute_response(h64, cfg)
    _, resp32 = compute_response(h32, cfg)

    ripple64, stop64 = summarize_response(freqs, resp64, cfg)
    ripple32, stop32 = summarize_response(freqs, resp32, cfg)

    stop_loss = abs(stop64) - abs(stop32)
    max_resp_diff = float(np.max(np.abs(resp64 - resp32)))

    coeff_error = h64 - h32
    rms_error = float(np.sqrt(np.mean(coeff_error**2)))
    peak_coeff = float(np.max(np.abs(h64)))
    coeff_snr = 20 * np.log10((peak_coeff + EPS) / (rms_error + EPS))
    zero_ratio = float(np.mean(h32 == 0.0))

    return BetaResult(
        beta=beta,
        stopband64_db=stop64,
        stopband32_db=stop32,
        passband_ripple64_db=ripple64,
        passband_ripple32_db=ripple32,
        stopband_loss_db=stop_loss,
        max_response_diff_db=max_resp_diff,
        coeff_snr_db=coeff_snr,
        zero_ratio=zero_ratio,
    )


def estimate_plateau_beta(
    results: Sequence[BetaResult], threshold_db: float
) -> float | None:
    if len(results) < 2:
        return None
    prev_abs = abs(results[0].stopband32_db)
    for current in results[1:]:
        current_abs = abs(current.stopband32_db)
        improvement = current_abs - prev_abs
        if improvement < threshold_db:
            return current.beta
        prev_abs = current_abs
    return None


def render_table(results: Sequence[BetaResult], target_stopband: float) -> None:
    header = (
        f"{'β':>6} {'|H|64[dB]':>12} {'|H|32[dB]':>12} {'Loss[dB]':>9} "
        f"{'Ripple32[dB]':>13} {'∆resp[dB]':>10} {'CoeffSNR[dB]':>13} {'Zeros[%]':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        stop64 = abs(r.stopband64_db)
        stop32 = abs(r.stopband32_db)
        print(
            f"{r.beta:6.2f} {stop64:12.2f} {stop32:12.2f} "
            f"{r.stopband_loss_db:9.2f} {r.passband_ripple32_db:13.4f} "
            f"{r.max_response_diff_db:10.3f} {r.coeff_snr_db:13.2f} {r.zero_ratio*100:9.4f}"
        )
    best = max(results, key=lambda r: abs(r.stopband32_db))
    print(
        "\n最良(実効)阻止帯域: "
        f"β={best.beta:.2f}, |H|32={abs(best.stopband32_db):.2f} dB "
        f"(目標 {target_stopband:.1f} dB)"
    )


def save_csv(path: Path, results: Sequence[BetaResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "beta",
                "stopband64_db",
                "stopband32_db",
                "stopband_loss_db",
                "passband_ripple64_db",
                "passband_ripple32_db",
                "max_response_diff_db",
                "coeff_snr_db",
                "coeff_zero_ratio",
                "effective_bits",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.beta,
                    r.stopband64_db,
                    r.stopband32_db,
                    r.stopband_loss_db,
                    r.passband_ripple64_db,
                    r.passband_ripple32_db,
                    r.max_response_diff_db,
                    r.coeff_snr_db,
                    r.zero_ratio,
                    r.effective_bits,
                ]
            )
    print(f"CSV出力: {path}")


def plot_results(path: Path, results: Sequence[BetaResult]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    betas = [r.beta for r in results]
    stop64 = [abs(r.stopband64_db) for r in results]
    stop32 = [abs(r.stopband32_db) for r in results]
    loss = [r.stopband_loss_db for r in results]

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)

    ax[0].plot(betas, stop64, label="Float64 stopband", marker="o")
    ax[0].plot(betas, stop32, label="Float32 stopband", marker="o")
    ax[0].set_ylabel("Stopband attenuation [dB]")
    ax[0].grid(True, which="both", ls="--", alpha=0.3)
    ax[0].legend()

    ax[1].plot(betas, loss, label="Loss (64→32)", marker="o", color="tab:red")
    ax[1].set_xlabel("Kaiser β")
    ax[1].set_ylabel("Attenuation loss [dB]")
    ax[1].grid(True, which="both", ls="--", alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"プロット出力: {path}")


def main() -> None:
    args = parse_args()
    stopband_start = (
        args.stopband_start if args.stopband_start is not None else args.input_rate / 2
    )
    cfg = SweepConfig(
        input_rate=args.input_rate,
        upsample_ratio=args.upsample_ratio,
        passband_end=args.passband_end,
        stopband_start=stopband_start,
        taps=args.taps,
        phase=args.phase,
        minimum_phase_method=args.minimum_phase_method,
        freq_points=args.freq_points,
        target_dc_gain=args.target_dc_gain
        if args.target_dc_gain is not None
        else float(args.upsample_ratio),
    )

    betas = build_beta_list(args)
    results: list[BetaResult] = []
    for beta in betas:
        result = evaluate_beta(beta, cfg)
        results.append(result)

    render_table(results, args.target_stopband_db)
    plateau = estimate_plateau_beta(results, args.plateau_threshold_db)
    if plateau is not None:
        print(
            f"改善頭打ち推定: β≈{plateau:.2f} "
            f"(隣接βでの実効阻止帯域改善が {args.plateau_threshold_db:.2f} dB 未満)"
        )

    if args.output_csv:
        save_csv(args.output_csv, results)
    if args.plot:
        plot_results(args.plot_path, results)


if __name__ == "__main__":
    main()
