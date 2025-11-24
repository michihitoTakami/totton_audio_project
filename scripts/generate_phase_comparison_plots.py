#!/usr/bin/env python3
"""
位相タイプ比較プロット生成スクリプト

最小位相（minimum phase）と線形位相（linear phase）フィルタの
特性を比較するプロットを生成する。

出力:
  - plots/analysis/phase_comparison_impulse.png
  - plots/analysis/phase_comparison_frequency.png
  - plots/analysis/phase_comparison_phase.png
  - plots/analysis/phase_comparison_preringing.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_filter import FilterConfig, FilterDesigner, PhaseType  # noqa: E402


def generate_comparison_plots(
    n_taps: int = 4096, output_dir: Path | None = None
) -> None:
    """位相タイプ比較プロットを生成する

    Args:
        n_taps: フィルタのタップ数（高速処理のためデフォルトは小さめ）
        output_dir: 出力ディレクトリ
    """
    if output_dir is None:
        output_dir = SCRIPTS_DIR.parent / "plots" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"位相タイプ比較プロット生成中... (タップ数: {n_taps:,})")

    # 両位相タイプのフィルタを生成
    config_min = FilterConfig(
        n_taps=n_taps,
        input_rate=44100,
        upsample_ratio=16,
        kaiser_beta=14,  # 小さいタップ用
        phase_type=PhaseType.MINIMUM,
    )
    config_lin = FilterConfig(
        n_taps=n_taps,
        input_rate=44100,
        upsample_ratio=16,
        kaiser_beta=14,
        phase_type=PhaseType.LINEAR,
    )

    designer_min = FilterDesigner(config_min)
    designer_lin = FilterDesigner(config_lin)

    h_min, _ = designer_min.design()
    h_lin, _ = designer_lin.design()

    fs = config_min.output_rate

    # 1. インパルス応答比較
    plot_impulse_comparison(h_min, h_lin, fs, output_dir)

    # 2. 周波数応答比較
    plot_frequency_comparison(h_min, h_lin, fs, output_dir)

    # 3. 位相特性・群遅延比較
    plot_phase_comparison(h_min, h_lin, fs, output_dir)

    # 4. プリリンギング比較
    plot_preringing_comparison(h_min, h_lin, fs, output_dir)

    print(f"プロット生成完了: {output_dir}")


def plot_impulse_comparison(
    h_min: np.ndarray, h_lin: np.ndarray, fs: int, output_dir: Path
) -> None:
    """インパルス応答比較プロット"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    n_samples = min(len(h_min), len(h_lin))
    t_ms = np.arange(n_samples) / fs * 1000  # ms

    # 最小位相
    ax = axes[0]
    ax.plot(t_ms, h_min[:n_samples], "b-", linewidth=0.5, label="Minimum Phase")
    ax.set_title("Minimum Phase - Impulse Response")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_ms[-1])

    # ピーク位置をマーク
    peak_idx = np.argmax(np.abs(h_min))
    ax.axvline(
        t_ms[peak_idx],
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Peak @ {t_ms[peak_idx]:.3f}ms",
    )
    ax.legend()

    # 線形位相
    ax = axes[1]
    ax.plot(t_ms, h_lin[:n_samples], "g-", linewidth=0.5, label="Linear Phase")
    ax.set_title("Linear Phase - Impulse Response")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_ms[-1])

    # ピーク位置をマーク
    peak_idx = np.argmax(np.abs(h_lin))
    ax.axvline(
        t_ms[peak_idx],
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Peak @ {t_ms[peak_idx]:.3f}ms",
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "phase_comparison_impulse.png", dpi=150)
    plt.close()
    print("  - phase_comparison_impulse.png")


def plot_frequency_comparison(
    h_min: np.ndarray, h_lin: np.ndarray, fs: int, output_dir: Path
) -> None:
    """周波数応答比較プロット"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 周波数応答を計算
    w_min, H_min = signal.freqz(h_min, worN=8192, fs=fs)
    w_lin, H_lin = signal.freqz(h_lin, worN=8192, fs=fs)

    H_min_db = 20 * np.log10(np.abs(H_min) + 1e-12)
    H_lin_db = 20 * np.log10(np.abs(H_lin) + 1e-12)

    ax.plot(w_min / 1000, H_min_db, "b-", linewidth=1, label="Minimum Phase", alpha=0.8)
    ax.plot(w_lin / 1000, H_lin_db, "g--", linewidth=1, label="Linear Phase", alpha=0.8)

    ax.set_title("Frequency Response Comparison")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_xlim(0, 25)  # 0-25kHz
    ax.set_ylim(-120, 5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # パスバンド/ストップバンド領域を示す
    ax.axvline(
        20, color="orange", linestyle=":", alpha=0.5, label="Passband End (20kHz)"
    )
    ax.axvline(
        22.05, color="red", linestyle=":", alpha=0.5, label="Stopband Start (22.05kHz)"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "phase_comparison_frequency.png", dpi=150)
    plt.close()
    print("  - phase_comparison_frequency.png")


def plot_phase_comparison(
    h_min: np.ndarray, h_lin: np.ndarray, fs: int, output_dir: Path
) -> None:
    """位相特性・群遅延比較プロット"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 周波数応答を計算
    w_min, H_min = signal.freqz(h_min, worN=8192, fs=fs)
    w_lin, H_lin = signal.freqz(h_lin, worN=8192, fs=fs)

    # 位相（パスバンド内のみ）
    passband_mask = w_min <= 20000
    phase_min = np.unwrap(np.angle(H_min))
    phase_lin = np.unwrap(np.angle(H_lin))

    ax = axes[0]
    ax.plot(
        w_min[passband_mask] / 1000,
        phase_min[passband_mask],
        "b-",
        linewidth=1,
        label="Minimum Phase",
    )
    ax.plot(
        w_lin[passband_mask] / 1000,
        phase_lin[passband_mask],
        "g--",
        linewidth=1,
        label="Linear Phase",
    )
    ax.set_title("Phase Response (Passband)")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Phase [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 群遅延
    _, gd_min = signal.group_delay((h_min, 1), w=w_min[: len(w_min) // 4], fs=fs)
    _, gd_lin = signal.group_delay((h_lin, 1), w=w_lin[: len(w_lin) // 4], fs=fs)

    ax = axes[1]
    gd_min_len = int(len(gd_min))
    gd_lin_len = int(len(gd_lin))
    ax.plot(
        w_min[:gd_min_len] / 1000,
        gd_min / fs * 1000,
        "b-",
        linewidth=1,
        label="Minimum Phase",
    )
    ax.plot(
        w_lin[:gd_lin_len] / 1000,
        gd_lin / fs * 1000,
        "g--",
        linewidth=1,
        label="Linear Phase",
    )
    ax.set_title("Group Delay")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Group Delay [ms]")
    ax.set_xlim(0, 20)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "phase_comparison_phase.png", dpi=150)
    plt.close()
    print("  - phase_comparison_phase.png")


def plot_preringing_comparison(
    h_min: np.ndarray, h_lin: np.ndarray, fs: int, output_dir: Path
) -> None:
    """プリリンギング比較プロット（拡大表示）"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 最小位相はピーク付近、線形位相は中央付近を拡大
    peak_min = np.argmax(np.abs(h_min))
    peak_lin = np.argmax(np.abs(h_lin))

    # 前後のサンプル数
    n_before = 200
    n_after = 500

    # 最小位相
    ax = axes[0]
    start_min: int = max(0, int(peak_min) - n_before)
    end_min: int = min(len(h_min), int(peak_min) + n_after)
    t_min = (np.arange(start_min, end_min) - peak_min) / fs * 1000  # ms from peak

    ax.plot(t_min, h_min[start_min:end_min], "b-", linewidth=1)
    ax.axvline(0, color="r", linestyle="--", alpha=0.7, label="Peak")
    ax.set_title("Minimum Phase - Pre-ringing Detail (No pre-ringing)")
    ax.set_xlabel("Time from Peak [ms]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # プリリンギングエネルギーを計算
    pre_energy_min = np.sum(h_min[:peak_min] ** 2)
    total_energy_min = np.sum(h_min**2)
    ax.text(
        0.02,
        0.98,
        f"Pre-ringing energy: {pre_energy_min/total_energy_min*100:.4f}%",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 線形位相
    ax = axes[1]
    start_lin: int = max(0, int(peak_lin) - n_before)
    end_lin: int = min(len(h_lin), int(peak_lin) + n_after)
    t_lin = (np.arange(start_lin, end_lin) - peak_lin) / fs * 1000  # ms from peak

    ax.plot(t_lin, h_lin[start_lin:end_lin], "g-", linewidth=1)
    ax.axvline(0, color="r", linestyle="--", alpha=0.7, label="Peak")
    ax.set_title(
        "Linear Phase - Pre-ringing Detail (Symmetric, significant pre-ringing)"
    )
    ax.set_xlabel("Time from Peak [ms]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # プリリンギングエネルギーを計算
    pre_energy_lin = np.sum(h_lin[:peak_lin] ** 2)
    total_energy_lin = np.sum(h_lin**2)
    ax.text(
        0.02,
        0.98,
        f"Pre-ringing energy: {pre_energy_lin/total_energy_lin*100:.2f}%",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "phase_comparison_preringing.png", dpi=150)
    plt.close()
    print("  - phase_comparison_preringing.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="位相タイプ比較プロット生成")
    parser.add_argument(
        "--taps", type=int, default=4096, help="タップ数 (default: 4096)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="出力ディレクトリ"
    )
    args = parser.parse_args()

    generate_comparison_plots(n_taps=args.taps, output_dir=args.output_dir)
