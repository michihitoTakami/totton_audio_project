#!/usr/bin/env python3
"""
GPU Audio Upsampler - Multi-Rate Filter Coefficient Generation

最小位相FIRフィルタを生成し、検証する。

サポートするアップサンプリング比率:
- 16x: 44.1kHz → 705.6kHz, 48kHz → 768kHz
- 8x:  88.2kHz → 705.6kHz, 96kHz → 768kHz
- 4x:  176.4kHz → 705.6kHz, 192kHz → 768kHz
- 2x:  352.8kHz → 705.6kHz, 384kHz → 768kHz

仕様:
- タップ数: 2,000,000 (2M) デフォルト
- 位相特性: 最小位相（プリリンギング排除）
- 通過帯域: 0-20,000 Hz
- 阻止帯域: 入力Nyquist周波数以降
- 阻止帯域減衰: -197 dB以下
- 窓関数: Kaiser (β ≈ 55)

注意:
- タップ数はアップサンプリング比率の倍数であること
- クリッピング防止のため係数は正規化される
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# デフォルト定数（必要に応じてコマンドラインで上書き）
N_TAPS = 2_000_000  # 2M taps (200万タップ)
SAMPLE_RATE_INPUT = 44100  # 入力サンプルレート (Hz)
UPSAMPLE_RATIO = 16  # アップサンプリング倍率
SAMPLE_RATE_OUTPUT = SAMPLE_RATE_INPUT * UPSAMPLE_RATIO  # 出力サンプルレート

# フィルタ設計パラメータ（デフォルト）
PASSBAND_END = 20000  # 通過帯域終端 (Hz) - 可聴帯域
STOPBAND_START = 22050  # 阻止帯域開始 (Hz) - 入力Nyquist周波数
STOPBAND_ATTENUATION_DB = 197  # 阻止帯域減衰量 (dB)
# Kaiser βパラメータ: A(dB)の減衰量に対して β ≈ 0.1102*(A-8.7)
# 2Mタップでは55を使用してより高い減衰を目指す
KAISER_BETA = 55  # Kaiser窓のβパラメータ
OUTPUT_PREFIX = None

# マルチレート設定
# 44.1kHz系と48kHz系、それぞれ16x/8x/4x/2xの組み合わせ
MULTI_RATE_CONFIGS = {
    # 44.1kHz family -> 705.6kHz output
    "44k_16x": {"input_rate": 44100, "ratio": 16, "stopband": 22050},
    "44k_8x": {"input_rate": 88200, "ratio": 8, "stopband": 44100},
    "44k_4x": {"input_rate": 176400, "ratio": 4, "stopband": 88200},
    "44k_2x": {"input_rate": 352800, "ratio": 2, "stopband": 176400},
    # 48kHz family -> 768kHz output
    "48k_16x": {"input_rate": 48000, "ratio": 16, "stopband": 24000},
    "48k_8x": {"input_rate": 96000, "ratio": 8, "stopband": 48000},
    "48k_4x": {"input_rate": 192000, "ratio": 4, "stopband": 96000},
    "48k_2x": {"input_rate": 384000, "ratio": 2, "stopband": 192000},
}


def design_linear_phase_filter():
    """
    線形位相FIRフィルタを設計する。

    Returns:
        np.ndarray: 線形位相FIRフィルタ係数
    """
    print("線形位相FIRフィルタ設計中...")
    print(f"  タップ数: {N_TAPS}")
    print(f"  出力サンプルレート: {SAMPLE_RATE_OUTPUT} Hz")
    print(f"  通過帯域: 0-{PASSBAND_END} Hz")
    print(f"  阻止帯域: {STOPBAND_START}+ Hz")

    # カットオフ周波数（通過帯域と阻止帯域の中間）
    cutoff_freq = (PASSBAND_END + STOPBAND_START) / 2

    # 正規化カットオフ周波数（ナイキスト周波数に対する比率）
    nyquist = SAMPLE_RATE_OUTPUT / 2
    normalized_cutoff = cutoff_freq / nyquist

    print(f"  カットオフ周波数: {cutoff_freq} Hz (正規化: {normalized_cutoff:.6f})")
    print(f"  Kaiser β: {KAISER_BETA}")

    # Kaiser窓を使用した線形位相ローパスフィルタ
    # numtaps: 奇数にする（タイプIフィルタ、対称性のため）
    numtaps = N_TAPS if N_TAPS % 2 == 1 else N_TAPS + 1

    h_linear = signal.firwin(
        numtaps=numtaps,
        cutoff=normalized_cutoff,
        window=("kaiser", KAISER_BETA),
        fs=1.0,  # 正規化周波数を使用
        scale=True,
    )

    print(f"  実際のタップ数: {len(h_linear)}")
    return h_linear


def convert_to_minimum_phase(h_linear):
    """
    線形位相フィルタを最小位相フィルタに変換する。

    scipy.signal.minimum_phaseを使用して、
    線形位相フィルタの振幅特性を保持したまま最小位相に変換。

    Args:
        h_linear: 線形位相フィルタ係数

    Returns:
        np.ndarray: 最小位相FIRフィルタ係数
    """
    print("\n最小位相変換中...")

    # scipy.signal.minimum_phaseで変換
    # method='homomorphic': ホモモルフィック法（振幅特性保持）
    # n_fft: FFTサイズ（時間エイリアシング防止のため、タップ数の8倍に設定）
    # ホモモルフィック処理では周波数領域で対数操作を行うため、
    # 十分なFFTサイズを確保しないと因果性が壊れプリリンギングが残留する
    # 例: 2Mタップの場合、8倍=16,777,216（2^24）を使用
    # これは非常に大きなFFTなので、処理に時間がかかる（数分～数十分）
    n_fft = 2 ** int(np.ceil(np.log2(len(h_linear) * 8)))
    print(
        f"  警告: FFTサイズ {n_fft:,} は非常に大きいため、処理に時間がかかります（数分～数十分）"
    )

    h_min_phase = signal.minimum_phase(h_linear, method="homomorphic", n_fft=n_fft)

    # 元のタップ数に合わせる
    if len(h_min_phase) > N_TAPS:
        h_min_phase = h_min_phase[:N_TAPS]
    elif len(h_min_phase) < N_TAPS:
        # ゼロパディング
        h_min_phase = np.pad(h_min_phase, (0, N_TAPS - len(h_min_phase)))

    print(f"  最小位相係数タップ数: {len(h_min_phase)}")
    print(f"  FFTサイズ: {n_fft}")
    return h_min_phase


def validate_specifications(h):
    """
    フィルタ係数が仕様を満たしているか検証する。

    Args:
        h: フィルタ係数

    Returns:
        dict: 検証結果
    """
    print("\n仕様検証中...")

    # 周波数応答を計算
    w, H = signal.freqz(h, worN=16384, fs=SAMPLE_RATE_OUTPUT)
    H_db = 20 * np.log10(np.abs(H) + 1e-12)

    # 通過帯域のリップル計算
    passband_mask = w <= PASSBAND_END
    passband_db = H_db[passband_mask]
    passband_ripple_db = np.max(passband_db) - np.min(passband_db)

    # 阻止帯域の減衰量計算
    stopband_mask = w >= STOPBAND_START
    stopband_attenuation = np.min(H_db[stopband_mask])

    # 最小位相特性の検証
    # 最小位相フィルタの特徴:
    # 1. ピーク位置が先頭付近（タップ数の1%以内）
    # 2. エネルギーが因果的に分布（前半に集中）
    peak_idx = np.argmax(np.abs(h))

    # 前半50%と後半50%のエネルギー比較
    mid_point = len(h) // 2
    energy_first_half = np.sum(h[:mid_point] ** 2)
    energy_second_half = np.sum(h[mid_point:] ** 2)
    energy_ratio = energy_first_half / (energy_second_half + 1e-12)

    # ピーク位置が先頭から1%以内 かつ エネルギーが前半に集中（比率>10）
    peak_threshold = int(len(h) * 0.01)
    is_peak_at_front = peak_idx < peak_threshold
    is_energy_causal = energy_ratio > 10

    results = {
        "passband_ripple_db": float(passband_ripple_db),
        "stopband_attenuation_db": float(abs(stopband_attenuation)),
        "peak_position": int(peak_idx),
        "peak_threshold_samples": int(peak_threshold),
        "energy_ratio_first_to_second_half": float(energy_ratio),
        "meets_stopband_spec": bool(
            abs(stopband_attenuation) >= STOPBAND_ATTENUATION_DB
        ),
        "is_minimum_phase": bool(is_peak_at_front and is_energy_causal),
    }

    print(f"  通過帯域リップル: {passband_ripple_db:.3f} dB")
    print(
        f"  阻止帯域減衰: {abs(stopband_attenuation):.1f} dB (目標: {STOPBAND_ATTENUATION_DB} dB)"
    )
    print(
        f"  阻止帯域スペック: {'✓ 合格' if results['meets_stopband_spec'] else '✗ 不合格'}"
    )
    print(
        f"  ピーク位置: サンプル {peak_idx} (先頭1%={peak_threshold}サンプル以内: {'✓' if is_peak_at_front else '✗'})"
    )
    print(f"  エネルギー比(前半/後半): {energy_ratio:.1f} (目標: >10)")
    print(f"  最小位相特性: {'✓ 確認' if results['is_minimum_phase'] else '✗ 未確認'}")

    return results


def plot_responses(
    h_linear, h_min_phase, output_dir="plots/analysis", filter_name=None
):
    """
    フィルタ特性をプロットする。

    Args:
        h_linear: 線形位相フィルタ係数
        h_min_phase: 最小位相フィルタ係数
        output_dir: プロット出力ディレクトリ
        filter_name: フィルタ名（ファイル名プレフィックスに使用）
    """
    print(f"\nプロット生成中... ({output_dir})")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ファイル名プレフィックス（filter_nameがあれば使用）
    prefix = f"{filter_name}_" if filter_name else ""

    # フォント設定（日本語対応）
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1. インパルス応答の比較
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 線形位相インパルス応答（中央部分のみ表示）
    center = len(h_linear) // 2
    display_range = min(2000, center)  # 配列サイズに合わせて調整
    t_linear = np.arange(-display_range, display_range)
    h_linear_center = h_linear[center - display_range : center + display_range]

    axes[0].plot(t_linear, h_linear_center, linewidth=0.5)
    axes[0].set_title("Linear Phase Impulse Response (Center Region)", fontsize=12)
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0, color="r", linestyle="--", alpha=0.5, label="Center")
    axes[0].legend()

    # 最小位相インパルス応答（先頭部分のみ表示）
    display_range_min = min(4000, len(h_min_phase))  # 配列サイズに合わせて調整
    t_min = np.arange(display_range_min)
    h_min_display = h_min_phase[:display_range_min]

    axes[1].plot(t_min, h_min_display, linewidth=0.5, color="orange")
    axes[1].set_title(
        "Minimum Phase Impulse Response (Front Region - No Pre-ringing)", fontsize=12
    )
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(0, color="r", linestyle="--", alpha=0.5, label="t=0")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path / f"{prefix}impulse_response.png", dpi=150)
    print(f"  保存: {prefix}impulse_response.png")
    plt.close()

    # 2. 周波数応答（振幅）
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    w_linear, H_linear = signal.freqz(h_linear, worN=16384, fs=SAMPLE_RATE_OUTPUT)
    w_min, H_min = signal.freqz(h_min_phase, worN=16384, fs=SAMPLE_RATE_OUTPUT)

    H_linear_db = 20 * np.log10(np.abs(H_linear) + 1e-12)
    H_min_db = 20 * np.log10(np.abs(H_min) + 1e-12)

    # 全体表示
    axes[0].plot(
        w_linear / 1000, H_linear_db, label="Linear Phase", linewidth=1, alpha=0.7
    )
    axes[0].plot(w_min / 1000, H_min_db, label="Minimum Phase", linewidth=1, alpha=0.7)
    axes[0].set_title("Magnitude Response (Full Range)", fontsize=12)
    axes[0].set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_ylim(-200, 5)
    axes[0].axhline(-180, color="r", linestyle="--", alpha=0.5, label="-180dB Target")
    axes[0].axvline(
        PASSBAND_END / 1000, color="g", linestyle="--", alpha=0.5, label="Passband End"
    )
    axes[0].axvline(
        STOPBAND_START / 1000,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label="Stopband Start",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 通過帯域詳細
    passband_mask = w_min <= PASSBAND_END * 1.1
    axes[1].plot(
        w_min[passband_mask] / 1000,
        H_min_db[passband_mask],
        linewidth=1,
        color="orange",
    )
    axes[1].set_title("Magnitude Response (Passband Detail)", fontsize=12)
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].axvline(
        PASSBAND_END / 1000, color="g", linestyle="--", alpha=0.5, label="Passband End"
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path / f"{prefix}frequency_response.png", dpi=150)
    print(f"  保存: {prefix}frequency_response.png")
    plt.close()

    # 3. 位相応答
    fig, ax = plt.subplots(figsize=(14, 6))

    _, H_linear_complex = signal.freqz(h_linear, worN=8192, fs=SAMPLE_RATE_OUTPUT)
    _, H_min_complex = signal.freqz(h_min_phase, worN=8192, fs=SAMPLE_RATE_OUTPUT)

    phase_linear = np.unwrap(np.angle(H_linear_complex))
    phase_min = np.unwrap(np.angle(H_min_complex))

    w_phase, _ = signal.freqz(h_linear, worN=8192, fs=SAMPLE_RATE_OUTPUT)

    ax.plot(w_phase / 1000, phase_linear, label="Linear Phase", linewidth=1, alpha=0.7)
    ax.plot(w_phase / 1000, phase_min, label="Minimum Phase", linewidth=1, alpha=0.7)
    ax.set_title("Phase Response", fontsize=12)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Phase (radians)")
    ax.axvline(
        PASSBAND_END / 1000, color="g", linestyle="--", alpha=0.5, label="Passband End"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / f"{prefix}phase_response.png", dpi=150)
    print(f"  保存: {prefix}phase_response.png")
    plt.close()


def export_coefficients(h, metadata, output_dir="data/coefficients", skip_header=False):
    """
    フィルタ係数をエクスポートする。

    Args:
        h: フィルタ係数
        metadata: メタデータ辞書
        output_dir: 出力ディレクトリ
        skip_header: Trueの場合、ヘッダファイル生成をスキップ（batch生成時用）
    """
    print(f"\n係数エクスポート中... ({output_dir})")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    taps_label = f"{N_TAPS // 1_000_000}m" if N_TAPS % 1_000_000 == 0 else f"{N_TAPS}"
    # ファイル名に比率を含める: filter_44k_16x_2m_min_phase.bin
    family = "44k" if SAMPLE_RATE_INPUT % 44100 == 0 else "48k"
    base_name = (
        OUTPUT_PREFIX or f"filter_{family}_{UPSAMPLE_RATIO}x_{taps_label}_min_phase"
    )

    # 1. バイナリ形式（float32）
    h_float32 = h.astype(np.float32)
    binary_path = output_path / f"{base_name}.bin"
    h_float32.tofile(binary_path)
    file_size_mb = binary_path.stat().st_size / (1024 * 1024)
    print(f"  保存: {binary_path} ({file_size_mb:.2f} MB)")

    # 2. C++ヘッダファイル（単一フィルタ生成時のみ）
    if not skip_header:
        header_path = output_path / "filter_coefficients.h"
        with open(header_path, "w") as f:
            f.write("// Auto-generated filter coefficients\n")
            f.write("// GPU Audio Upsampler - Phase 1\n")
            f.write(f"// Generated: {metadata['generation_date']}\n\n")
            f.write("#ifndef FILTER_COEFFICIENTS_H\n")
            f.write("#define FILTER_COEFFICIENTS_H\n\n")
            f.write("#include <cstddef>\n\n")
            f.write(f"constexpr size_t FILTER_TAPS = {len(h)};\n")
            f.write(
                f"constexpr int SAMPLE_RATE_INPUT = {metadata['sample_rate_input']};\n"
            )
            f.write(
                f"constexpr int SAMPLE_RATE_OUTPUT = {metadata['sample_rate_output']};\n"
            )
            f.write(f"constexpr int UPSAMPLE_RATIO = {metadata['upsample_ratio']};\n\n")
            f.write("// Filter coefficients are stored in external .bin files.\n")
            f.write(f"// Default binary: {base_name}.bin\n\n")
            f.write("#endif // FILTER_COEFFICIENTS_H\n")
        print(f"  保存: {header_path}")

    # 3. メタデータJSON
    metadata_path = output_path / f"{base_name}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  保存: {metadata_path}")

    return base_name  # 生成したファイル名を返す


def generate_multi_rate_header(
    filter_infos, output_dir="data/coefficients", taps=2_000_000
):
    """
    全フィルタ情報をまとめたC++ヘッダファイルを生成する。

    Args:
        filter_infos: 各フィルタの情報リスト [(name, base_name, config), ...]
        output_dir: 出力ディレクトリ
        taps: タップ数
    """
    output_path = Path(output_dir)
    header_path = output_path / "filter_coefficients.h"

    with open(header_path, "w") as f:
        f.write("// Auto-generated multi-rate filter coefficients\n")
        f.write("// GPU Audio Upsampler - Multi-Rate Support\n")
        f.write(f"// Generated: {datetime.now().isoformat()}\n\n")
        f.write("#ifndef FILTER_COEFFICIENTS_H\n")
        f.write("#define FILTER_COEFFICIENTS_H\n\n")
        f.write("#include <cstddef>\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"constexpr size_t FILTER_TAPS = {taps};\n\n")
        f.write("// Multi-rate filter configurations\n")
        f.write("struct FilterConfig {\n")
        f.write("    const char* name;\n")
        f.write("    const char* filename;\n")
        f.write("    int32_t input_rate;\n")
        f.write("    int32_t output_rate;\n")
        f.write("    int32_t ratio;\n")
        f.write("};\n\n")
        f.write(f"constexpr size_t FILTER_COUNT = {len(filter_infos)};\n\n")
        f.write("constexpr FilterConfig FILTER_CONFIGS[FILTER_COUNT] = {\n")
        for name, base_name, cfg in filter_infos:
            output_rate = cfg["input_rate"] * cfg["ratio"]
            f.write(
                f'    {{"{name}", "{base_name}.bin", '
                f'{cfg["input_rate"]}, {output_rate}, {cfg["ratio"]}}},\n'
            )
        f.write("};\n\n")
        f.write("#endif // FILTER_COEFFICIENTS_H\n")

    print(f"\n✓ マルチレートヘッダファイル生成: {header_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate minimum-phase FIR filter coefficients.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single filter (44.1kHz, 16x)
  %(prog)s --input-rate 44100 --upsample-ratio 16

  # Generate all 8 filter configurations
  %(prog)s --generate-all

  # Generate only 44.1kHz family (4 filters)
  %(prog)s --generate-all --family 44k

  # Generate only 48kHz family (4 filters)
  %(prog)s --generate-all --family 48k
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
        "--upsample-ratio", type=int, default=16, help="Upsampling ratio. Default: 16"
    )
    parser.add_argument(
        "--taps",
        type=int,
        default=2_000_000,
        help="Number of filter taps. Default: 2000000 (2M)",
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
        default=197,
        help="Target stopband attenuation (dB). Default: 197",
    )
    parser.add_argument(
        "--kaiser-beta",
        type=float,
        default=55.0,
        help="Kaiser window beta. Default: 55",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file basename (without extension). Default: auto. "
        "NOTE: Ignored when --generate-all is used.",
    )
    return parser.parse_args()


def validate_tap_count(taps, upsample_ratio):
    """
    タップ数がアップサンプリング比率の倍数であることを確認する。

    Args:
        taps: タップ数
        upsample_ratio: アップサンプリング比率

    Raises:
        ValueError: タップ数が倍数でない場合
    """
    if taps % upsample_ratio != 0:
        raise ValueError(
            f"タップ数 {taps:,} はアップサンプリング比率 {upsample_ratio} の倍数である必要があります。"
            f"\n  推奨: {(taps // upsample_ratio) * upsample_ratio:,} または "
            f"{((taps // upsample_ratio) + 1) * upsample_ratio:,}"
        )
    print(f"✓ タップ数 {taps:,} は {upsample_ratio} の倍数です")


def normalize_coefficients(h):
    """
    フィルタ係数を正規化してクリッピングを防止する。

    DCゲインを1.0に正規化し、最大振幅をチェックする。

    Args:
        h: フィルタ係数

    Returns:
        正規化されたフィルタ係数と正規化情報

    Raises:
        ValueError: DCゲインが0に近すぎる場合
    """
    # DCゲイン（係数の総和）を計算
    dc_gain = np.sum(h)

    # ゼロ除算防止
    if abs(dc_gain) < 1e-12:
        raise ValueError("DCゲインが0に近すぎます。フィルター係数が不正です。")

    # 正規化（DCゲイン = 1.0）
    h_normalized = h / dc_gain

    # 最大振幅チェック
    max_amplitude = np.max(np.abs(h_normalized))

    info = {
        "original_dc_gain": float(dc_gain),
        "normalized_dc_gain": float(np.sum(h_normalized)),
        "max_coefficient_amplitude": float(max_amplitude),
        "normalization_applied": True,
    }

    print("\n係数正規化:")
    print(f"  元のDCゲイン: {dc_gain:.6f}")
    print(f"  正規化後DCゲイン: {np.sum(h_normalized):.6f}")
    print(f"  最大係数振幅: {max_amplitude:.6f}")

    return h_normalized, info


def generate_single_filter(args, filter_name=None, skip_header=False):
    """
    単一フィルタを生成する。

    Args:
        args: コマンドライン引数
        filter_name: フィルタ名（プロットファイル名に使用、Noneなら使用しない）
        skip_header: Trueの場合、ヘッダファイル生成をスキップ（batch生成時用）

    Returns:
        str: 生成したファイルのbase_name
    """
    global SAMPLE_RATE_INPUT, UPSAMPLE_RATIO, SAMPLE_RATE_OUTPUT
    global PASSBAND_END, STOPBAND_START, STOPBAND_ATTENUATION_DB, KAISER_BETA
    global N_TAPS, OUTPUT_PREFIX

    SAMPLE_RATE_INPUT = args.input_rate
    UPSAMPLE_RATIO = args.upsample_ratio
    SAMPLE_RATE_OUTPUT = SAMPLE_RATE_INPUT * UPSAMPLE_RATIO
    PASSBAND_END = args.passband_end
    # stopband_startが指定されていない場合は入力Nyquist周波数を使用
    STOPBAND_START = (
        args.stopband_start if args.stopband_start else (SAMPLE_RATE_INPUT // 2)
    )
    STOPBAND_ATTENUATION_DB = args.stopband_attenuation
    KAISER_BETA = args.kaiser_beta
    N_TAPS = args.taps
    OUTPUT_PREFIX = args.output_prefix

    # 0. タップ数の検証
    validate_tap_count(N_TAPS, UPSAMPLE_RATIO)

    # 1. 線形位相フィルタ設計
    h_linear = design_linear_phase_filter()

    # 2. 最小位相変換
    h_min_phase = convert_to_minimum_phase(h_linear)

    # 3. 係数正規化（クリッピング防止）
    h_min_phase, normalization_info = normalize_coefficients(h_min_phase)

    # 4. 仕様検証
    validation_results = validate_specifications(h_min_phase)
    validation_results["normalization"] = normalization_info

    # 5. プロット生成（filter_nameがあればファイル名に含める）
    plot_responses(h_linear, h_min_phase, filter_name=filter_name)

    # 6. メタデータ作成
    metadata = {
        "generation_date": datetime.now().isoformat(),
        "n_taps": N_TAPS,
        "sample_rate_input": SAMPLE_RATE_INPUT,
        "sample_rate_output": SAMPLE_RATE_OUTPUT,
        "upsample_ratio": UPSAMPLE_RATIO,
        "passband_end_hz": PASSBAND_END,
        "stopband_start_hz": STOPBAND_START,
        "target_stopband_attenuation_db": STOPBAND_ATTENUATION_DB,
        "kaiser_beta": KAISER_BETA,
        "validation_results": validation_results,
    }

    # 7. 係数エクスポート
    taps_label = f"{N_TAPS // 1_000_000}m" if N_TAPS % 1_000_000 == 0 else f"{N_TAPS}"
    family = "44k" if SAMPLE_RATE_INPUT % 44100 == 0 else "48k"
    base_name = (
        OUTPUT_PREFIX or f"filter_{family}_{UPSAMPLE_RATIO}x_{taps_label}_min_phase"
    )
    metadata["output_basename"] = base_name
    export_coefficients(h_min_phase, metadata, skip_header=skip_header)

    # 8. 最終レポート
    print("\n" + "=" * 70)
    print(f"Phase 1 完了 - {N_TAPS:,}タップフィルタ")
    print("=" * 70)
    print(f"✓ {N_TAPS:,}タップ最小位相FIRフィルタ生成完了")
    print(f"✓ 阻止帯域減衰: {validation_results['stopband_attenuation_db']:.1f} dB")
    print(
        f"  {'合格' if validation_results['meets_stopband_spec'] else '不合格'} (目標: {STOPBAND_ATTENUATION_DB} dB以上)"
    )
    print(
        f"✓ 最小位相特性: {'確認済み' if validation_results['is_minimum_phase'] else '要確認'}"
    )
    print(f"✓ 係数正規化: DCゲイン={normalization_info['normalized_dc_gain']:.6f}")
    print(f"✓ 係数ファイル: data/coefficients/{base_name}.bin")
    print("✓ 検証プロット: plots/analysis/")
    print("=" * 70)

    return base_name


def generate_all_filters(args):
    """
    全8種類のフィルタを一括生成する。

    44.1kHz系: 16x, 8x, 4x, 2x
    48kHz系: 16x, 8x, 4x, 2x

    Note:
        --output-prefixは--generate-all時は無視されます（各フィルタは自動命名）。
    """
    import copy

    # 対象ファミリーを決定
    if args.family == "44k":
        configs = {k: v for k, v in MULTI_RATE_CONFIGS.items() if k.startswith("44k")}
    elif args.family == "48k":
        configs = {k: v for k, v in MULTI_RATE_CONFIGS.items() if k.startswith("48k")}
    else:
        configs = MULTI_RATE_CONFIGS

    total = len(configs)
    print("=" * 70)
    print(f"Multi-Rate Filter Generation - {total} filters")
    print("=" * 70)
    print("\nTarget configurations:")
    for name, cfg in configs.items():
        output_rate = cfg["input_rate"] * cfg["ratio"]
        print(f"  {name}: {cfg['input_rate']}Hz × {cfg['ratio']}x → {output_rate}Hz")

    if args.output_prefix:
        print("\n注意: --output-prefix は --generate-all 時は無視されます")
    print()

    results = []  # [(name, status, base_name, config), ...]
    filter_infos = []  # 成功したフィルタの情報（ヘッダ生成用）

    for i, (name, cfg) in enumerate(configs.items(), 1):
        print("\n" + "=" * 70)
        print(f"[{i}/{total}] Generating {name}...")
        print("=" * 70)

        # 引数をコピーして設定を上書き
        filter_args = copy.copy(args)
        filter_args.input_rate = cfg["input_rate"]
        filter_args.upsample_ratio = cfg["ratio"]
        filter_args.stopband_start = cfg["stopband"]
        filter_args.output_prefix = None  # 自動生成

        try:
            # skip_header=True: 個別のヘッダ生成をスキップ
            # filter_name=name: プロットファイル名にフィルタ名を含める
            base_name = generate_single_filter(
                filter_args, filter_name=name, skip_header=True
            )
            results.append((name, "✓ Success"))
            filter_infos.append((name, base_name, cfg))
        except Exception as e:
            results.append((name, f"✗ Failed: {e}"))
            print(f"ERROR: {e}")

    # 全フィルタ情報をまとめたヘッダファイルを生成
    if filter_infos:
        generate_multi_rate_header(filter_infos, taps=args.taps)

    # 最終サマリー
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    for name, status in results:
        print(f"  {name}: {status}")
    print("=" * 70)

    success_count = sum(1 for _, s in results if s.startswith("✓"))
    print(f"\nCompleted: {success_count}/{total} filters generated successfully")


def main():
    """メイン処理"""
    args = parse_args()

    if args.generate_all:
        generate_all_filters(args)
    else:
        print("=" * 70)
        print("GPU Audio Upsampler - Filter Coefficient Generation")
        print("=" * 70)
        generate_single_filter(args)


if __name__ == "__main__":
    main()
