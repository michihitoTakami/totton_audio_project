#!/usr/bin/env python3
"""
GPU Audio Upsampler - Phase 1: Filter Coefficient Generation

131,072タップの最小位相FIRフィルタを生成し、検証する。

仕様:
- タップ数: 131,072 (128k)
- 位相特性: 最小位相（プリリンギング排除）
- 通過帯域: 0-20,000 Hz
- 阻止帯域: 22,050 Hz以降
- 阻止帯域減衰: -180 dB以下
- 窓関数: Kaiser (β ≈ 18)
- アップサンプリング倍率: 16倍
"""

import json
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 定数定義
N_TAPS = 131072  # 128k taps
SAMPLE_RATE_INPUT = 44100  # 入力サンプルレート (Hz)
UPSAMPLE_RATIO = 16  # アップサンプリング倍率
SAMPLE_RATE_OUTPUT = SAMPLE_RATE_INPUT * UPSAMPLE_RATIO  # 705.6 kHz

# フィルタ設計パラメータ
PASSBAND_END = 20000  # 通過帯域終端 (Hz)
STOPBAND_START = 22050  # 阻止帯域開始 (Hz) - ナイキスト周波数
STOPBAND_ATTENUATION_DB = 180  # 阻止帯域減衰量 (dB)
# Kaiser βパラメータ: A(dB)の減衰量に対して β ≈ 0.1102*(A-8.7) + 0.07886*(A-8.7)
# A=180dBの場合、β ≈ 18.9 + 13.5 ≈ 32が必要
# より安全に β=40 を使用（より急峻な遮断特性）
KAISER_BETA = 40  # Kaiser窓のβパラメータ


def design_linear_phase_filter():
    """
    線形位相FIRフィルタを設計する。

    Returns:
        np.ndarray: 線形位相FIRフィルタ係数
    """
    print(f"線形位相FIRフィルタ設計中...")
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
        window=('kaiser', KAISER_BETA),
        fs=1.0,  # 正規化周波数を使用
        scale=True
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
    # 131k tapsの場合、より精度の高い変換のため8倍=1,048,576を使用
    n_fft = 2 ** int(np.ceil(np.log2(len(h_linear) * 8)))

    h_min_phase = signal.minimum_phase(h_linear, method='homomorphic', n_fft=n_fft)

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
    energy_first_half = np.sum(h[:mid_point]**2)
    energy_second_half = np.sum(h[mid_point:]**2)
    energy_ratio = energy_first_half / (energy_second_half + 1e-12)

    # ピーク位置が先頭から1%以内 かつ エネルギーが前半に集中（比率>10）
    peak_threshold = int(len(h) * 0.01)
    is_peak_at_front = peak_idx < peak_threshold
    is_energy_causal = energy_ratio > 10

    results = {
        'passband_ripple_db': float(passband_ripple_db),
        'stopband_attenuation_db': float(abs(stopband_attenuation)),
        'peak_position': int(peak_idx),
        'peak_threshold_samples': int(peak_threshold),
        'energy_ratio_first_to_second_half': float(energy_ratio),
        'meets_stopband_spec': bool(abs(stopband_attenuation) >= STOPBAND_ATTENUATION_DB),
        'is_minimum_phase': bool(is_peak_at_front and is_energy_causal)
    }

    print(f"  通過帯域リップル: {passband_ripple_db:.3f} dB")
    print(f"  阻止帯域減衰: {abs(stopband_attenuation):.1f} dB (目標: {STOPBAND_ATTENUATION_DB} dB)")
    print(f"  阻止帯域スペック: {'✓ 合格' if results['meets_stopband_spec'] else '✗ 不合格'}")
    print(f"  ピーク位置: サンプル {peak_idx} (先頭1%={peak_threshold}サンプル以内: {'✓' if is_peak_at_front else '✗'})")
    print(f"  エネルギー比(前半/後半): {energy_ratio:.1f} (目標: >10)")
    print(f"  最小位相特性: {'✓ 確認' if results['is_minimum_phase'] else '✗ 未確認'}")

    return results


def plot_responses(h_linear, h_min_phase, output_dir='plots/analysis'):
    """
    フィルタ特性をプロットする。

    Args:
        h_linear: 線形位相フィルタ係数
        h_min_phase: 最小位相フィルタ係数
        output_dir: プロット出力ディレクトリ
    """
    print(f"\nプロット生成中... ({output_dir})")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # フォント設定（日本語対応）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. インパルス応答の比較
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 線形位相インパルス応答（中央部分のみ表示）
    center = len(h_linear) // 2
    display_range = 2000
    t_linear = np.arange(-display_range, display_range)
    h_linear_center = h_linear[center-display_range:center+display_range]

    axes[0].plot(t_linear, h_linear_center, linewidth=0.5)
    axes[0].set_title('Linear Phase Impulse Response (Center Region)', fontsize=12)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0, color='r', linestyle='--', alpha=0.5, label='Center')
    axes[0].legend()

    # 最小位相インパルス応答（先頭部分のみ表示）
    display_range_min = 4000
    t_min = np.arange(display_range_min)
    h_min_display = h_min_phase[:display_range_min]

    axes[1].plot(t_min, h_min_display, linewidth=0.5, color='orange')
    axes[1].set_title('Minimum Phase Impulse Response (Front Region - No Pre-ringing)', fontsize=12)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(0, color='r', linestyle='--', alpha=0.5, label='t=0')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path / 'impulse_response.png', dpi=150)
    print(f"  保存: impulse_response.png")
    plt.close()

    # 2. 周波数応答（振幅）
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    w_linear, H_linear = signal.freqz(h_linear, worN=16384, fs=SAMPLE_RATE_OUTPUT)
    w_min, H_min = signal.freqz(h_min_phase, worN=16384, fs=SAMPLE_RATE_OUTPUT)

    H_linear_db = 20 * np.log10(np.abs(H_linear) + 1e-12)
    H_min_db = 20 * np.log10(np.abs(H_min) + 1e-12)

    # 全体表示
    axes[0].plot(w_linear / 1000, H_linear_db, label='Linear Phase', linewidth=1, alpha=0.7)
    axes[0].plot(w_min / 1000, H_min_db, label='Minimum Phase', linewidth=1, alpha=0.7)
    axes[0].set_title('Magnitude Response (Full Range)', fontsize=12)
    axes[0].set_xlabel('Frequency (kHz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_ylim(-200, 5)
    axes[0].axhline(-180, color='r', linestyle='--', alpha=0.5, label='-180dB Target')
    axes[0].axvline(PASSBAND_END / 1000, color='g', linestyle='--', alpha=0.5, label='Passband End')
    axes[0].axvline(STOPBAND_START / 1000, color='orange', linestyle='--', alpha=0.5, label='Stopband Start')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 通過帯域詳細
    passband_mask = w_min <= PASSBAND_END * 1.1
    axes[1].plot(w_min[passband_mask] / 1000, H_min_db[passband_mask], linewidth=1, color='orange')
    axes[1].set_title('Magnitude Response (Passband Detail)', fontsize=12)
    axes[1].set_xlabel('Frequency (kHz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].axvline(PASSBAND_END / 1000, color='g', linestyle='--', alpha=0.5, label='Passband End')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path / 'frequency_response.png', dpi=150)
    print(f"  保存: frequency_response.png")
    plt.close()

    # 3. 位相応答
    fig, ax = plt.subplots(figsize=(14, 6))

    _, H_linear_complex = signal.freqz(h_linear, worN=8192, fs=SAMPLE_RATE_OUTPUT)
    _, H_min_complex = signal.freqz(h_min_phase, worN=8192, fs=SAMPLE_RATE_OUTPUT)

    phase_linear = np.unwrap(np.angle(H_linear_complex))
    phase_min = np.unwrap(np.angle(H_min_complex))

    w_phase, _ = signal.freqz(h_linear, worN=8192, fs=SAMPLE_RATE_OUTPUT)

    ax.plot(w_phase / 1000, phase_linear, label='Linear Phase', linewidth=1, alpha=0.7)
    ax.plot(w_phase / 1000, phase_min, label='Minimum Phase', linewidth=1, alpha=0.7)
    ax.set_title('Phase Response', fontsize=12)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Phase (radians)')
    ax.axvline(PASSBAND_END / 1000, color='g', linestyle='--', alpha=0.5, label='Passband End')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'phase_response.png', dpi=150)
    print(f"  保存: phase_response.png")
    plt.close()


def export_coefficients(h, metadata, output_dir='data/coefficients'):
    """
    フィルタ係数をエクスポートする。

    Args:
        h: フィルタ係数
        metadata: メタデータ辞書
        output_dir: 出力ディレクトリ
    """
    print(f"\n係数エクスポート中... ({output_dir})")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. バイナリ形式（float32）
    h_float32 = h.astype(np.float32)
    binary_path = output_path / 'filter_131k_min_phase.bin'
    h_float32.tofile(binary_path)
    print(f"  保存: {binary_path} ({binary_path.stat().st_size / 1024:.1f} KB)")

    # 2. C++ヘッダファイル
    header_path = output_path / 'filter_coefficients.h'
    with open(header_path, 'w') as f:
        f.write("// Auto-generated filter coefficients\n")
        f.write("// GPU Audio Upsampler - Phase 1\n")
        f.write(f"// Generated: {metadata['generation_date']}\n\n")
        f.write("#ifndef FILTER_COEFFICIENTS_H\n")
        f.write("#define FILTER_COEFFICIENTS_H\n\n")
        f.write("#include <cstddef>\n\n")
        f.write(f"constexpr size_t FILTER_TAPS = {len(h)};\n")
        f.write(f"constexpr int SAMPLE_RATE_INPUT = {metadata['sample_rate_input']};\n")
        f.write(f"constexpr int SAMPLE_RATE_OUTPUT = {metadata['sample_rate_output']};\n")
        f.write(f"constexpr int UPSAMPLE_RATIO = {metadata['upsample_ratio']};\n\n")
        f.write("// Filter coefficients (float32)\n")
        f.write("// IMPORTANT: 131k taps (512KB) is too large for embedding in source code.\n")
        f.write("// Recommended approach: Load from binary file at runtime using std::ifstream.\n")
        f.write("// Binary file: filter_131k_min_phase.bin (same directory)\n")
        f.write("// Example:\n")
        f.write("//   std::ifstream ifs(\"filter_131k_min_phase.bin\", std::ios::binary);\n")
        f.write("//   std::vector<float> coeffs(FILTER_TAPS);\n")
        f.write("//   ifs.read(reinterpret_cast<char*>(coeffs.data()), FILTER_TAPS * sizeof(float));\n")
        f.write("extern const float FILTER_COEFFICIENTS[FILTER_TAPS];\n\n")
        f.write("#endif // FILTER_COEFFICIENTS_H\n")
    print(f"  保存: {header_path}")

    # 3. メタデータJSON
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  保存: {metadata_path}")


def main():
    """メイン処理"""
    print("=" * 70)
    print("GPU Audio Upsampler - Phase 1: Filter Coefficient Generation")
    print("=" * 70)

    # 1. 線形位相フィルタ設計
    h_linear = design_linear_phase_filter()

    # 2. 最小位相変換
    h_min_phase = convert_to_minimum_phase(h_linear)

    # 3. 仕様検証
    validation_results = validate_specifications(h_min_phase)

    # 4. プロット生成
    plot_responses(h_linear, h_min_phase)

    # 5. メタデータ作成
    from datetime import datetime
    metadata = {
        'generation_date': datetime.now().isoformat(),
        'n_taps': N_TAPS,
        'sample_rate_input': SAMPLE_RATE_INPUT,
        'sample_rate_output': SAMPLE_RATE_OUTPUT,
        'upsample_ratio': UPSAMPLE_RATIO,
        'passband_end_hz': PASSBAND_END,
        'stopband_start_hz': STOPBAND_START,
        'target_stopband_attenuation_db': STOPBAND_ATTENUATION_DB,
        'kaiser_beta': KAISER_BETA,
        'validation_results': validation_results
    }

    # 6. 係数エクスポート
    export_coefficients(h_min_phase, metadata)

    # 7. 最終レポート
    print("\n" + "=" * 70)
    print("Phase 1 完了")
    print("=" * 70)
    print(f"✓ {N_TAPS}タップ最小位相FIRフィルタ生成完了")
    print(f"✓ 阻止帯域減衰: {validation_results['stopband_attenuation_db']:.1f} dB")
    print(f"  {'合格' if validation_results['meets_stopband_spec'] else '不合格'} (目標: {STOPBAND_ATTENUATION_DB} dB以上)")
    print(f"✓ 最小位相特性: {'確認済み' if validation_results['is_minimum_phase'] else '要確認'}")
    print(f"✓ 係数ファイル: data/coefficients/")
    print(f"✓ 検証プロット: plots/analysis/")
    print("=" * 70)


if __name__ == '__main__':
    main()
