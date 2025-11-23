#!/usr/bin/env python3
"""
HRTF Linear-Phase FIR Filter Generation

SOFAファイルから±30°（正三角形配置）のHRIRを抽出し、
2M-tap線形位相FIRフィルタを生成する。

タスク:
- ±30°, elevation 0° のHRIR抽出
- 4チャンネル構成: LL, LR, RL, RR
- 705.6kHz / 768kHz へリサンプリング
- 2M-tap線形位相FIRへ変換

Data Source: HUTUBS - Head-related Transfer Function Database
    https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960

License:
    - HUTUBS SOFA data: CC BY-SA 4.0
    - pysofaconventions: BSD-3-Clause

Attribution:
    HUTUBS - Head-related Transfer Function Database of the
    Technical University of Berlin
    F. Brinkmann et al., TU Berlin, 2019

Usage:
    python scripts/generate_hrtf.py
    python scripts/generate_hrtf.py --size M --output-dir data/crossfeed/hrtf
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal

# SOFAファイル読み込み用（オプショナル）
try:
    from pysofaconventions import SOFAFile

    HAS_SOFA = True
except ImportError:
    HAS_SOFA = False
    print("Warning: pysofaconventions not installed. Run: uv sync")


# デフォルト定数
N_TAPS = 2_000_000  # 2M taps
# Note: HUTUBS uses 0-360° azimuth convention, so -30° is represented as 330°
TARGET_AZIMUTH_LEFT = 330.0  # 左スピーカー方位角 (-30° in HUTUBS coordinate)
TARGET_AZIMUTH_RIGHT = 30.0  # 右スピーカー方位角
TARGET_ELEVATION = 0.0  # 仰角

# 代表被験者（hutubs_subjects.jsonより）
REPRESENTATIVE_SUBJECTS = {
    "XS": "pp77",
    "S": "pp6",
    "M": "pp20",
    "L": "pp32",
    "XL": "pp53",
}

# マルチレート設定
RATE_CONFIGS = {
    "44k": {"input_rate": 44100, "output_rate": 705600, "ratio": 16},
    "48k": {"input_rate": 48000, "output_rate": 768000, "ratio": 16},
}

# デフォルトパス
DEFAULT_SOFA_DIR = Path(__file__).parent.parent / "data" / "crossfeed" / "raw" / "sofa"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "crossfeed" / "hrtf"


@dataclass
class HRTFChannels:
    """4チャンネルHRTFデータ"""

    ll: np.ndarray  # 左スピーカー → 左耳（同側）
    lr: np.ndarray  # 左スピーカー → 右耳（対側）
    rl: np.ndarray  # 右スピーカー → 左耳（対側）
    rr: np.ndarray  # 右スピーカー → 右耳（同側）
    sample_rate: int


def find_nearest_position(
    positions: np.ndarray, target_azimuth: float, target_elevation: float
) -> int:
    """
    最も近い測定位置のインデックスを返す。

    Args:
        positions: 測定位置配列 (N, 3) - [azimuth, elevation, distance]
        target_azimuth: 目標方位角 (度)
        target_elevation: 目標仰角 (度)

    Returns:
        最近傍位置のインデックス
    """
    # 方位角と仰角の差を計算（度単位）
    azimuth_diff = np.abs(positions[:, 0] - target_azimuth)
    elevation_diff = np.abs(positions[:, 1] - target_elevation)

    # 角度距離（簡易計算）
    distance = np.sqrt(azimuth_diff**2 + elevation_diff**2)

    return int(np.argmin(distance))


def load_hrir_from_sofa(sofa_path: Path) -> HRTFChannels:
    """
    SOFAファイルからHRIRを読み込む。

    Args:
        sofa_path: SOFAファイルパス

    Returns:
        HRTFChannels: 4チャンネルHRIRデータ
    """
    if not HAS_SOFA:
        raise ImportError("pysofaconventions is required. Run: uv sync")

    print(f"Loading SOFA file: {sofa_path}")

    sofa = SOFAFile(str(sofa_path), "r")

    # サンプルレート取得（MaskedArrayの場合があるのでfloatに変換）
    sample_rate_raw = sofa.getSamplingRate()
    sample_rate = float(np.asarray(sample_rate_raw).flat[0])
    print(f"  Sample rate: {sample_rate} Hz")

    # 測定位置取得
    positions = sofa.getSourcePositionValues()
    print(f"  Measurement positions: {len(positions)}")

    # HRIRデータ取得
    hrir_data = sofa.getDataIR()
    print(
        f"  HRIR shape: {hrir_data.shape}"
    )  # (M, R, N) - positions, receivers, samples

    # ±30°の位置を探す
    idx_left = find_nearest_position(positions, TARGET_AZIMUTH_LEFT, TARGET_ELEVATION)
    idx_right = find_nearest_position(positions, TARGET_AZIMUTH_RIGHT, TARGET_ELEVATION)

    actual_left = positions[idx_left]
    actual_right = positions[idx_right]
    print(
        f"  Left speaker position: azimuth={actual_left[0]:.1f}°, elevation={actual_left[1]:.1f}°"
    )
    print(
        f"  Right speaker position: azimuth={actual_right[0]:.1f}°, elevation={actual_right[1]:.1f}°"
    )

    # HRIRを抽出（receiver 0 = 左耳, receiver 1 = 右耳）
    # 左スピーカー (-30°) からのHRIR
    ll = hrir_data[idx_left, 0, :]  # 左スピーカー → 左耳
    lr = hrir_data[idx_left, 1, :]  # 左スピーカー → 右耳

    # 右スピーカー (+30°) からのHRIR
    rl = hrir_data[idx_right, 0, :]  # 右スピーカー → 左耳
    rr = hrir_data[idx_right, 1, :]  # 右スピーカー → 右耳

    print(f"  HRIR length: {len(ll)} samples ({len(ll)/sample_rate*1000:.1f} ms)")

    sofa.close()

    return HRTFChannels(
        ll=ll.astype(np.float64),
        lr=lr.astype(np.float64),
        rl=rl.astype(np.float64),
        rr=rr.astype(np.float64),
        sample_rate=int(sample_rate),
    )


def resample_hrir(hrir: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """
    HRIRをターゲットサンプルレートにリサンプリング。

    Args:
        hrir: 入力HRIR
        orig_rate: 元のサンプルレート
        target_rate: ターゲットサンプルレート

    Returns:
        リサンプリングされたHRIR
    """
    if orig_rate == target_rate:
        return hrir

    # リサンプリング比率
    gcd = np.gcd(target_rate, orig_rate)
    up = target_rate // gcd
    down = orig_rate // gcd

    print(f"  Resampling: {orig_rate} Hz → {target_rate} Hz (up={up}, down={down})")

    # scipy.signal.resample_polyを使用
    resampled = signal.resample_poly(hrir, up, down)

    return resampled


def convert_to_linear_phase_fir(
    hrir: np.ndarray, target_length: int, sample_rate: int
) -> np.ndarray:
    """
    HRIRを線形位相FIRフィルタに変換。

    線形位相フィルタは対称性を持ち、位相歪みがない。
    クロスフィード用途では位置感再現のためプリリンギングが許容される。

    Args:
        hrir: 入力HRIR
        target_length: ターゲットタップ数
        sample_rate: サンプルレート

    Returns:
        線形位相FIRフィルタ係数
    """
    # HRIRの長さ
    hrir_len = len(hrir)

    # FFTサイズ（2のべき乗に丸める）
    fft_size = 2 ** int(np.ceil(np.log2(max(hrir_len, target_length) * 2)))

    # HRIRの周波数応答を取得
    H = np.fft.fft(hrir, fft_size)

    # 振幅スペクトル
    magnitude = np.abs(H)

    # 線形位相（ゼロ位相）を適用
    # 対称なインパルス応答を生成するため、位相をゼロに
    H_linear = magnitude

    # IFFTで時間領域に戻す
    h_linear = np.fft.ifft(H_linear).real

    # 対称化（線形位相フィルタは対称）
    # 中心をシフトして対称にする
    h_linear = np.fft.fftshift(h_linear)

    # ターゲット長にトリミング/パディング
    if len(h_linear) > target_length:
        # 中心から切り出し
        center = len(h_linear) // 2
        start = center - target_length // 2
        h_linear = h_linear[start : start + target_length]
    else:
        # ゼロパディング
        pad_total = target_length - len(h_linear)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        h_linear = np.pad(h_linear, (pad_left, pad_right))

    # 正規化（DCゲイン = 1.0）
    dc_gain = np.sum(h_linear)
    if abs(dc_gain) > 1e-10:
        h_linear = h_linear / dc_gain

    return h_linear


def generate_hrtf_filters(
    sofa_path: Path,
    output_dir: Path,
    size: str,
    rate_key: str,
    n_taps: int = N_TAPS,
) -> dict:
    """
    HRTFフィルタを生成して保存。

    Args:
        sofa_path: SOFAファイルパス
        output_dir: 出力ディレクトリ
        size: サイズカテゴリ (XS/S/M/L/XL)
        rate_key: レート設定キー (44k/48k)
        n_taps: タップ数

    Returns:
        メタデータ辞書
    """
    config = RATE_CONFIGS[rate_key]
    output_rate = config["output_rate"]

    print(f"\n{'='*60}")
    print(f"Generating HRTF filter: size={size}, rate={rate_key}")
    print(f"{'='*60}")

    # HRIRをロード
    hrtf = load_hrir_from_sofa(sofa_path)

    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 各チャンネルを処理
    channels = {}
    for name, hrir in [
        ("ll", hrtf.ll),
        ("lr", hrtf.lr),
        ("rl", hrtf.rl),
        ("rr", hrtf.rr),
    ]:
        print(f"\nProcessing channel: {name.upper()}")

        # リサンプリング
        resampled = resample_hrir(hrir, hrtf.sample_rate, output_rate)
        print(f"  Resampled length: {len(resampled)} samples")

        # 線形位相FIRに変換
        fir = convert_to_linear_phase_fir(resampled, n_taps, output_rate)
        print(f"  FIR length: {len(fir)} taps")

        channels[name] = fir.astype(np.float32)

    # インターリーブして保存（LL, LR, RL, RR の順）
    interleaved = np.column_stack(
        [channels["ll"], channels["lr"], channels["rl"], channels["rr"]]
    ).flatten()

    # バイナリ出力
    bin_path = output_dir / f"hrtf_{size.lower()}_{rate_key}.bin"
    interleaved.tofile(bin_path)
    print(
        f"\nSaved binary: {bin_path} ({bin_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )

    # メタデータ
    metadata = {
        "description": f"HRTF linear-phase FIR filter for head size {size}",
        "size_category": size,
        "subject_id": REPRESENTATIVE_SUBJECTS[size],
        "sample_rate": output_rate,
        "rate_family": rate_key,
        "n_taps": n_taps,
        "n_channels": 4,
        "channel_order": ["LL", "LR", "RL", "RR"],
        "phase_type": "linear",
        "source_azimuth_left": -30.0,  # Logical value (HUTUBS uses 330°)
        "source_azimuth_right": TARGET_AZIMUTH_RIGHT,
        "source_elevation": TARGET_ELEVATION,
        "license": "CC BY-SA 4.0",
        "attribution": "HUTUBS - Head-related Transfer Function Database, TU Berlin",
        "source": "https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960",
        "generated_at": datetime.now().isoformat(),
    }

    # JSON出力
    json_path = output_dir / f"hrtf_{size.lower()}_{rate_key}.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {json_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate HRTF linear-phase FIR filters from SOFA files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all sizes and rates
  python scripts/generate_hrtf.py --all

  # Generate specific size
  python scripts/generate_hrtf.py --size M

  # Specify SOFA directory
  python scripts/generate_hrtf.py --sofa-dir data/crossfeed/raw/sofa

License: HUTUBS data is CC BY-SA 4.0
Attribution: HUTUBS - Head-related Transfer Function Database, TU Berlin
        """,
    )

    parser.add_argument(
        "--size",
        choices=["XS", "S", "M", "L", "XL"],
        help="Head size category to generate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all sizes and rates",
    )
    parser.add_argument(
        "--rate",
        choices=["44k", "48k"],
        default=None,
        help="Rate family (default: both)",
    )
    parser.add_argument(
        "--taps",
        type=int,
        default=N_TAPS,
        help=f"Number of filter taps (default: {N_TAPS})",
    )
    parser.add_argument(
        "--sofa-dir",
        type=Path,
        default=DEFAULT_SOFA_DIR,
        help="Directory containing SOFA files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated filters",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if not HAS_SOFA:
        print("Error: pysofaconventions is required. Run: uv sync")
        return 1

    # 処理対象を決定
    if args.all:
        sizes = list(REPRESENTATIVE_SUBJECTS.keys())
    elif args.size:
        sizes = [args.size]
    else:
        parser.print_help()
        return 1

    rates = [args.rate] if args.rate else list(RATE_CONFIGS.keys())

    print("HRTF Linear-Phase FIR Filter Generation")
    print(f"SOFA directory: {args.sofa_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sizes: {sizes}")
    print(f"Rates: {rates}")
    print(f"Taps: {args.taps:,}")

    # SOFAディレクトリチェック
    if not args.sofa_dir.exists():
        print(f"\nError: SOFA directory not found: {args.sofa_dir}")
        print("Please download HUTUBS SOFA files from:")
        print(
            "  https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960"
        )
        return 1

    # 各サイズ・レートで生成
    generated = []
    for size in sizes:
        subject_id = REPRESENTATIVE_SUBJECTS[size]
        sofa_path = args.sofa_dir / f"{subject_id}.sofa"

        if not sofa_path.exists():
            print(f"\nWarning: SOFA file not found: {sofa_path}")
            print(f"  Expected: {subject_id}.sofa for size {size}")
            continue

        for rate in rates:
            try:
                generate_hrtf_filters(
                    sofa_path=sofa_path,
                    output_dir=args.output_dir,
                    size=size,
                    rate_key=rate,
                    n_taps=args.taps,
                )
                generated.append(f"{size}_{rate}")
            except Exception as e:
                print(f"\nError generating {size}/{rate}: {e}")
                continue

    # サマリー
    print(f"\n{'='*60}")
    print("Generation Summary")
    print(f"{'='*60}")
    print(f"Generated: {len(generated)} filters")
    for name in generated:
        print(f"  - hrtf_{name.lower()}.bin")

    if len(generated) < len(sizes) * len(rates):
        print("\nWarning: Some filters could not be generated.")
        print("Please ensure all SOFA files are present in the SOFA directory.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
