#!/usr/bin/env python3
"""
HRTF FIR Filter Generation (Phase-Preserving)

SOFAファイルから±30°（正三角形配置）のHRIRを抽出し、
2M-tap FIRフィルタを生成する。

クロスフィード用途では位相情報（ITD/ILD）が空間定位に必須のため、
元のHRIRの位相をそのまま保持する。

タスク:
- ±30°, elevation 0° のHRIR抽出
- 4チャンネル構成: LL, LR, RL, RR
- 705.6kHz / 768kHz へリサンプリング
- 2M-tapへゼロパディング（位相保持）

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


def angular_distance(az1: float, az2: float) -> float:
    """
    方位角間の最短角度距離を計算（360°周期を考慮）。

    Args:
        az1: 方位角1 (度、0-360)
        az2: 方位角2 (度、0-360)

    Returns:
        最短角度距離 (度、0-180)
    """
    diff = abs(az1 - az2)
    return min(diff, 360.0 - diff)


def find_nearest_position(
    positions: np.ndarray, target_azimuth: float, target_elevation: float
) -> int:
    """
    最も近い測定位置のインデックスを返す。

    Args:
        positions: 測定位置配列 (N, 3) - [azimuth, elevation, distance]
        target_azimuth: 目標方位角 (度、0-360)
        target_elevation: 目標仰角 (度)

    Returns:
        最近傍位置のインデックス
    """
    # 方位角差を360°周期で計算
    azimuth_diff = np.array(
        [angular_distance(az, target_azimuth) for az in positions[:, 0]]
    )
    elevation_diff = np.abs(positions[:, 1] - target_elevation)

    # 角度距離（球面上の近似距離）
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


def pad_hrir_to_length(hrir: np.ndarray, target_length: int) -> np.ndarray:
    """
    HRIRをターゲット長にゼロパディング。

    HRTFクロスフィードでは位相情報（ITD/ILD）が空間定位に必須のため、
    元のHRIRの位相をそのまま保持する。変換は行わない。

    Args:
        hrir: 入力HRIR（リサンプリング済み）
        target_length: ターゲットタップ数

    Returns:
        ゼロパディングされたHRIR
    """
    hrir_len = len(hrir)

    if hrir_len >= target_length:
        # 長すぎる場合は先頭から切り出し（通常ありえない）
        return hrir[:target_length]

    # 末尾にゼロパディング（因果的フィルタを維持）
    padded = np.zeros(target_length, dtype=hrir.dtype)
    padded[:hrir_len] = hrir

    return padded


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

    # 各チャンネルを処理（第1パス: リサンプリング・パディング）
    channels = {}
    dc_gains = {}
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

        # ゼロパディングでターゲット長に拡張（位相保持）
        fir = pad_hrir_to_length(resampled, n_taps)
        print(f"  FIR length: {len(fir)} taps")

        dc_gains[name] = np.sum(fir)
        print(f"  DC gain (raw): {dc_gains[name]:.6f}")

        channels[name] = fir

    # ILD保持のため、全チャンネル共通スケールで正規化
    # 最大DCゲイン（通常は同側: LL/RR）を基準に正規化
    max_dc_gain = max(abs(g) for g in dc_gains.values())
    print("\n=== DC Normalization (ILD-preserving) ===")
    print(f"Max DC gain: {max_dc_gain:.6f}")

    if max_dc_gain <= 1e-10:
        raise ValueError(
            f"Invalid HRTF data: max DC gain ({max_dc_gain:.2e}) is near zero. "
            "This indicates corrupted or invalid SOFA data."
        )

    for name in channels:
        channels[name] = channels[name] / max_dc_gain
        normalized_dc = np.sum(channels[name])
        print(f"  {name.upper()}: {dc_gains[name]:.6f} → {normalized_dc:.6f}")

    # float32に変換
    for name in channels:
        channels[name] = channels[name].astype(np.float32)

    # チャンネル単位で連続配置（LL 全サンプル → LR → RL → RR）
    channel_major = np.concatenate(
        [channels["ll"], channels["lr"], channels["rl"], channels["rr"]]
    )

    # バイナリ出力
    bin_path = output_dir / f"hrtf_{size.lower()}_{rate_key}.bin"
    channel_major.tofile(bin_path)
    print(
        f"\nSaved binary: {bin_path} ({bin_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )

    # メタデータ
    metadata = {
        "description": f"HRTF FIR filter for head size {size} (phase-preserving)",
        "size_category": size,
        "subject_id": REPRESENTATIVE_SUBJECTS[size],
        "sample_rate": output_rate,
        "rate_family": rate_key,
        "n_taps": n_taps,
        "n_channels": 4,
        "channel_order": ["LL", "LR", "RL", "RR"],
        "phase_type": "original",  # 位相保持（ITD/ILD維持）
        "normalization": "ild_preserving",  # 共通スケール正規化（ILD保持）
        "max_dc_gain": 1.0,  # 最大DCゲインチャンネル=1.0、他はILD分だけ小さい
        "source_azimuth_left": -30.0,  # Logical value (HUTUBS uses 330°)
        "source_azimuth_right": TARGET_AZIMUTH_RIGHT,
        "source_elevation": TARGET_ELEVATION,
        "license": "CC BY-SA 4.0",
        "attribution": "HUTUBS - Head-related Transfer Function Database, TU Berlin",
        "source": "https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960",
        "generated_at": datetime.now().isoformat(),
        "storage_format": "channel_major_v1",
    }

    # JSON出力
    json_path = output_dir / f"hrtf_{size.lower()}_{rate_key}.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {json_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate HRTF FIR filters from SOFA files (phase-preserving)",
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

    print("HRTF FIR Filter Generation (Phase-Preserving)")
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
