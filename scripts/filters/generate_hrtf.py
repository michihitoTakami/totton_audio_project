#!/usr/bin/env python3
"""
HRTF FIR Filter Generation (Phase-Preserving)

SOFAファイルから±30°（正三角形配置）のHRIRを抽出し、
位相を保持したままFIRフィルタを生成する。

クロスフィード用途では位相情報（ITD/ILD）が空間定位に必須のため、
元のHRIRの位相をそのまま保持する。

タスク:
- ±30°, elevation 0° のHRIR抽出
- 4チャンネル構成: LL, LR, RL, RR
- 705.6kHz / 768kHz へリサンプリング
- デフォルトで16k-tapへゼロパディング（必要に応じて --taps で変更）

Data Source: HUTUBS - Head-related Transfer Function Database
    https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960

License:
    - HUTUBS SOFA data: CC BY 4.0
    - pysofaconventions: BSD-3-Clause

Attribution:
    HUTUBS - Head-related Transfer Function Database of the
    Technical University of Berlin
    F. Brinkmann et al., TU Berlin, 2019

Usage:
    python scripts/filters/generate_hrtf.py
    python scripts/filters/generate_hrtf.py --size M --output-dir data/crossfeed/hrtf
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
# 以前は 640k-tap を採用していたが、有効IRは数千サンプルで済むため
# デフォルトを 16k に短縮し、必要に応じて --taps で上書きする。
N_TAPS = 16_384
# 高域荒れ対策: 有効成分より十分小さいエネルギーで打ち切る
TRIM_THRESHOLD_DB = -80.0
TRIM_PADDING = 512  # 打ち切り位置からの余白
# 追加の高域緩和（全チャネル共通、緩やかなロールオフ）
GLOBAL_HF_CUTOFF_HZ = 20_000.0
GLOBAL_HF_MIN_GAIN_DB = -9.0
GLOBAL_HF_SLOPE = 6.0
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

# 聴感調整用定数
IPSILATERAL_DIRECT_BLEND = 0.45  # 0=完全にドライ, 1=完全に計測HRTF
CONTRALATERAL_TAIL_START_MS = 0.8
CONTRALATERAL_TAIL_DECAY_MS = 5.5
CONTRALATERAL_HF_CUTOFF_HZ = 5200.0
CONTRALATERAL_HF_MIN_GAIN_DB = -12.0
CONTRALATERAL_HF_SLOPE = 2.4
DC_HEADROOM_DB = 0.5  # 左右耳のDCゲイン上限に与えるヘッドルーム

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


def blend_with_direct_path(hrir: np.ndarray, mix_ratio: float) -> np.ndarray:
    """
    同側のHRTFにドライ信号をブレンドして色付けを抑制する。
    """
    mix = np.clip(mix_ratio, 0.0, 1.0)
    if mix <= 0.0:
        result = np.zeros_like(hrir)
        result[0] = 1.0
        return result
    if mix >= 0.999:
        return hrir

    direct = np.zeros_like(hrir)
    direct[0] = 1.0
    return mix * hrir + (1.0 - mix) * direct


def apply_exponential_tail_taper(
    hrir: np.ndarray, sample_rate: int, start_ms: float, decay_ms: float
) -> np.ndarray:
    """
    遅延成分を指数減衰させて低域のボワつきを抑える。
    """
    if decay_ms <= 0:
        return hrir

    start_idx = int(max(0.0, start_ms) * 1e-3 * sample_rate)
    start_idx = min(start_idx, len(hrir))
    if start_idx >= len(hrir):
        return hrir

    tail_length = len(hrir) - start_idx
    times = np.arange(tail_length, dtype=np.float64) / float(sample_rate)
    tau = decay_ms / 1000.0
    envelope = np.exp(-times / max(tau, 1e-6))

    tapered = hrir.copy()
    tapered[start_idx:] *= envelope.astype(hrir.dtype)
    return tapered


def apply_high_frequency_tilt(
    hrir: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    min_gain_db: float,
    slope: float,
) -> np.ndarray:
    """
    高域にかける緩やかな減衰カーブ（contralateral向け）。
    """
    if cutoff_hz <= 0 or slope <= 0:
        return hrir

    min_gain = 10.0 ** (min_gain_db / 20.0)
    freq = np.fft.rfftfreq(len(hrir), d=1.0 / sample_rate)
    norm = np.clip(freq / cutoff_hz, 0.0, None)
    rolloff = 1.0 / (1.0 + np.power(norm, slope))
    tilt = min_gain + (1.0 - min_gain) * rolloff

    spectrum = np.fft.rfft(hrir)
    shaped = np.fft.irfft(spectrum * tilt, n=len(hrir))
    return shaped.astype(hrir.dtype)


def trim_hrir(hrir: np.ndarray, threshold_db: float, pad: int) -> np.ndarray:
    """
    エネルギーしきい値に基づいてHRIR末尾を打ち切り、余白を付ける。

    Args:
        hrir: 入力HRIR
        threshold_db: ピーク比しきい値[dB]（例: -80dB）
        pad: 打ち切り点からの余白サンプル数

    Returns:
        トリム後のHRIR
    """
    if hrir.size == 0:
        return hrir

    peak = float(np.max(np.abs(hrir)))
    if peak <= 0.0:
        return hrir

    threshold = peak * 10.0 ** (threshold_db / 20.0)
    non_zero = np.nonzero(np.abs(hrir) >= threshold)[0]
    if non_zero.size == 0:
        return hrir[:1]

    end = int(min(len(hrir), non_zero[-1] + pad + 1))
    return hrir[:end]


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
    trim_threshold_db: float = TRIM_THRESHOLD_DB,
    trim_padding: int = TRIM_PADDING,
    global_hf_cutoff_hz: float = GLOBAL_HF_CUTOFF_HZ,
    global_hf_min_gain_db: float = GLOBAL_HF_MIN_GAIN_DB,
    global_hf_slope: float = GLOBAL_HF_SLOPE,
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
    effective_lengths = {}
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

        shaped = resampled
        if name in ("ll", "rr"):
            shaped = blend_with_direct_path(shaped, IPSILATERAL_DIRECT_BLEND)
            print(
                f"  Ipsilateral blend: mix={IPSILATERAL_DIRECT_BLEND:.2f} "
                "(1.0=raw HRTF, 0.0=dry)"
            )
        else:
            shaped = apply_exponential_tail_taper(
                shaped,
                output_rate,
                CONTRALATERAL_TAIL_START_MS,
                CONTRALATERAL_TAIL_DECAY_MS,
            )
            shaped = apply_high_frequency_tilt(
                shaped,
                output_rate,
                CONTRALATERAL_HF_CUTOFF_HZ,
                CONTRALATERAL_HF_MIN_GAIN_DB,
                CONTRALATERAL_HF_SLOPE,
            )
            print(
                "  Contralateral shaping: tail taper start "
                f"{CONTRALATERAL_TAIL_START_MS} ms / decay {CONTRALATERAL_TAIL_DECAY_MS} ms,"
                f" HF tilt cutoff {CONTRALATERAL_HF_CUTOFF_HZ:.0f} Hz"
            )

        trimmed = trim_hrir(shaped, trim_threshold_db, trim_padding)
        if len(trimmed) != len(shaped):
            print(
                f"  Trimmed tail: {len(shaped)} -> {len(trimmed)} samples "
                f"(threshold {trim_threshold_db} dB, pad {trim_padding})"
            )

        if global_hf_cutoff_hz > 0.0 and global_hf_slope > 0.0:
            trimmed = apply_high_frequency_tilt(
                trimmed,
                output_rate,
                global_hf_cutoff_hz,
                global_hf_min_gain_db,
                global_hf_slope,
            )
            print(
                "  Global HF tilt: "
                f"cutoff {global_hf_cutoff_hz:.0f} Hz, min {global_hf_min_gain_db} dB, "
                f"slope {global_hf_slope:.1f}"
            )

        effective_lengths[name] = len(trimmed)

        # ゼロパディングでターゲット長に拡張（位相保持）
        fir = pad_hrir_to_length(trimmed, n_taps)
        print(f"  FIR length: {len(fir)} taps (effective {len(trimmed)})")

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

    # 左右耳それぞれのDCゲイン（同側＋対側）を測定し、上限を超える場合は全体スケール
    left_dc = float(np.sum(channels["ll"]) + np.sum(channels["rl"]))
    right_dc = float(np.sum(channels["rr"]) + np.sum(channels["lr"]))
    ear_dc = {"left": left_dc, "right": right_dc}

    headroom_linear = 10 ** (DC_HEADROOM_DB / 20.0)
    max_allowed = 1.0 / headroom_linear
    max_ear_dc = max(abs(left_dc), abs(right_dc))
    if max_ear_dc > max_allowed + 1e-6:
        scale = max_allowed / max_ear_dc
        print(
            f"\nEar DC clamp: max {max_ear_dc:.4f} > {max_allowed:.3f}, "
            f"applying global scale {scale:.3f}"
        )
        for name in channels:
            channels[name] *= scale
        ear_dc = {ear: gain * scale for ear, gain in ear_dc.items()}
        print(
            f"  Post-clamp left/right DC: {ear_dc['left']:.4f} / {ear_dc['right']:.4f} "
            f"(headroom {DC_HEADROOM_DB} dB)"
        )
    else:
        print(
            f"\nEar DC totals within headroom: left={ear_dc['left']:.4f}, "
            f"right={ear_dc['right']:.4f}"
        )

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

    max_effective = max(effective_lengths.values()) if effective_lengths else n_taps

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
        "ear_dc_gain": ear_dc,
        "source_azimuth_left": -30.0,  # Logical value (HUTUBS uses 330°)
        "source_azimuth_right": TARGET_AZIMUTH_RIGHT,
        "source_elevation": TARGET_ELEVATION,
        "license": "CC BY 4.0",
        "attribution": "HUTUBS - Head-related Transfer Function Database, TU Berlin",
        "source": "https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960",
        "generated_at": datetime.now().isoformat(),
        "storage_format": "channel_major_v1",
        "shaping": {
            "ipsilateral_direct_blend": IPSILATERAL_DIRECT_BLEND,
            "contralateral_tail_start_ms": CONTRALATERAL_TAIL_START_MS,
            "contralateral_tail_decay_ms": CONTRALATERAL_TAIL_DECAY_MS,
            "contralateral_highfreq_cutoff_hz": CONTRALATERAL_HF_CUTOFF_HZ,
            "contralateral_highfreq_min_gain_db": CONTRALATERAL_HF_MIN_GAIN_DB,
            "contralateral_highfreq_slope": CONTRALATERAL_HF_SLOPE,
            "ear_dc_headroom_db": DC_HEADROOM_DB,
            "trim_threshold_db": trim_threshold_db,
            "trim_padding": trim_padding,
            "global_highfreq_cutoff_hz": global_hf_cutoff_hz,
            "global_highfreq_min_gain_db": global_hf_min_gain_db,
            "global_highfreq_slope": global_hf_slope,
            "tap_target": n_taps,
            "tap_effective": max_effective,
        },
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
  python scripts/filters/generate_hrtf.py --all

  # Generate specific size
  python scripts/filters/generate_hrtf.py --size M

  # Specify SOFA directory
  python scripts/filters/generate_hrtf.py --sofa-dir data/crossfeed/raw/sofa

License: HUTUBS data is CC BY 4.0
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
        "--trim-threshold-db",
        type=float,
        default=TRIM_THRESHOLD_DB,
        help="Tail trim threshold relative to peak (dB, default: -80)",
    )
    parser.add_argument(
        "--trim-padding",
        type=int,
        default=TRIM_PADDING,
        help="Samples to keep after trim point (default: 512)",
    )
    parser.add_argument(
        "--global-hf-cutoff",
        type=float,
        default=GLOBAL_HF_CUTOFF_HZ,
        help="Global high-frequency tilt cutoff in Hz (0 to disable)",
    )
    parser.add_argument(
        "--global-hf-min-gain-db",
        type=float,
        default=GLOBAL_HF_MIN_GAIN_DB,
        help="Minimum gain at high frequency for global tilt (dB)",
    )
    parser.add_argument(
        "--global-hf-slope",
        type=float,
        default=GLOBAL_HF_SLOPE,
        help="Slope for global HF tilt (higher = steeper)",
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
    print(
        f"Taps: {args.taps:,} "
        f"(trim @{args.trim_threshold_db} dB, pad {args.trim_padding}, "
        f"global HF {args.global_hf_cutoff} Hz)"
    )

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
                    trim_threshold_db=args.trim_threshold_db,
                    trim_padding=args.trim_padding,
                    global_hf_cutoff_hz=args.global_hf_cutoff,
                    global_hf_min_gain_db=args.global_hf_min_gain_db,
                    global_hf_slope=args.global_hf_slope,
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
