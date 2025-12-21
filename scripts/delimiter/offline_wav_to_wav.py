from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

DEFAULT_WEIGHTS_URL_BASE = (
    "https://raw.githubusercontent.com/jeonchangbin49/De-limiter/main/weight"
)
DEFAULT_EXPECTED_SAMPLE_RATE = 44100
DelimiterBackend = Literal["delimiter", "bypass"]


def _as_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.ndim == 2 and x.shape[1] == 1:
        return np.concatenate([x, x], axis=1)
    if x.ndim == 2 and x.shape[1] == 2:
        return x
    raise ValueError(f"Unsupported channel layout: shape={x.shape}")


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _integrated_lufs(x: np.ndarray, sr: int) -> float | None:
    """Compute integrated LUFS; return None if pyloudnorm is unavailable."""
    try:
        import pyloudnorm as pyln
    except Exception:
        return None

    data = x if x.ndim == 1 else x
    try:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(data))
    except Exception:
        return None


def _analyze_audio(audio: np.ndarray, sr: int) -> dict[str, Any]:
    from scripts.analysis.check_headroom import analyze_buffer

    stats = analyze_buffer(audio)
    stats["sample_rate"] = int(sr)
    stats["duration_sec"] = float(len(audio) / sr if sr else 0.0)
    stats["rms"] = _rms(audio)
    stats["lufs"] = _integrated_lufs(audio, int(sr))
    return stats


def _fmt_lufs(value: float | None) -> str:
    return f"{value:.2f} LUFS" if value is not None else "n/a"


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio

    # Polyphase resampling per-channel.
    # Use a rational approximation: up/down = sr_out/sr_in reduced.
    from math import gcd

    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g

    if audio.ndim == 1:
        return resample_poly(audio, up=up, down=down).astype(np.float32)

    out_ch = []
    for ch in range(audio.shape[1]):
        out_ch.append(resample_poly(audio[:, ch], up=up, down=down))
    return np.stack(out_ch, axis=1).astype(np.float32)


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def _ensure_weights(weights_dir: Path, *, url_base: str, download: bool) -> None:
    config_json = weights_dir / "all.json"
    state_pth = weights_dir / "all.pth"

    if config_json.exists() and state_pth.exists():
        return

    if not download:
        raise FileNotFoundError(
            "De-limiter weights not found. Download them first:\n"
            f"- {config_json}\n"
            f"- {state_pth}\n"
            "You can use:\n"
            f"  mkdir -p {weights_dir}\n"
            f"  curl -L -o {config_json} {url_base}/all.json\n"
            f"  curl -L -o {state_pth} {url_base}/all.pth\n"
            "or pass --download-weights to fetch automatically."
        )

    _download_file(f"{url_base}/all.json", config_json)
    _download_file(f"{url_base}/all.pth", state_pth)


def _overlap_add(
    chunks: list[np.ndarray], hop: int, overlap: int, total_len: int
) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 2), dtype=np.float32)

    channels = int(chunks[0].shape[1])
    out_len = hop * (len(chunks) - 1) + int(chunks[0].shape[0])
    out = np.zeros((out_len, channels), dtype=np.float32)
    wsum = np.zeros((out_len,), dtype=np.float32)

    if overlap > 0:
        t = np.linspace(0.0, 1.0, overlap, endpoint=True, dtype=np.float32)
        fade = 0.5 - 0.5 * np.cos(np.pi * t)  # raised-cosine
    else:
        fade = np.zeros((0,), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        start = i * hop
        length = int(chunk.shape[0])
        w = np.ones((length,), dtype=np.float32)
        if overlap > 0 and i > 0:
            w[:overlap] = fade
        if overlap > 0 and i < len(chunks) - 1:
            w[-overlap:] = fade[::-1]

        out[start : start + length] += chunk * w[:, None]
        wsum[start : start + length] += w

    wsum = np.maximum(wsum, 1e-8)
    out = out / wsum[:, None]
    return out[:total_len]


def _write_debug_artifacts(
    debug_dir: Path,
    *,
    input_audio: np.ndarray,
    input_sr: int,
    output_audio: np.ndarray,
    output_sr: int,
    ab_seconds: float,
    ab_gap_sec: float,
) -> dict[str, Any]:
    """Write artifacts that help eyeball/hear the difference quickly.

    Outputs:
    - input.wav / output.wav (resampled to output_sr)
    - ab.wav (input + silence + output)
    - waveform.png (overview + zoom around peak + histogram)
    """

    import matplotlib.pyplot as plt

    debug_dir.mkdir(parents=True, exist_ok=True)

    in_resampled = _resample(input_audio, sr_in=input_sr, sr_out=output_sr)
    out_resampled = output_audio

    # Limit A/B duration to keep files small and comparison quick.
    limit_samples = (
        int(round(max(0.0, ab_seconds) * output_sr)) if ab_seconds > 0 else None
    )
    if limit_samples is not None and limit_samples > 0:
        in_ab = in_resampled[:limit_samples]
        out_ab = out_resampled[:limit_samples]
    else:
        in_ab = in_resampled
        out_ab = out_resampled

    gap = np.zeros((int(round(max(0.0, ab_gap_sec) * output_sr)), 2), dtype=np.float32)
    ab = np.concatenate([in_ab, gap, out_ab], axis=0)

    input_path = debug_dir / "input.wav"
    output_path = debug_dir / "output.wav"
    ab_path = debug_dir / "ab.wav"
    plot_path = debug_dir / "waveform.png"

    sf.write(input_path, in_resampled, output_sr)
    sf.write(output_path, out_resampled, output_sr)
    sf.write(ab_path, ab, output_sr)

    # Plot: overview + zoom around peak + histogram (abs)
    in_left = in_resampled[:, 0]
    out_left = out_resampled[:, 0]

    peak_idx = int(np.argmax(np.abs(in_left))) if in_left.size else 0
    zoom_half = int(round(0.02 * output_sr))  # 20ms window around peak
    z0 = max(0, peak_idx - zoom_half)
    z1 = min(len(in_left), peak_idx + zoom_half)
    t_over = np.arange(len(in_left), dtype=np.float64) / output_sr
    t_zoom = np.arange(z0, z1, dtype=np.float64) / output_sr

    plt.figure(figsize=(14, 9))

    ax1 = plt.subplot(3, 1, 1)
    max_sec = (
        min(float(ab_seconds), float(len(in_left) / output_sr))
        if ab_seconds > 0
        else float(len(in_left) / output_sr)
    )
    view_samples = int(round(max_sec * output_sr))
    view_samples = max(1, min(view_samples, len(in_left)))
    ax1.plot(
        t_over[:view_samples],
        in_left[:view_samples],
        label="input",
        alpha=0.8,
        linewidth=0.9,
    )
    ax1.plot(
        t_over[:view_samples],
        out_left[:view_samples],
        label="output",
        alpha=0.8,
        linewidth=0.9,
    )
    ax1.set_title("Waveform overview (Left channel)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t_zoom, in_left[z0:z1], label="input", alpha=0.8, linewidth=0.9)
    ax2.plot(t_zoom, out_left[z0:z1], label="output", alpha=0.8, linewidth=0.9)
    ax2.set_title("Zoom around input peak (±20ms)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    ax3 = plt.subplot(3, 1, 3)
    bins = 200
    ax3.hist(np.abs(in_left), bins=bins, range=(0.0, 1.2), alpha=0.6, label="|input|")
    ax3.hist(np.abs(out_left), bins=bins, range=(0.0, 1.2), alpha=0.6, label="|output|")
    ax3.set_title("Abs amplitude histogram (Left channel)")
    ax3.set_xlabel("|Amplitude|")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return {
        "debug_dir": str(debug_dir),
        "input_wav": str(input_path),
        "output_wav": str(output_path),
        "ab_wav": str(ab_path),
        "waveform_png": str(plot_path),
        "ab_seconds": float(ab_seconds),
        "ab_gap_sec": float(ab_gap_sec),
        "output_sr": int(output_sr),
    }


def run_backend(
    backend: DelimiterBackend,
    audio_target_sr_stereo: np.ndarray,
    *,
    weights_dir: Path,
    use_gpu: bool,
    chunk_sec: float,
    overlap_sec: float,
    expected_sample_rate: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    meta: dict[str, Any] = {
        "backend": backend,
        "chunk_sec": float(chunk_sec),
        "overlap_sec": float(overlap_sec),
        "expected_sample_rate": int(expected_sample_rate),
    }

    if backend == "bypass":
        return audio_target_sr_stereo, meta

    from scripts.delimiter.model import (
        DelimiterDependencyError,
        build_delimiter_model,
        load_weights_from_dir,
    )

    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise DelimiterDependencyError(
            "PyTorch is not available. Install with: uv sync --extra delimiter"
        ) from e

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    meta["device"] = str(device)

    weights = load_weights_from_dir(weights_dir, target="all")
    model, cfg = build_delimiter_model(weights, device)
    meta["weights"] = {
        "config_json": str(weights.config_json),
        "state_dict_pth": str(weights.state_dict_pth),
        "architecture": cfg.get("args", {})
        .get("model_loss_params", {})
        .get("architecture"),
        "sample_rate": cfg.get("args", {}).get("data_params", {}).get("sample_rate"),
        "nb_channels": cfg.get("args", {}).get("data_params", {}).get("nb_channels"),
    }

    with torch.no_grad():
        if chunk_sec <= 0:
            x = (
                torch.from_numpy(audio_target_sr_stereo.T).unsqueeze(0).to(device)
            )  # (1, 2, T)
            y = model(x)
            out = y[1] if isinstance(y, (tuple, list)) and len(y) >= 2 else y
            out_np = out.squeeze(0).detach().cpu().numpy().T.astype(np.float32)
            return out_np, meta

        sr = int(expected_sample_rate)
        chunk_len = int(round(chunk_sec * sr))
        overlap = int(round(overlap_sec * sr))
        hop = chunk_len - overlap
        if hop <= 0:
            raise ValueError("chunk_sec must be > overlap_sec")

        total_len = int(audio_target_sr_stereo.shape[0])
        if total_len <= chunk_len:
            padded = np.pad(
                audio_target_sr_stereo,
                ((0, chunk_len - total_len), (0, 0)),
                mode="constant",
            )
            x = torch.from_numpy(padded.T).unsqueeze(0).to(device)
            y = model(x)
            out = y[1] if isinstance(y, (tuple, list)) and len(y) >= 2 else y
            out_np = out.squeeze(0).detach().cpu().numpy().T.astype(np.float32)
            return out_np[:total_len], meta

        pad = (hop - (max(0, total_len - chunk_len) % hop)) % hop
        padded = np.pad(audio_target_sr_stereo, ((0, pad), (0, 0)), mode="constant")

        chunks_out: list[np.ndarray] = []
        for start in range(0, padded.shape[0] - chunk_len + 1, hop):
            chunk = padded[start : start + chunk_len]
            x = torch.from_numpy(chunk.T).unsqueeze(0).to(device)
            y = model(x)
            out = y[1] if isinstance(y, (tuple, list)) and len(y) >= 2 else y
            out_np = out.squeeze(0).detach().cpu().numpy().T.astype(np.float32)
            chunks_out.append(out_np)

        out_np = _overlap_add(
            chunks_out, hop=hop, overlap=overlap, total_len=padded.shape[0]
        )
        return out_np[:total_len], meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="De-limiter offline wav->wav PoC (#1008)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--backend",
        choices=["delimiter", "bypass"],
        default="delimiter",
        help="Use 'bypass' to validate pipeline without ML deps",
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("data/delimiter/weight"),
        help="Directory containing all.json/all.pth",
    )
    p.add_argument(
        "--weights-url-base",
        type=str,
        default=DEFAULT_WEIGHTS_URL_BASE,
        help="Base URL to fetch all.json/all.pth from when using --download-weights",
    )
    p.add_argument(
        "--download-weights",
        action="store_true",
        help="Download weights into --weights-dir if missing (for local PoC)",
    )
    p.add_argument("--use-gpu", action="store_true", help="Use CUDA if available")
    p.add_argument(
        "--expected-sample-rate",
        type=int,
        default=DEFAULT_EXPECTED_SAMPLE_RATE,
        help="Target sample rate for inference (resample input to this)",
    )
    p.add_argument(
        "--chunk-sec",
        type=float,
        default=6.0,
        help="Chunk size in seconds (0 to disable chunking)",
    )
    p.add_argument("--overlap-sec", type=float, default=0.25, help="Crossfade overlap")
    p.add_argument(
        "--resample-back",
        action="store_true",
        help="Resample output back to original sample rate",
    )
    p.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="If set, write A/B wav + waveform PNG for quick inspection",
    )
    p.add_argument(
        "--ab-seconds",
        type=float,
        default=8.0,
        help="Seconds to include in A/B output (0 to keep full length)",
    )
    p.add_argument(
        "--ab-gap-sec",
        type=float,
        default=0.5,
        help="Silence gap (seconds) inserted between A and B in ab.wav",
    )
    p.add_argument("--report", type=Path, default=None, help="Write JSON report")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Allow running as a script: `python scripts/delimiter/offline_wav_to_wav.py ...`
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    audio, sr = sf.read(args.input)
    audio = np.asarray(audio, dtype=np.float32)
    audio = _as_stereo(audio)

    if args.expected_sample_rate <= 0:
        raise ValueError("--expected-sample-rate must be positive")

    target_sr = int(args.expected_sample_rate)

    in_stats = _analyze_audio(audio, int(sr))

    audio_target = _resample(audio, sr_in=int(sr), sr_out=target_sr)

    if args.backend == "delimiter":
        _ensure_weights(
            args.weights_dir,
            url_base=str(args.weights_url_base).rstrip("/"),
            download=bool(args.download_weights),
        )

    t0 = time.perf_counter()
    out_target, meta = run_backend(
        args.backend,
        audio_target,
        weights_dir=args.weights_dir,
        use_gpu=bool(args.use_gpu),
        chunk_sec=float(args.chunk_sec),
        overlap_sec=float(args.overlap_sec),
        expected_sample_rate=target_sr,
    )
    dt = time.perf_counter() - t0

    weight_sr = meta.get("weights", {}).get("sample_rate")
    if args.backend == "delimiter" and weight_sr is not None:
        if int(weight_sr) != target_sr:
            raise ValueError(
                f"Weight sample rate ({weight_sr}) does not match --expected-sample-rate ({target_sr})"
            )

    out_audio = out_target
    out_sr = target_sr
    if args.resample_back and int(sr) != target_sr:
        out_audio = _resample(out_target, sr_in=target_sr, sr_out=int(sr))
        out_sr = int(sr)
    meta["input_sample_rate"] = int(sr)
    meta["output_sample_rate"] = int(out_sr)
    meta["resample_back"] = bool(args.resample_back and int(sr) != target_sr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, out_audio, out_sr)

    out_stats = _analyze_audio(out_audio, out_sr)

    duration = float(len(audio) / sr) if sr else 0.0
    rtf = (dt / duration) if duration > 0 else None

    print(
        f"[input] sr={sr} ch=2 dur={duration:.3f}s "
        f"peak={in_stats['peak_linear']:.6f} ({in_stats['peak_dbfs']:.2f} dBFS) "
        f"clip={in_stats['clip_count']} ({in_stats['clip_rate']*100:.6f}%) "
        f"rms={in_stats['rms']:.6f} lufs={_fmt_lufs(in_stats['lufs'])}"
    )
    print(
        f"[output] sr={out_sr} ch=2 "
        f"peak={out_stats['peak_linear']:.6f} ({out_stats['peak_dbfs']:.2f} dBFS) "
        f"clip={out_stats['clip_count']} ({out_stats['clip_rate']*100:.6f}%) "
        f"rms={out_stats['rms']:.6f} lufs={_fmt_lufs(out_stats['lufs'])}"
    )
    print(
        f"[perf] elapsed={dt:.3f}s rtf={rtf:.3f} (lower is faster)"
        if rtf is not None
        else f"[perf] elapsed={dt:.3f}s"
    )

    debug_info: dict[str, Any] | None = None
    if args.debug_dir is not None:
        debug_info = _write_debug_artifacts(
            args.debug_dir,
            input_audio=audio,
            input_sr=int(sr),
            output_audio=out_audio,
            output_sr=int(out_sr),
            ab_seconds=float(args.ab_seconds),
            ab_gap_sec=float(args.ab_gap_sec),
        )
        print(f"[debug] waveform={debug_info['waveform_png']}")
        print(f"[debug] ab={debug_info['ab_wav']}")

    if args.report:
        report = {
            "input": {
                "path": str(args.input),
                "sample_rate": int(sr),
                "stats": in_stats,
            },
            "output": {
                "path": str(args.output),
                "sample_rate": int(out_sr),
                "stats": out_stats,
            },
            "perf": {"elapsed_sec": dt, "rtf": rtf},
            "meta": meta,
            "debug": debug_info,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[report] wrote {args.report}")


if __name__ == "__main__":
    main()
