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


def run_backend(
    backend: DelimiterBackend,
    audio_44100_stereo: np.ndarray,
    *,
    weights_dir: Path,
    use_gpu: bool,
    chunk_sec: float,
    overlap_sec: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    meta: dict[str, Any] = {
        "backend": backend,
        "chunk_sec": float(chunk_sec),
        "overlap_sec": float(overlap_sec),
    }

    if backend == "bypass":
        return audio_44100_stereo, meta

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
                torch.from_numpy(audio_44100_stereo.T).unsqueeze(0).to(device)
            )  # (1, 2, T)
            y = model(x)
            out = y[1] if isinstance(y, (tuple, list)) and len(y) >= 2 else y
            out_np = out.squeeze(0).detach().cpu().numpy().T.astype(np.float32)
            return out_np, meta

        sr = 44100
        chunk_len = int(round(chunk_sec * sr))
        overlap = int(round(overlap_sec * sr))
        hop = chunk_len - overlap
        if hop <= 0:
            raise ValueError("chunk_sec must be > overlap_sec")

        total_len = int(audio_44100_stereo.shape[0])
        if total_len <= chunk_len:
            padded = np.pad(
                audio_44100_stereo,
                ((0, chunk_len - total_len), (0, 0)),
                mode="constant",
            )
            x = torch.from_numpy(padded.T).unsqueeze(0).to(device)
            y = model(x)
            out = y[1] if isinstance(y, (tuple, list)) and len(y) >= 2 else y
            out_np = out.squeeze(0).detach().cpu().numpy().T.astype(np.float32)
            return out_np[:total_len], meta

        pad = (hop - (max(0, total_len - chunk_len) % hop)) % hop
        padded = np.pad(audio_44100_stereo, ((0, pad), (0, 0)), mode="constant")

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
    p.add_argument("--report", type=Path, default=None, help="Write JSON report")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Allow running as a script: `python scripts/delimiter/offline_wav_to_wav.py ...`
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.analysis.check_headroom import analyze_buffer

    audio, sr = sf.read(args.input)
    audio = np.asarray(audio, dtype=np.float32)
    audio = _as_stereo(audio)

    in_stats = analyze_buffer(audio)
    in_rms = _rms(audio)

    audio_44100 = _resample(audio, sr_in=int(sr), sr_out=44100)

    if args.backend == "delimiter":
        _ensure_weights(
            args.weights_dir,
            url_base=str(args.weights_url_base).rstrip("/"),
            download=bool(args.download_weights),
        )

    t0 = time.perf_counter()
    out_44100, meta = run_backend(
        args.backend,
        audio_44100,
        weights_dir=args.weights_dir,
        use_gpu=bool(args.use_gpu),
        chunk_sec=float(args.chunk_sec),
        overlap_sec=float(args.overlap_sec),
    )
    dt = time.perf_counter() - t0

    out_audio = out_44100
    out_sr = 44100
    if args.resample_back and int(sr) != 44100:
        out_audio = _resample(out_44100, sr_in=44100, sr_out=int(sr))
        out_sr = int(sr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, out_audio, out_sr)

    out_stats = analyze_buffer(out_audio)
    out_rms = _rms(out_audio)

    duration = float(len(audio) / sr) if sr else 0.0
    rtf = (dt / duration) if duration > 0 else None

    print(
        f"[input] sr={sr} ch=2 dur={duration:.3f}s "
        f"peak={in_stats['peak_linear']:.6f} ({in_stats['peak_dbfs']:.2f} dBFS) "
        f"clip={in_stats['clip_count']} ({in_stats['clip_rate']*100:.6f}%) "
        f"rms={in_rms:.6f}"
    )
    print(
        f"[output] sr={out_sr} ch=2 "
        f"peak={out_stats['peak_linear']:.6f} ({out_stats['peak_dbfs']:.2f} dBFS) "
        f"clip={out_stats['clip_count']} ({out_stats['clip_rate']*100:.6f}%) "
        f"rms={out_rms:.6f}"
    )
    print(
        f"[perf] elapsed={dt:.3f}s rtf={rtf:.3f} (lower is faster)"
        if rtf is not None
        else f"[perf] elapsed={dt:.3f}s"
    )

    if args.report:
        report = {
            "input": {
                "path": str(args.input),
                "sample_rate": int(sr),
                "stats": in_stats,
                "rms": in_rms,
            },
            "output": {
                "path": str(args.output),
                "sample_rate": int(out_sr),
                "stats": out_stats,
                "rms": out_rms,
            },
            "perf": {"elapsed_sec": dt, "rtf": rtf},
            "meta": meta,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[report] wrote {args.report}")


if __name__ == "__main__":
    main()
