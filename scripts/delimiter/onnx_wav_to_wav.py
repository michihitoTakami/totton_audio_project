from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ExecutionProvider = Literal["cpu", "cuda", "tensorrt"]


def _as_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.ndim == 2 and x.shape[1] == 1:
        return np.concatenate([x, x], axis=1)
    if x.ndim == 2 and x.shape[1] == 2:
        return x
    raise ValueError(f"Unsupported channel layout: shape={x.shape}")


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio

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


def _providers_for(ep: ExecutionProvider) -> list[str]:
    if ep == "cpu":
        return ["CPUExecutionProvider"]
    if ep == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if ep == "tensorrt":
        return [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    raise ValueError(f"Unknown execution provider: {ep}")


def _infer_onnx(
    audio_44100_stereo: np.ndarray,
    *,
    model_path: Path,
    provider: ExecutionProvider,
    intra_op_threads: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import onnxruntime as ort
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "onnxruntime is not available. Install with: uv sync --extra onnxruntime"
        ) from e

    meta: dict[str, Any] = {
        "backend": "ort",
        "provider": provider,
        "model_path": str(model_path),
    }

    sess_options = ort.SessionOptions()
    if intra_op_threads > 0:
        sess_options.intra_op_num_threads = int(intra_op_threads)
        meta["intra_op_threads"] = int(intra_op_threads)

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=_providers_for(provider),
    )

    input_name = session.get_inputs()[0].name
    x = audio_44100_stereo.T[None, :, :].astype(np.float32)  # (1, 2, T)

    outputs = session.run(None, {input_name: x})
    if not outputs:
        raise RuntimeError("onnxruntime returned no outputs")

    # Heuristic: many enhancement models output multiple tensors; "enhanced" tends to be last.
    out = np.asarray(outputs[-1])

    # Normalize to (T, 2)
    if out.ndim == 3 and out.shape[0] == 1 and out.shape[1] == 2:
        y = out[0].T
    elif out.ndim == 2 and out.shape[0] == 2:
        y = out.T
    elif out.ndim == 2 and out.shape[1] == 2:
        y = out
    else:
        raise RuntimeError(f"Unsupported ONNX output shape: {out.shape}")

    return y.astype(np.float32), meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="De-limiter ONNX wav->wav runner (#1017)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model", type=Path, required=True, help="Path to delimiter ONNX model"
    )
    p.add_argument(
        "--provider",
        choices=["cpu", "cuda", "tensorrt"],
        default="cpu",
        help="ONNX Runtime execution provider",
    )
    p.add_argument(
        "--chunk-sec",
        type=float,
        default=6.0,
        help="Chunk size in seconds (0 to disable chunking)",
    )
    p.add_argument("--overlap-sec", type=float, default=0.25, help="Crossfade overlap")
    p.add_argument(
        "--intra-op-threads",
        type=int,
        default=0,
        help="ORT intra-op threads (0 = default)",
    )
    p.add_argument(
        "--resample-back",
        action="store_true",
        help="Resample output back to original sample rate",
    )
    p.add_argument("--report", type=Path, default=None, help="Write JSON report")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Allow running as a script: `python scripts/delimiter/onnx_wav_to_wav.py ...`
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    audio, sr = sf.read(args.input)
    audio = np.asarray(audio, dtype=np.float32)
    audio = _as_stereo(audio)

    audio_44100 = _resample(audio, sr_in=int(sr), sr_out=44100)

    t0 = time.perf_counter()
    meta: dict[str, Any] = {}
    if args.chunk_sec <= 0:
        out_44100, meta = _infer_onnx(
            audio_44100,
            model_path=args.model,
            provider=args.provider,
            intra_op_threads=int(args.intra_op_threads),
        )
    else:
        sr_infer = 44100
        chunk_len = int(round(float(args.chunk_sec) * sr_infer))
        overlap = int(round(float(args.overlap_sec) * sr_infer))
        hop = chunk_len - overlap
        if hop <= 0:
            raise ValueError("chunk-sec must be > overlap-sec")

        total_len = int(audio_44100.shape[0])
        if total_len <= chunk_len:
            padded = np.pad(
                audio_44100, ((0, chunk_len - total_len), (0, 0)), mode="constant"
            )
            out_chunk, meta = _infer_onnx(
                padded,
                model_path=args.model,
                provider=args.provider,
                intra_op_threads=int(args.intra_op_threads),
            )
            out_44100 = out_chunk[:total_len]
        else:
            pad = (hop - (max(0, total_len - chunk_len) % hop)) % hop
            padded = np.pad(audio_44100, ((0, pad), (0, 0)), mode="constant")

            chunks_out: list[np.ndarray] = []
            for start in range(0, padded.shape[0] - chunk_len + 1, hop):
                chunk = padded[start : start + chunk_len]
                out_chunk, meta = _infer_onnx(
                    chunk,
                    model_path=args.model,
                    provider=args.provider,
                    intra_op_threads=int(args.intra_op_threads),
                )
                chunks_out.append(out_chunk)

            out_44100 = _overlap_add(
                chunks_out, hop=hop, overlap=overlap, total_len=padded.shape[0]
            )[:total_len]

    dt = time.perf_counter() - t0

    out_audio = out_44100
    out_sr = 44100
    if args.resample_back and int(sr) != 44100:
        out_audio = _resample(out_44100, sr_in=44100, sr_out=int(sr))
        out_sr = int(sr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, out_audio, out_sr)

    duration = float(len(audio) / sr) if sr else 0.0
    rtf = (dt / duration) if duration > 0 else None

    print(f"[input] sr={sr} ch=2 dur={duration:.3f}s")
    print(f"[output] sr={out_sr} ch=2")
    print(
        f"[perf] elapsed={dt:.3f}s rtf={rtf:.3f} (lower is faster)"
        if rtf is not None
        else f"[perf] elapsed={dt:.3f}s"
    )

    if args.report:
        report = {
            "input": {"path": str(args.input), "sample_rate": int(sr)},
            "output": {"path": str(args.output), "sample_rate": int(out_sr)},
            "perf": {"elapsed_sec": dt, "rtf": rtf},
            "meta": meta,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[report] wrote {args.report}")


if __name__ == "__main__":
    main()
