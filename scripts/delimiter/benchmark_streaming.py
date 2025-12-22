from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ExecutionProvider = Literal["cpu", "cuda", "tensorrt"]
InferFn = Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]]


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
    out: np.ndarray = np.zeros((out_len, channels), dtype=np.float32)
    wsum: np.ndarray = np.zeros((out_len,), dtype=np.float32)

    if overlap > 0:
        t = np.linspace(0.0, 1.0, overlap, endpoint=True, dtype=np.float32)
        fade = 0.5 - 0.5 * np.cos(np.pi * t)  # raised-cosine
    else:
        fade = np.zeros((0,), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        start = i * hop
        length = int(chunk.shape[0])
        w: np.ndarray = np.ones((length,), dtype=np.float32)
        if overlap > 0 and i > 0:
            w[:overlap] = fade
        if overlap > 0 and i < len(chunks) - 1:
            w[-overlap:] = fade[::-1]

        out[start : start + length] += chunk * w[:, None]
        wsum[start : start + length] += w

    wsum = np.maximum(wsum, 1e-8)
    out = out / wsum[:, None]
    return out[:total_len]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


@dataclass
class ChunkTiming:
    index: int
    frames: int
    duration_sec: float
    infer_ms: float


@dataclass
class BenchmarkReport:
    chunk_timings: list[ChunkTiming]
    total_elapsed_sec: float
    realtime_factor: Optional[float]
    throughput_x: Optional[float]
    estimated_latency_sec: float
    hop_sec: float
    resources: Optional[dict[str, Any]]
    meta: dict[str, Any]
    failed_chunks: list[int]
    failure_messages: list[str]
    fallback_on_error: bool

    def to_dict(self) -> dict[str, Any]:
        chunk_ms = [c.infer_ms for c in self.chunk_timings]
        chunk_count = len(self.chunk_timings)
        error_count = len(self.failed_chunks)
        error_rate = float(error_count) / float(chunk_count) if chunk_count else 0.0
        return {
            "metrics": {
                "chunk_count": chunk_count,
                "mean_ms_per_chunk": float(np.mean(chunk_ms)) if chunk_ms else 0.0,
                "p95_ms_per_chunk": _percentile(chunk_ms, 95.0),
                "max_ms_per_chunk": max(chunk_ms) if chunk_ms else 0.0,
                "realtime_factor": self.realtime_factor,
                "throughput_x": self.throughput_x,
                "estimated_initial_latency_sec": self.estimated_latency_sec,
                "steady_state_hop_sec": self.hop_sec,
                "total_elapsed_sec": self.total_elapsed_sec,
                "drop_rate": 0.0,  # offline benchmark, no playback buffer underrun
                "error_rate": error_rate,
                "error_chunks": self.failed_chunks,
            },
            "resources": self.resources or {},
            "meta": self.meta,
            "chunks": [
                {
                    "index": c.index,
                    "frames": c.frames,
                    "duration_sec": c.duration_sec,
                    "infer_ms": c.infer_ms,
                }
                for c in self.chunk_timings
            ],
        }


class ResourceMonitor:
    def __init__(self, sample_interval: float = 0.25):
        self.sample_interval = sample_interval
        self.cpu_samples: list[float] = []
        self.gpu_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[Any] = None
        self._nvml_handle: Optional[Any] = None
        self._nvml_get_util: Optional[Callable[[Any], Any]] = None
        self._nvml_shutdown: Optional[Callable[[], None]] = None

    @staticmethod
    def start(sample_interval: float = 0.25) -> Optional["ResourceMonitor"]:
        try:
            import psutil
        except Exception:
            return None

        monitor = ResourceMonitor(sample_interval=sample_interval)
        monitor._process = psutil.Process()
        monitor._process.cpu_percent(None)

        try:
            from pynvml import (
                nvmlDeviceGetHandleByIndex,
                nvmlDeviceGetUtilizationRates,
                nvmlInit,
                nvmlShutdown,
            )

            nvmlInit()
            monitor._nvml_handle = nvmlDeviceGetHandleByIndex(0)
            monitor._nvml_get_util = nvmlDeviceGetUtilizationRates
            monitor._nvml_shutdown = nvmlShutdown
        except Exception:
            monitor._nvml_handle = None
            monitor._nvml_get_util = None
            monitor._nvml_shutdown = None

        monitor._thread = threading.Thread(target=monitor._run, daemon=True)
        monitor._thread.start()
        return monitor

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._process is None:
                time.sleep(self.sample_interval)
                continue
            self.cpu_samples.append(
                float(self._process.cpu_percent(interval=self.sample_interval))
            )
            if self._nvml_handle and self._nvml_get_util:
                try:
                    util = self._nvml_get_util(self._nvml_handle)
                    self.gpu_samples.append(float(util.gpu))
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._nvml_shutdown:
            try:
                self._nvml_shutdown()
            except Exception:
                pass

    def summary(self) -> dict[str, Any]:
        def _stats(xs: list[float]) -> dict[str, float]:
            if not xs:
                return {"avg": 0.0, "max": 0.0}
            return {"avg": float(np.mean(xs)), "max": float(np.max(xs))}

        return {
            "cpu_percent": _stats(self.cpu_samples),
            "gpu_percent": _stats(self.gpu_samples),
        }


def run_streaming_benchmark(
    audio: np.ndarray,
    sample_rate: int,
    infer_fn: InferFn,
    *,
    chunk_sec: float,
    overlap_sec: float,
    target_sr: int = 44100,
    measure_resources: bool = False,
    resource_interval: float = 0.25,
    fallback_on_error: bool = False,
) -> tuple[np.ndarray, BenchmarkReport]:
    audio = np.asarray(audio, dtype=np.float32)
    audio = _as_stereo(audio)
    audio = _resample(audio, sr_in=int(sample_rate), sr_out=int(target_sr))

    chunk_len = int(round(chunk_sec * target_sr))
    overlap = int(round(overlap_sec * target_sr))
    hop = chunk_len - overlap
    if chunk_len <= 0:
        raise ValueError("chunk-sec must be > 0")
    if hop <= 0:
        raise ValueError("chunk-sec must be greater than overlap-sec")

    total_len = int(audio.shape[0])
    if total_len == 0:
        return np.zeros((0, 2), dtype=np.float32), BenchmarkReport(
            chunk_timings=[],
            total_elapsed_sec=0.0,
            realtime_factor=None,
            throughput_x=None,
            estimated_latency_sec=chunk_sec,
            hop_sec=float(hop) / float(target_sr),
            resources=None,
            meta={
                "input_sample_rate": int(sample_rate),
                "target_sample_rate": int(target_sr),
                "output_sample_rate": int(target_sr),
                "fallback_on_error": bool(fallback_on_error),
                "failed_chunks": [],
                "failure_messages": [],
            },
            failed_chunks=[],
            failure_messages=[],
            fallback_on_error=fallback_on_error,
        )

    if total_len <= chunk_len:
        pad = chunk_len - total_len
    else:
        pad = (hop - (max(0, total_len - chunk_len) % hop)) % hop
    padded = np.pad(audio, ((0, pad), (0, 0)), mode="constant")

    monitor = (
        ResourceMonitor.start(sample_interval=resource_interval)
        if measure_resources
        else None
    )

    chunk_timings: list[ChunkTiming] = []
    chunks_out: list[np.ndarray] = []
    last_meta: dict[str, Any] = {}
    failed_chunks: list[int] = []
    failure_messages: list[str] = []

    t_total_start = time.perf_counter()
    for idx, start in enumerate(range(0, padded.shape[0] - chunk_len + 1, hop)):
        chunk = padded[start : start + chunk_len]
        t0 = time.perf_counter()
        try:
            out_chunk, last_meta = infer_fn(chunk)
        except Exception as e:
            if not fallback_on_error:
                raise
            failed_chunks.append(idx)
            failure_messages.append(str(e))
            # Fallback: bypass the chunk to keep continuity for analysis.
            out_chunk = chunk
            last_meta = {
                "error": str(e),
                "backend": "fallback-bypass",
            }
        infer_ms = (time.perf_counter() - t0) * 1000.0
        chunks_out.append(out_chunk)
        chunk_timings.append(
            ChunkTiming(
                index=idx,
                frames=int(chunk.shape[0]),
                duration_sec=float(chunk.shape[0]) / float(target_sr),
                infer_ms=infer_ms,
            )
        )
    total_elapsed = time.perf_counter() - t_total_start

    if monitor:
        monitor.stop()
        resources = monitor.summary()
    else:
        resources = None

    out = _overlap_add(chunks_out, hop=hop, overlap=overlap, total_len=padded.shape[0])
    out = out[:total_len]

    duration = float(total_len) / float(target_sr) if target_sr > 0 else 0.0
    rtf = (total_elapsed / duration) if duration > 0 else None
    throughput = (duration / total_elapsed) if total_elapsed > 0 else None

    meta_full = {
        **last_meta,
        "input_sample_rate": int(sample_rate),
        "target_sample_rate": int(target_sr),
        "output_sample_rate": int(target_sr),
        "fallback_on_error": bool(fallback_on_error),
        "failed_chunks": failed_chunks,
        "failure_messages": failure_messages,
    }

    report = BenchmarkReport(
        chunk_timings=chunk_timings,
        total_elapsed_sec=total_elapsed,
        realtime_factor=rtf,
        throughput_x=throughput,
        estimated_latency_sec=chunk_sec,
        hop_sec=float(hop) / float(target_sr),
        resources=resources,
        meta=meta_full,
        failed_chunks=failed_chunks,
        failure_messages=failure_messages,
        fallback_on_error=fallback_on_error,
    )

    return out, report


def run_streaming_benchmark_roundtrip(
    audio: np.ndarray,
    sample_rate: int,
    infer_fn: InferFn,
    *,
    chunk_sec: float,
    overlap_sec: float,
    target_sr: int = 44100,
    measure_resources: bool = False,
    resource_interval: float = 0.25,
    fallback_on_error: bool = False,
) -> tuple[np.ndarray, BenchmarkReport]:
    """Convenience wrapper that resamples back to input sample rate.

    This mirrors the real-time path: input SR -> target SR (inference) -> input SR.
    """
    out_target, report = run_streaming_benchmark(
        audio,
        sample_rate=sample_rate,
        infer_fn=infer_fn,
        chunk_sec=chunk_sec,
        overlap_sec=overlap_sec,
        target_sr=target_sr,
        measure_resources=measure_resources,
        resource_interval=resource_interval,
        fallback_on_error=fallback_on_error,
    )
    out_audio = _resample(out_target, sr_in=int(target_sr), sr_out=int(sample_rate))
    meta = dict(report.meta)
    meta["output_sample_rate"] = int(sample_rate)
    report.meta = meta
    return out_audio, report


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


def _build_onnx_infer(
    model_path: Path, provider: ExecutionProvider, intra_op_threads: int
) -> InferFn:
    try:
        import onnxruntime as ort
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "onnxruntime is not available. Install with: uv sync --extra onnxruntime"
        ) from e

    sess_options = ort.SessionOptions()
    if intra_op_threads > 0:
        sess_options.intra_op_num_threads = int(intra_op_threads)

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=_providers_for(provider),
    )
    input_name = session.get_inputs()[0].name

    def _infer(chunk: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        x: np.ndarray = chunk.T[None, :, :].astype(np.float32)  # (1, 2, T)
        outputs = session.run(None, {input_name: x})
        if not outputs:
            raise RuntimeError("onnxruntime returned no outputs")
        out = np.asarray(outputs[-1])
        if out.ndim == 3 and out.shape[0] == 1 and out.shape[1] == 2:
            y = out[0].T
        elif out.ndim == 2 and out.shape[0] == 2:
            y = out.T
        elif out.ndim == 2 and out.shape[1] == 2:
            y = out
        else:
            raise RuntimeError(f"Unsupported ONNX output shape: {out.shape}")
        meta = {"backend": "ort", "provider": provider, "model_path": str(model_path)}
        return y.astype(np.float32), meta

    return _infer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="De-limiter streaming benchmark (#1015)")
    p.add_argument("--input", type=Path, required=True, help="Input wav file (any SR)")
    p.add_argument(
        "--model", type=Path, required=True, help="Path to delimiter ONNX model"
    )
    p.add_argument("--provider", choices=["cpu", "cuda", "tensorrt"], default="cpu")
    p.add_argument("--chunk-sec", type=float, default=6.0, help="Chunk size in seconds")
    p.add_argument(
        "--overlap-sec", type=float, default=0.25, help="Crossfade overlap seconds"
    )
    p.add_argument(
        "--intra-op-threads", type=int, default=0, help="ORT intra-op threads"
    )
    p.add_argument(
        "--target-sr",
        type=int,
        default=44100,
        help="Sample rate expected by the model (default: 44100)",
    )
    p.add_argument(
        "--measure-resources", action="store_true", help="Sample CPU/GPU utilization"
    )
    p.add_argument(
        "--resource-interval",
        type=float,
        default=0.25,
        help="Sampling interval for CPU/GPU utilization (seconds)",
    )
    p.add_argument(
        "--fallback-on-error",
        action="store_true",
        help="Bypass chunk and continue when inference raises (for failure campaigns)",
    )
    p.add_argument("--report", type=Path, help="Write benchmark JSON report")
    p.add_argument(
        "--save-output",
        type=Path,
        help="Optional path to write processed audio (resampled back to input SR)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    audio, sr = sf.read(args.input)
    audio = np.asarray(audio, dtype=np.float32)
    audio = _as_stereo(audio)

    infer_fn = _build_onnx_infer(
        model_path=args.model,
        provider=cast(ExecutionProvider, args.provider),
        intra_op_threads=int(args.intra_op_threads),
    )

    out, report = run_streaming_benchmark_roundtrip(
        audio,
        sample_rate=int(sr),
        infer_fn=infer_fn,
        chunk_sec=float(args.chunk_sec),
        overlap_sec=float(args.overlap_sec),
        target_sr=int(args.target_sr),
        measure_resources=bool(args.measure_resources),
        resource_interval=float(args.resource_interval),
        fallback_on_error=bool(args.fallback_on_error),
    )

    metrics = report.to_dict()["metrics"]
    throughput_str = (
        f"{report.throughput_x:.3f}" if report.throughput_x is not None else "n/a"
    )
    rtf_str = (
        f"{report.realtime_factor:.3f}" if report.realtime_factor is not None else "n/a"
    )
    print(
        f"[perf] chunks={len(report.chunk_timings)} "
        f"mean={metrics['mean_ms_per_chunk']:.2f}ms "
        f"p95={metrics['p95_ms_per_chunk']:.2f}ms "
        f"throughput_x={throughput_str} "
        f"rtf={rtf_str}"
    )
    if metrics.get("error_rate", 0.0) > 0.0:
        print(
            f"[warn] fallback_on_error={args.fallback_on_error} "
            f"errors={len(report.failed_chunks)} "
            f"indices={report.failed_chunks}"
        )
    if report.resources:
        cpu = report.resources.get("cpu_percent", {})
        gpu = report.resources.get("gpu_percent", {})
        print(
            f"[resources] cpu_avg={cpu.get('avg', 0.0):.1f}% cpu_max={cpu.get('max', 0.0):.1f}% "
            f"gpu_avg={gpu.get('avg', 0.0):.1f}% gpu_max={gpu.get('max', 0.0):.1f}%"
        )
    else:
        print("[resources] not collected (install psutil / nvidia-ml-py3 to enable)")

    if args.save_output:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.save_output, out, int(sr))
        print(f"[output] wrote {args.save_output}")

    if args.report:
        report_dict = {
            "input": {
                "path": str(args.input),
                "sample_rate": int(sr),
                "duration_sec": float(len(audio) / sr) if sr else 0.0,
            },
            "params": {
                "chunk_sec": float(args.chunk_sec),
                "overlap_sec": float(args.overlap_sec),
                "provider": args.provider,
                "intra_op_threads": int(args.intra_op_threads),
                "target_sr": int(args.target_sr),
                "fallback_on_error": bool(args.fallback_on_error),
            },
            "report": report.to_dict(),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        print(f"[report] wrote {args.report}")


if __name__ == "__main__":  # pragma: no cover
    main()
