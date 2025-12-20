import numpy as np
import pytest

from scripts.delimiter.benchmark_streaming import run_streaming_benchmark


def _dummy_infer(chunk: np.ndarray):
    return chunk * 0.5, {"backend": "dummy"}


def test_run_streaming_benchmark_basic():
    sr = 44100
    duration = 0.3
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    audio = np.stack(
        [np.sin(2 * np.pi * 440.0 * t), np.sin(2 * np.pi * 440.0 * t)], axis=1
    )

    out, report = run_streaming_benchmark(
        audio,
        sample_rate=sr,
        infer_fn=_dummy_infer,
        chunk_sec=0.1,
        overlap_sec=0.02,
        target_sr=sr,
        measure_resources=False,
    )

    assert out.shape == audio.shape
    assert len(report.chunk_timings) >= 2

    metrics = report.to_dict()["metrics"]
    assert metrics["chunk_count"] == len(report.chunk_timings)
    assert metrics["mean_ms_per_chunk"] >= 0.0
    assert pytest.approx(metrics["steady_state_hop_sec"], rel=1e-3) == 0.08
    assert report.estimated_latency_sec == pytest.approx(0.1, rel=1e-3)
    assert report.meta.get("backend") == "dummy"


def test_run_streaming_benchmark_invalid_overlap():
    with pytest.raises(ValueError):
        run_streaming_benchmark(
            np.zeros((100, 2), dtype=np.float32),
            sample_rate=44100,
            infer_fn=_dummy_infer,
            chunk_sec=0.01,
            overlap_sec=0.02,
        )
