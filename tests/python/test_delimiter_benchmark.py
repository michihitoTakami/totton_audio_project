import numpy as np
import pytest

from scripts.delimiter.benchmark_streaming import (
    run_streaming_benchmark,
    run_streaming_benchmark_roundtrip,
)


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


def test_run_streaming_benchmark_resample_roundtrip(sample_rate_48k: int):
    duration = 0.12
    t = np.linspace(
        0.0, duration, int(sample_rate_48k * duration), endpoint=False, dtype=np.float32
    )
    audio = np.stack(
        [np.sin(2 * np.pi * 220.0 * t), np.sin(2 * np.pi * 220.0 * t)], axis=1
    )

    out, report = run_streaming_benchmark_roundtrip(
        audio,
        sample_rate=sample_rate_48k,
        infer_fn=_dummy_infer,
        chunk_sec=0.05,
        overlap_sec=0.01,
        target_sr=44100,
        measure_resources=False,
    )

    # Length should be preserved after resample -> inference -> resample-back.
    assert abs(out.shape[0] - audio.shape[0]) <= 1
    assert out.shape[1] == 2
    assert report.meta["input_sample_rate"] == sample_rate_48k
    assert report.meta["output_sample_rate"] == sample_rate_48k
    assert report.meta["target_sample_rate"] == 44100
    assert report.to_dict()["metrics"]["error_rate"] == 0.0


def test_run_streaming_benchmark_fallback_on_error():
    calls = {"count": 0}

    def _flaky_infer(chunk: np.ndarray):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("boom")
        return chunk * 0.5, {"backend": "dummy"}

    sr = 44100
    duration = 0.18
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    audio = np.stack(
        [np.sin(2 * np.pi * 330.0 * t), np.sin(2 * np.pi * 660.0 * t)], axis=1
    )

    out, report = run_streaming_benchmark(
        audio,
        sample_rate=sr,
        infer_fn=_flaky_infer,
        chunk_sec=0.05,
        overlap_sec=0.01,
        target_sr=sr,
        fallback_on_error=True,
    )

    assert out.shape == audio.shape
    assert report.failed_chunks == [0]
    metrics = report.to_dict()["metrics"]
    assert metrics["error_rate"] > 0.0
    assert metrics["error_chunks"] == [0]
    assert report.meta["fallback_on_error"] is True
