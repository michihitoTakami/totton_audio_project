from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "delimiter" / "offline_wav_to_wav.py"


def test_delimiter_offline_poc_bypass_smoke(
    tmp_path: Path, sample_rate_48k: int
) -> None:
    input_wav = tmp_path / "input.wav"
    output_wav = tmp_path / "output.wav"
    report = tmp_path / "report.json"
    debug_dir = tmp_path / "debug"

    duration_sec = 0.25
    t = (
        np.arange(int(sample_rate_48k * duration_sec), dtype=np.float64)
        / sample_rate_48k
    )
    audio = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(input_wav, audio, sample_rate_48k)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_wav),
            "--output",
            str(output_wav),
            "--backend",
            "bypass",
            "--chunk-sec",
            "6.0",
            "--overlap-sec",
            "0.25",
            "--resample-back",
            "--debug-dir",
            str(debug_dir),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    assert output_wav.exists()
    out_audio, out_sr = sf.read(output_wav)
    assert out_sr == sample_rate_48k
    assert out_audio.ndim == 2 and out_audio.shape[1] == 2

    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["meta"]["backend"] == "bypass"
    assert payload["meta"]["expected_sample_rate"] == 44100
    assert payload["debug"] is not None
    assert Path(payload["debug"]["waveform_png"]).exists()
    assert Path(payload["debug"]["ab_wav"]).exists()
    assert "lufs" in payload["input"]["stats"]
    assert "lufs" in payload["output"]["stats"]

    assert "[input]" in result.stdout
    assert "[output]" in result.stdout
    assert "[perf]" in result.stdout


def test_delimiter_offline_poc_delimiter_optional(
    tmp_path: Path, sample_rate_48k: int
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("asteroid")
    pytest.importorskip("asteroid_filterbanks")
    pytest.importorskip("einops")

    weights_dir = PROJECT_ROOT / "data" / "delimiter" / "weight"
    if (
        not (weights_dir / "all.json").exists()
        or not (weights_dir / "all.pth").exists()
    ):
        pytest.skip(
            "De-limiter weights are not present (download locally for this optional test)"
        )

    input_wav = tmp_path / "input.wav"
    output_wav = tmp_path / "output.wav"

    duration_sec = 0.25
    t = (
        np.arange(int(sample_rate_48k * duration_sec), dtype=np.float64)
        / sample_rate_48k
    )
    audio = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(input_wav, audio, sample_rate_48k)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_wav),
            "--output",
            str(output_wav),
            "--backend",
            "delimiter",
            "--chunk-sec",
            "0",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    assert output_wav.exists()
