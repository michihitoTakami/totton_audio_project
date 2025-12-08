"""
E2E/回帰テスト: レート自動交渉がコントロールプレーン経由で正しく露出されることを検証 (Issue #221)
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.main import app
from web.models import Settings
from web.services import daemon
import web.routers.status as status_router


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def stats_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "stats.json"
    monkeypatch.setattr(daemon, "STATS_FILE_PATH", path)
    return path


def test_get_configured_rates_reads_stats_file(stats_path: Path):
    stats_path.write_text(json.dumps({"input_rate": 96000, "output_rate": 768000}))
    input_rate, output_rate = daemon.get_configured_rates()
    assert input_rate == 96000
    assert output_rate == 768000


def test_get_configured_rates_returns_zero_when_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(daemon, "STATS_FILE_PATH", missing)
    assert daemon.get_configured_rates() == (0, 0)


def test_load_stats_computes_clip_rate_and_peaks(stats_path: Path):
    stats_path.write_text(
        json.dumps(
            {
                "clip_count": 50,
                "total_samples": 1000,
                "input_rate": 88200,
                "output_rate": 705600,
                "peaks": {
                    "input": {"linear": 0.8, "dbfs": -1.0},
                    "upsampler": {"linear": 0.9, "dbfs": -0.5},
                },
            }
        )
    )

    stats = daemon.load_stats()
    assert stats["clip_rate"] == pytest.approx(0.05)
    assert stats["input_rate"] == 88200
    assert stats["output_rate"] == 705600
    assert stats["peaks"]["input"]["linear"] == 0.8
    assert stats["peaks"]["upsampler"]["linear"] == 0.9
    # default fallback for missing stages
    assert stats["peaks"]["post_gain"]["dbfs"] == -200.0


def test_status_endpoint_surfaces_negotiated_rates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    sample_stats = {
        "clip_rate": 0.0,
        "clip_count": 0,
        "total_samples": 0,
        "input_rate": 176400,
        "output_rate": 705600,
        "peaks": daemon.load_stats()["peaks"],
    }

    monkeypatch.setattr(status_router, "load_config", lambda: Settings())
    monkeypatch.setattr(status_router, "check_daemon_running", lambda: True)
    monkeypatch.setattr(status_router, "load_stats", lambda: sample_stats)

    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()

    assert data["daemon_running"] is True
    assert data["input_rate"] == 176400
    assert data["output_rate"] == 705600
