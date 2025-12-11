from __future__ import annotations

import json
import subprocess

from raspberry_pi import rtp_sender


def test_parse_hw_params_rate_extracts_value() -> None:
    content = "access: RW_INTERLEAVED\nrate: 48000 (48000/1)\n"
    assert rtp_sender._parse_hw_params_rate(content) == 48000


def test_parse_hw_params_rate_closed_returns_none() -> None:
    assert rtp_sender._parse_hw_params_rate("closed") is None


def test_parse_arecord_rate_prefers_last_value() -> None:
    stdout = "FORMAT S16_LE\nRATE 44100 48000\n"
    assert rtp_sender._parse_arecord_rate(stdout) == 48000


def test_detect_sample_rate_prefers_hw_params(monkeypatch) -> None:
    monkeypatch.setattr(rtp_sender, "_probe_hw_params_rate", lambda device: 96000)
    monkeypatch.setattr(rtp_sender, "_probe_arecord_rate", lambda device, ch: 44100)
    assert rtp_sender._detect_sample_rate("hw:0,0", 2, 44100) == 96000


def test_detect_sample_rate_fallback(monkeypatch) -> None:
    monkeypatch.setattr(rtp_sender, "_probe_hw_params_rate", lambda device: None)
    monkeypatch.setattr(rtp_sender, "_probe_arecord_rate", lambda device, ch: None)
    assert rtp_sender._detect_sample_rate("hw:0,0", 2, 44100) == 44100


def test_parse_arecord_rate_handles_no_numbers() -> None:
    stdout = "FORMAT S16_LE\nRATE foo bar\n"
    assert rtp_sender._parse_arecord_rate(stdout) is None


def test_parse_hw_params_rate_handles_closed() -> None:
    assert rtp_sender._parse_hw_params_rate("closed\nrate: 48000") is None


def test_auto_follow_restarts_on_rate_change(monkeypatch, tmp_path) -> None:
    stats_path = tmp_path / "stats.json"

    class FakeProc:
        def __init__(self, waits: list[object]) -> None:
            self._waits = waits
            self.terminated = False
            self.killed = False

        def wait(self, timeout=None):
            if not self._waits:
                return 0
            value = self._waits.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    procs = [
        FakeProc([subprocess.TimeoutExpired(cmd="x", timeout=1), 0]),
        FakeProc([0]),
    ]
    launch_calls: list[int] = []

    def _fake_launch(cfg):
        launch_calls.append(cfg.sample_rate)
        return procs.pop(0)

    rates = [44100, 48000]

    def _fake_detect(device, ch, fallback):
        return rates.pop(0)

    monkeypatch.setattr(rtp_sender, "_launch_pipeline", _fake_launch)
    monkeypatch.setattr(rtp_sender, "_detect_sample_rate", _fake_detect)
    monkeypatch.setattr(rtp_sender, "time", type("T", (), {"sleep": lambda _: None}))

    rtp_sender.main(
        [
            "--auto-sample-rate",
            "--rate-poll-interval",
            "0.1",
            "--stats-path",
            str(stats_path),
            "--sample-rate",
            "44100",
        ]
    )

    assert launch_calls == [44100, 48000]
    data = json.loads(stats_path.read_text())
    assert data["sample_rate"] == 48000


def test_auto_sample_rate_false_runs_once(monkeypatch) -> None:
    calls = {"launch": 0, "detect": 0}

    class FakeProc:
        def wait(self, timeout=None):
            return 0

    def _fake_launch(cfg):
        calls["launch"] += 1
        return FakeProc()

    def _fake_detect(device, ch, fallback):
        calls["detect"] += 1
        return fallback

    monkeypatch.setattr(rtp_sender, "_launch_pipeline", _fake_launch)
    monkeypatch.setattr(rtp_sender, "_detect_sample_rate", _fake_detect)

    rtp_sender.main(
        [
            "--no-auto-sample-rate",
            "--sample-rate",
            "44100",
        ]
    )

    assert calls["launch"] == 1
    assert calls["detect"] == 0
