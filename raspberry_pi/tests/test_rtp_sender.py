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


def test_build_gst_command_includes_stability_buffers_by_default() -> None:
    cfg = rtp_sender.RtpSenderConfig(
        device="hw:0,0",
        host="192.168.55.1",
        sample_rate=44100,
        channels=2,
        audio_format="S24_3BE",
        payload_type=96,
        rtp_port=46000,
        rtcp_port=46001,
        rtcp_listen_port=46002,
    )
    cmd = rtp_sender.build_gst_command(cfg)

    # rtpbin: clock sync stabilization
    assert "rtpbin" in cmd
    assert "buffer-mode=synced" in cmd

    # alsasrc: explicit buffering to avoid period/avail brinkmanship
    assert any(part.startswith("buffer-time=") for part in cmd)
    assert any(part.startswith("latency-time=") for part in cmd)
    assert "do-timestamp=true" in cmd

    # queue: absorb conversion/resample jitter
    assert "queue" in cmd
    assert any(part.startswith("max-size-time=") for part in cmd)

    # udpsink: enlarge kernel send buffer
    assert sum(1 for part in cmd if part.startswith("buffer-size=")) >= 2


def test_build_gst_command_respects_overrides() -> None:
    cfg = rtp_sender.RtpSenderConfig(
        device="hw:0,0",
        host="192.168.55.1",
        sample_rate=48000,
        channels=2,
        audio_format="S24_3BE",
        payload_type=96,
        rtp_port=46000,
        rtcp_port=46001,
        rtcp_listen_port=46002,
        alsa_buffer_time_us=123_000,
        alsa_latency_time_us=7_000,
        queue_time_ns=42_000_000,
        udp_buffer_size_bytes=262_144,
    )
    cmd = rtp_sender.build_gst_command(cfg)
    assert f"buffer-time={cfg.alsa_buffer_time_us}" in cmd
    assert f"latency-time={cfg.alsa_latency_time_us}" in cmd
    assert f"max-size-time={cfg.queue_time_ns}" in cmd
    assert cmd.count(f"buffer-size={cfg.udp_buffer_size_bytes}") >= 2
