from __future__ import annotations

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
