"""Unit tests for raspberry_pi.rtp_sender."""

from __future__ import annotations

import pytest

from raspberry_pi import rtp_sender


def test_build_gst_command_default_values():
    cfg = rtp_sender.RtpSenderConfig()
    args = rtp_sender.build_gst_command(cfg)

    assert args[0] == "gst-launch-1.0"
    assert "audioresample" in args
    assert "rtpbin.send_rtp_sink_0" in args
    # Ensure RTP/RTCP ports are wired
    assert f"port={cfg.rtp_port}" in args
    assert f"port={cfg.rtcp_port}" in args
    assert f"port={cfg.rtcp_listen_port}" in args


def test_format_mapping_variants():
    cfg_16 = rtp_sender.RtpSenderConfig(audio_format="S16_LE")
    args_16 = rtp_sender.build_gst_command(cfg_16)
    assert "rtpL16pay" in args_16
    assert any("format=S16LE" in a for a in args_16)

    cfg_32 = rtp_sender.RtpSenderConfig(audio_format="S32_LE")
    args_32 = rtp_sender.build_gst_command(cfg_32)
    assert "rtpL32pay" in args_32
    assert any("format=S32LE" in a for a in args_32)


def test_invalid_format_raises():
    cfg = rtp_sender.RtpSenderConfig(audio_format="PCM_S24")
    with pytest.raises(ValueError):
        rtp_sender.build_gst_command(cfg)


def test_command_string_roundtrip():
    cfg = rtp_sender.RtpSenderConfig(latency_ms=150, payload_type=97)
    args = rtp_sender.build_gst_command(cfg)
    cmd_str = rtp_sender.command_to_string(args)

    assert "latency=150" in cmd_str
    assert "payload=97" in cmd_str
