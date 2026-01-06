"""Unit tests for raspberry_pi.usb_i2s_bridge (Issue #823)."""

from __future__ import annotations

from raspberry_pi.usb_i2s_bridge import bridge


def test_parse_hw_params_rate_returns_value() -> None:
    content = "access: RW_INTERLEAVED\nrate: 44100 (44100/1)\n"
    assert bridge._parse_hw_params_rate(content) == 44100


def test_parse_hw_params_rate_tolerates_leading_spaces() -> None:
    content = "access: RW_INTERLEAVED\n   rate: 44100 (44100/1)\n"
    assert bridge._parse_hw_params_rate(content) == 44100


def test_parse_hw_params_rate_closed_is_none() -> None:
    assert bridge._parse_hw_params_rate("closed\nrate: 48000") is None


def test_parse_hw_params_format_returns_value() -> None:
    content = "access: RW_INTERLEAVED\nformat: S24_3LE\nrate: 48000\n"
    assert bridge._parse_hw_params_format(content) == "S24_3LE"


def test_parse_hw_params_format_tolerates_leading_spaces() -> None:
    content = "access: RW_INTERLEAVED\n  format: S32_LE\nrate: 48000\n"
    assert bridge._parse_hw_params_format(content) == "S32_LE"


def test_parse_hw_params_format_closed_is_none() -> None:
    assert bridge._parse_hw_params_format("closed\nformat: S32_LE") is None


def test_pcm_device_node_mapping_capture() -> None:
    assert str(bridge._pcm_device_node("hw:2,0", "c")) == "/dev/snd/pcmC2D0c"


def test_pcm_device_node_mapping_playback() -> None:
    assert str(bridge._pcm_device_node("plughw:0,0", "p")) == "/dev/snd/pcmC0D0p"


def test_default_control_endpoint_is_disabled_by_default() -> None:
    # NOTE(#950): 60100/60101 の制御プレーンはデフォルト無効（空）へ寄せる
    assert bridge._DEFAULT_CONTROL_ENDPOINT == ""
