"""Unit tests for raspberry_pi.usb_i2s_bridge (Issue #823)."""

from __future__ import annotations

from raspberry_pi.usb_i2s_bridge import bridge


def test_parse_hw_params_rate_returns_value() -> None:
    content = "access: RW_INTERLEAVED\nrate: 44100 (44100/1)\n"
    assert bridge._parse_hw_params_rate(content) == 44100


def test_parse_hw_params_rate_closed_is_none() -> None:
    assert bridge._parse_hw_params_rate("closed\nrate: 48000") is None


def test_parse_hw_params_format_returns_value() -> None:
    content = "access: RW_INTERLEAVED\nformat: S24_3LE\nrate: 48000\n"
    assert bridge._parse_hw_params_format(content) == "S24_3LE"


def test_parse_hw_params_format_closed_is_none() -> None:
    assert bridge._parse_hw_params_format("closed\nformat: S32_LE") is None


def test_gst_raw_format_from_alsa_prefers_actual() -> None:
    assert bridge._gst_raw_format_from_alsa("S24_3LE", "S32_LE") == "S24LE"


def test_gst_raw_format_from_alsa_falls_back_to_preferred() -> None:
    assert bridge._gst_raw_format_from_alsa(None, "S32_LE") == "S32LE"


def test_build_gst_launch_command_contains_devices() -> None:
    cfg = bridge.UsbI2sBridgeConfig(
        capture_device="hw:2,0",
        playback_device="hw:0,0",
        channels=2,
        fallback_rate=48000,
        preferred_format="S32_LE",
    )
    cmd = bridge.build_gst_launch_command(
        cfg, mode="capture", sample_rate=48000, raw_format="S32LE", conversion=False
    )
    assert "gst-launch-1.0" in cmd[0]
    assert any(part == "alsasrc" for part in cmd)
    assert any(part == "alsasink" for part in cmd)
    assert any(part == "device=hw:2,0" for part in cmd)
    assert any(part == "device=hw:0,0" for part in cmd)


def test_build_gst_launch_command_includes_convert_when_enabled() -> None:
    cfg = bridge.UsbI2sBridgeConfig(
        capture_device="hw:2,0",
        playback_device="hw:0,0",
        channels=2,
        fallback_rate=48000,
        preferred_format="S32_LE",
    )
    cmd = bridge.build_gst_launch_command(
        cfg, mode="capture", sample_rate=48000, raw_format="S32LE", conversion=True
    )
    assert "audioresample" in cmd
    assert "audioconvert" in cmd


def test_pcm_device_node_mapping_capture() -> None:
    assert str(bridge._pcm_device_node("hw:2,0", "c")) == "/dev/snd/pcmC2D0c"


def test_pcm_device_node_mapping_playback() -> None:
    assert str(bridge._pcm_device_node("plughw:0,0", "p")) == "/dev/snd/pcmC0D0p"
