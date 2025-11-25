"""Tests for PipeWire/PulseAudio sink management.

Tests the functions in web/services/pipewire.py that handle:
- GPU sink creation and management
- Default sink settings
- PipeWire link configuration
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


from web.services.pipewire import (
    create_gpu_sink,
    get_default_sink,
    remember_default_sink,
    restore_default_sink,
    select_fallback_sink,
    setup_audio_routing,
    setup_pipewire_links,
    sink_exists,
    wait_for_daemon_node,
)


class TestGetDefaultSink:
    """Tests for get_default_sink()."""

    def test_returns_sink_name_from_pactl_info(self):
        """Test parsing default sink from pactl info output."""
        mock_output = """Server String: /run/user/1000/pulse/native
Server Name: PulseAudio (on PipeWire 1.0.0)
Default Sink: alsa_output.usb-SMSL_SMSL_USB_AUDIO-00.analog-stereo
Default Source: alsa_input.pci-0000_00_1f.3.analog-stereo"""

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_output, returncode=0)
            result = get_default_sink()
            assert result == "alsa_output.usb-SMSL_SMSL_USB_AUDIO-00.analog-stereo"

    def test_returns_none_when_pactl_fails(self):
        """Test returns None when pactl command fails."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_default_sink()
            assert result is None

    def test_returns_none_when_no_default_sink_line(self):
        """Test returns None when output doesn't contain Default Sink."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Server Name: test", returncode=0)
            result = get_default_sink()
            assert result is None


class TestSinkExists:
    """Tests for sink_exists()."""

    def test_returns_true_when_sink_found(self):
        """Test returns True when sink is in the list."""
        mock_output = """1\talsa_output.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\ts32le 2ch 48000Hz
2\tgpu_upsampler_sink\tmodule-null-sink.c\ts32le 2ch 44100Hz"""

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_output, returncode=0)
            assert sink_exists("gpu_upsampler_sink") is True

    def test_returns_false_when_sink_not_found(self):
        """Test returns False when sink is not in the list."""
        mock_output = (
            "1\talsa_output.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c"
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_output, returncode=0)
            assert sink_exists("gpu_upsampler_sink") is False

    def test_returns_false_on_command_failure(self):
        """Test returns False when pactl command fails."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert sink_exists("any_sink") is False


class TestSelectFallbackSink:
    """Tests for select_fallback_sink()."""

    def test_returns_first_non_gpu_sink(self):
        """Test returns first sink that is not gpu_upsampler_sink."""
        mock_output = """1\tgpu_upsampler_sink\tmodule-null-sink.c
2\talsa_output.usb-DAC-00\tmodule-alsa-card.c
3\talsa_output.pci-00\tmodule-alsa-card.c"""

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_output, returncode=0)
            result = select_fallback_sink()
            assert result == "alsa_output.usb-DAC-00"

    def test_returns_none_when_only_gpu_sink(self):
        """Test returns None when only gpu_upsampler_sink exists."""
        mock_output = "1\tgpu_upsampler_sink\tmodule-null-sink.c"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_output, returncode=0)
            result = select_fallback_sink()
            assert result is None


class TestCreateGpuSink:
    """Tests for create_gpu_sink()."""

    def test_returns_true_when_sink_already_exists(self):
        """Test returns True immediately if sink already exists."""
        with patch("web.services.pipewire.sink_exists", return_value=True):
            result = create_gpu_sink()
            assert result is True

    def test_creates_sink_when_not_exists(self):
        """Test creates sink via pactl when it doesn't exist."""
        with (
            patch("web.services.pipewire.sink_exists", return_value=False),
            patch("subprocess.run") as mock_run,
            patch("time.sleep"),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = create_gpu_sink()
            assert result is True
            # Verify pactl load-module was called
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "pactl" in args
            assert "load-module" in args
            assert "module-null-sink" in args

    def test_returns_false_on_creation_failure(self):
        """Test returns False when pactl fails."""
        with (
            patch("web.services.pipewire.sink_exists", return_value=False),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="Module load failed")
            result = create_gpu_sink()
            assert result is False


class TestWaitForDaemonNode:
    """Tests for wait_for_daemon_node()."""

    def test_returns_true_when_node_found_immediately(self):
        """Test returns True when node is found on first check."""
        mock_output = """GPU Upsampler Input:input_FL
GPU Upsampler Input:input_FR"""

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_output, returncode=0)
            result = wait_for_daemon_node(timeout_sec=0)
            assert result is True

    def test_returns_false_when_node_not_found(self):
        """Test returns False when node doesn't appear within timeout."""
        with (
            patch("subprocess.run") as mock_run,
            patch("time.sleep"),
            patch("time.time", side_effect=[0, 0.1, 0.5, 1.0]),
        ):
            mock_run.return_value = MagicMock(
                stdout="other_node:input_FL", returncode=0
            )
            result = wait_for_daemon_node(timeout_sec=0.5)
            assert result is False

    def test_immediate_check_with_zero_timeout(self):
        """Test timeout=0 performs exactly one check."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            result = wait_for_daemon_node(timeout_sec=0)
            assert result is False
            # Should be called exactly once for immediate check
            assert mock_run.call_count == 1


class TestSetupPipewireLinks:
    """Tests for setup_pipewire_links()."""

    def test_creates_fl_and_fr_links(self):
        """Test creates both FL and FR channel links."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            success, msg = setup_pipewire_links()
            assert success is True
            assert "configured" in msg.lower()
            # Should be called twice (FL and FR)
            assert mock_run.call_count == 2

    def test_ignores_already_linked_error(self):
        """Test treats 'already linked' as success."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="Error: already linked"
            )
            success, msg = setup_pipewire_links()
            assert success is True

    def test_returns_false_on_real_error(self):
        """Test returns False on actual linking error."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="Error: Node not found"
            )
            success, msg = setup_pipewire_links()
            assert success is False
            assert "Node not found" in msg


class TestSetupAudioRouting:
    """Tests for setup_audio_routing()."""

    def test_complete_setup_flow(self):
        """Test complete audio routing setup succeeds."""
        with (
            patch("web.services.pipewire.create_gpu_sink", return_value=True),
            patch("web.services.pipewire.remember_default_sink"),
            patch("web.services.pipewire.set_default_sink", return_value=True),
            patch(
                "web.services.pipewire.get_default_sink",
                return_value="gpu_upsampler_sink",
            ),
        ):
            success, msg = setup_audio_routing()
            assert success is True
            assert "gpu_upsampler_sink" in msg

    def test_fails_when_sink_creation_fails(self):
        """Test returns failure when sink creation fails."""
        with patch("web.services.pipewire.create_gpu_sink", return_value=False):
            success, msg = setup_audio_routing()
            assert success is False
            assert "Failed to create" in msg

    def test_fails_when_set_default_fails(self):
        """Test returns failure when setting default sink fails."""
        with (
            patch("web.services.pipewire.create_gpu_sink", return_value=True),
            patch("web.services.pipewire.remember_default_sink"),
            patch("web.services.pipewire.set_default_sink", return_value=False),
        ):
            success, msg = setup_audio_routing()
            assert success is False
            assert "Failed to set" in msg


class TestRememberDefaultSink:
    """Tests for remember_default_sink()."""

    def test_saves_current_sink_to_file(self, tmp_path: Path):
        """Test saves non-GPU sink to file."""
        sink_file = tmp_path / "default_sink"

        with (
            patch("web.services.pipewire.get_default_sink", return_value="my_dac_sink"),
            patch("web.services.pipewire.DEFAULT_SINK_FILE_PATH", sink_file),
        ):
            remember_default_sink()
            assert sink_file.read_text() == "my_dac_sink"

    def test_saves_fallback_when_gpu_is_default(self, tmp_path: Path):
        """Test saves fallback sink when GPU sink is already default."""
        sink_file = tmp_path / "default_sink"

        with (
            patch(
                "web.services.pipewire.get_default_sink",
                return_value="gpu_upsampler_sink",
            ),
            patch(
                "web.services.pipewire.select_fallback_sink",
                return_value="fallback_sink",
            ),
            patch("web.services.pipewire.DEFAULT_SINK_FILE_PATH", sink_file),
        ):
            remember_default_sink()
            assert sink_file.read_text() == "fallback_sink"


class TestRestoreDefaultSink:
    """Tests for restore_default_sink()."""

    def test_restores_from_file(self, tmp_path: Path):
        """Test restores sink from saved file."""
        sink_file = tmp_path / "default_sink"
        sink_file.write_text("my_dac_sink")

        with (
            patch("web.services.pipewire.DEFAULT_SINK_FILE_PATH", sink_file),
            patch(
                "web.services.pipewire.set_default_sink", return_value=True
            ) as mock_set,
        ):
            restore_default_sink()
            mock_set.assert_called_once_with("my_dac_sink")

    def test_uses_fallback_when_no_file(self, tmp_path: Path):
        """Test uses fallback sink when no saved file exists."""
        sink_file = tmp_path / "default_sink"  # Does not exist

        with (
            patch("web.services.pipewire.DEFAULT_SINK_FILE_PATH", sink_file),
            patch(
                "web.services.pipewire.select_fallback_sink",
                return_value="fallback_sink",
            ),
            patch(
                "web.services.pipewire.set_default_sink", return_value=True
            ) as mock_set,
        ):
            restore_default_sink()
            mock_set.assert_called_once_with("fallback_sink")
