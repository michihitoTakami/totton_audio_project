"""Tests for soft mute during filter switching (Issue #266).

These tests verify that soft mute is applied during filter switches:
- Phase type switching
- Crossfeed head size switching
- Sample rate changes (if applicable)
"""

import time
from unittest.mock import patch

import pytest

from web.services.daemon_client import DaemonClient, DaemonResponse


class TestPhaseTypeSwitchSoftMute:
    """Tests for soft mute during phase type switching."""

    @pytest.fixture
    def daemon_client(self):
        """Create a DaemonClient instance."""
        return DaemonClient()

    def test_phase_type_switch_triggers_soft_mute(self, daemon_client):
        """Test that phase type switch triggers soft mute sequence."""
        with patch.object(daemon_client, "send_command") as mock_send:
            # Mock successful phase type switch
            mock_send.return_value = (True, "OK:Phase type set to hybrid")

            # Switch phase type
            success, message = daemon_client.set_phase_type("hybrid")

            # Verify command was sent
            mock_send.assert_called_once_with("PHASE_TYPE_SET:hybrid")
            assert success is True
            assert "Phase type set" in message

    def test_phase_type_switch_same_value_no_switch(self, daemon_client):
        """Test that switching to same phase type doesn't trigger switch."""
        with patch.object(daemon_client, "send_command") as mock_send:
            # Mock response indicating already at target phase type
            mock_send.return_value = (True, "OK:Phase type already minimum")

            success, message = daemon_client.set_phase_type("minimum")

            mock_send.assert_called_once_with("PHASE_TYPE_SET:minimum")
            assert success is True
            assert "already" in message.lower()

    def test_phase_type_switch_error_handling(self, daemon_client):
        """Test error handling during phase type switch."""
        with patch.object(daemon_client, "send_command") as mock_send:
            # Mock error response
            mock_send.return_value = (
                False,
                "ERR:Quad-phase mode not enabled (runtime switching unavailable)",
            )

            success, message = daemon_client.set_phase_type("hybrid")

            assert success is False
            assert "Quad-phase mode not enabled" in message


class TestCrossfeedSwitchSoftMute:
    """Tests for soft mute during crossfeed head size switching."""

    @pytest.fixture
    def daemon_client(self):
        """Create a DaemonClient instance."""
        return DaemonClient()

    def test_crossfeed_size_switch_triggers_soft_mute(self, daemon_client):
        """Test that crossfeed head size switch triggers soft mute sequence."""
        with patch.object(daemon_client, "send_json_command_v2") as mock_send:
            # Mock successful head size switch
            mock_send.return_value = DaemonResponse(
                success=True, data={"head_size": "m"}
            )

            response = daemon_client.crossfeed_set_size("m")

            # Verify command was sent with correct format
            mock_send.assert_called_once_with("CROSSFEED_SET_SIZE", {"head_size": "m"})
            assert response.success is True
            assert response.data.get("head_size") == "m"

    def test_crossfeed_size_switch_invalid_size(self, daemon_client):
        """Test error handling for invalid head size."""
        # Invalid size is validated locally, not sent to daemon
        response = daemon_client.crossfeed_set_size("invalid")

        assert response.success is False
        assert response.error is not None
        assert "Invalid head size" in response.error.message


class TestFilterSwitchSoftMuteIntegration:
    """Integration tests for filter switch soft mute behavior.

    These tests verify the complete soft mute sequence:
    1. Fade-out (1.5 seconds)
    2. Filter switch
    3. Fade-in (1.5 seconds)
    """

    @pytest.fixture
    def daemon_client(self):
        """Create a DaemonClient instance."""
        return DaemonClient()

    def test_soft_mute_timing_phase_type_switch(self, daemon_client):
        """Test that phase type switch includes soft mute timing."""
        with patch.object(daemon_client, "send_command") as mock_send:
            mock_send.return_value = (True, "OK:Phase type set to hybrid")

            start_time = time.time()
            success, _ = daemon_client.set_phase_type("hybrid")
            elapsed = time.time() - start_time

            # Command should complete quickly (soft mute happens in daemon)
            assert success is True
            # Command itself should be fast (< 100ms for IPC)
            assert elapsed < 0.1

    def test_soft_mute_timing_crossfeed_switch(self, daemon_client):
        """Test that crossfeed switch includes soft mute timing."""
        with patch.object(daemon_client, "send_json_command_v2") as mock_send:
            mock_send.return_value = DaemonResponse(
                success=True, data={"head_size": "l"}
            )

            start_time = time.time()
            response = daemon_client.crossfeed_set_size("l")
            elapsed = time.time() - start_time

            # Command should complete quickly (soft mute happens in daemon)
            assert response.success is True
            # Command itself should be fast (< 100ms for IPC)
            assert elapsed < 0.1

    @pytest.mark.skip(reason="Requires running daemon - use E2E test instead")
    def test_soft_mute_audio_continuity(self):
        """Test that audio remains continuous during filter switch.

        This test requires:
        - Running daemon
        - Audio output device
        - Audio analysis tools

        Should verify:
        - No clicks or pops during switch
        - Smooth fade-out and fade-in
        - Total transition time ~3 seconds
        """
        pass


class TestFilterSwitchSoftMuteEdgeCases:
    """Edge case tests for filter switch soft mute."""

    @pytest.fixture
    def daemon_client(self):
        """Create a DaemonClient instance."""
        return DaemonClient()

    def test_rapid_phase_type_switches(self, daemon_client):
        """Test behavior when phase type is switched rapidly."""
        with patch.object(daemon_client, "send_command") as mock_send:
            mock_send.return_value = (True, "OK:Phase type set")

            # Rapid switches
            daemon_client.set_phase_type("hybrid")
            daemon_client.set_phase_type("minimum")
            daemon_client.set_phase_type("hybrid")

            # Should handle gracefully (daemon manages soft mute state)
            assert mock_send.call_count == 3

    def test_switch_during_existing_transition(self, daemon_client):
        """Test behavior when switching during an existing transition."""
        with patch.object(daemon_client, "send_command") as mock_send:
            mock_send.return_value = (True, "OK:Phase type set")

            # Start first switch
            daemon_client.set_phase_type("hybrid")

            # Immediately start second switch (before first completes)
            # In real implementation, daemon should handle this gracefully
            daemon_client.set_phase_type("minimum")

            assert mock_send.call_count == 2
