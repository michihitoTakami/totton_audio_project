"""Tests for partitioned convolution API endpoints (Issue #354)."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from web.main import app
from web.models import PartitionedConvolutionSettings

client = TestClient(app)


def _sample_settings() -> PartitionedConvolutionSettings:
    """Helper to build a valid settings object."""
    return PartitionedConvolutionSettings(
        enabled=True,
        fast_partition_taps=49152,
        min_partition_taps=8192,
        max_partitions=6,
        tail_fft_multiple=4,
    )


class TestPartitionedConvolutionGet:
    """GET /partitioned-convolution."""

    def test_returns_current_settings(self):
        """The endpoint should serialize the settings model."""
        with patch(
            "web.routers.partitioned.load_partitioned_convolution_settings",
            return_value=_sample_settings(),
        ) as mock_load:
            response = client.get("/partitioned-convolution")

        mock_load.assert_called_once()
        assert response.status_code == 200
        body = response.json()
        assert body["enabled"] is True
        assert body["fast_partition_taps"] == 49152
        assert body["tail_fft_multiple"] == 4


class TestPartitionedConvolutionPut:
    """PUT /partitioned-convolution."""

    def test_updates_settings_and_reports_restart(self):
        """Successful updates should return ApiResponse with restart flag."""
        payload = {
            "enabled": True,
            "fast_partition_taps": 65536,
            "min_partition_taps": 32768,
            "max_partitions": 8,
            "tail_fft_multiple": 6,
        }
        with patch(
            "web.routers.partitioned.save_partitioned_convolution_settings",
            return_value=True,
        ) as mock_save, patch(
            "web.routers.partitioned.check_daemon_running",
            return_value=True,
        ) as mock_running, patch(
            "web.routers.partitioned.save_phase_type",
            return_value=True,
        ) as mock_save_phase, patch(
            "web.routers.partitioned.get_daemon_client",
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.send_command_v2.return_value = MagicMock(success=True)
            mock_get_client.return_value = mock_client
            response = client.put("/partitioned-convolution", json=payload)

        mock_save.assert_called_once()
        mock_running.assert_called_once()
        mock_save_phase.assert_called_once_with("minimum")
        mock_client.send_command_v2.assert_called_once()
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["restart_required"] is True
        assert body["data"]["partitioned_convolution"]["max_partitions"] == 8
        assert body["data"]["phase_adjusted"] is True

    def test_enabling_forces_phase_to_minimum_even_when_daemon_stopped(self):
        """Even if daemon is offline we still mark phase as minimum."""
        payload = {
            "enabled": True,
            "fast_partition_taps": 32768,
            "min_partition_taps": 32768,
            "max_partitions": 4,
            "tail_fft_multiple": 2,
        }
        with patch(
            "web.routers.partitioned.save_partitioned_convolution_settings",
            return_value=True,
        ), patch(
            "web.routers.partitioned.check_daemon_running",
            return_value=False,
        ), patch(
            "web.routers.partitioned.save_phase_type",
            return_value=True,
        ) as mock_save_phase:
            response = client.put("/partitioned-convolution", json=payload)

        mock_save_phase.assert_called_once_with("minimum")
        assert response.status_code == 200
        assert response.json()["restart_required"] is False

    def test_handles_persistence_failure(self):
        """When config save fails, HTTP 500 should be returned."""
        payload = {
            "enabled": False,
            "fast_partition_taps": 32768,
            "min_partition_taps": 32768,
            "max_partitions": 4,
            "tail_fft_multiple": 2,
        }
        with patch(
            "web.routers.partitioned.save_partitioned_convolution_settings",
            return_value=False,
        ):
            response = client.put("/partitioned-convolution", json=payload)

        assert response.status_code == 500
        assert "Failed to save partitioned convolution settings" in response.json()["detail"]

    def test_validation_error_from_pydantic(self):
        """Invalid payload should map to VALIDATION_ERROR response."""
        payload = {
            "enabled": True,
            "fast_partition_taps": 2048,
            "min_partition_taps": 4096,  # invalid: min > fast
            "max_partitions": 2,
            "tail_fft_multiple": 1,  # invalid: < 2
        }

        response = client.put("/partitioned-convolution", json=payload)

        assert response.status_code == 422
        body = response.json()
        assert body["error_code"] == "VALIDATION_ERROR"
        assert "tail_fft_multiple" in body["detail"]

