"""Tests for partitioned convolution API endpoints (Issue #354)."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from web.main import app
from web.models import PartitionedConvolutionSettings

client = TestClient(app)


def _sample_settings() -> PartitionedConvolutionSettings:
    """Helper to build a valid settings object."""
    return PartitionedConvolutionSettings(
        enabled=True,
        fast_partition_taps=48000,
        min_partition_taps=8000,
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
        assert body["fast_partition_taps"] == 48000
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
        ) as mock_running:
            response = client.put("/partitioned-convolution", json=payload)

        mock_save.assert_called_once()
        mock_running.assert_called_once()
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["restart_required"] is True
        assert body["data"]["partitioned_convolution"]["max_partitions"] == 8

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
        assert "Partitioned convolution" in response.json()["detail"]

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
        assert "min_partition_taps" in body["detail"]
        assert "tail_fft_multiple" in body["detail"]

