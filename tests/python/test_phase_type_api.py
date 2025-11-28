"""Tests for phase type API endpoints."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from web.main import app
from web.models import PartitionedConvolutionSettings
from web.services.daemon_client import DaemonError, DaemonResponse


client = TestClient(app)


class TestPhaseTypeGet:
    """Tests for GET /daemon/phase-type endpoint."""

    def test_get_phase_type_minimum(self):
        """Test getting minimum phase type."""
        with patch("web.routers.daemon.get_daemon_client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.send_command_v2.return_value = DaemonResponse(
                success=True, data={"phase_type": "minimum"}
            )
            mock_client.return_value = mock_instance

            response = client.get("/daemon/phase-type")

            assert response.status_code == 200
            data = response.json()
            assert data["phase_type"] == "minimum"
            assert data["latency_warning"] is None

    def test_get_phase_type_linear_with_warning(self):
        """Test getting linear phase type includes latency warning."""
        with patch("web.routers.daemon.get_daemon_client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.send_command_v2.return_value = DaemonResponse(
                success=True, data={"phase_type": "linear"}
            )
            mock_client.return_value = mock_instance

            response = client.get("/daemon/phase-type")

            assert response.status_code == 200
            data = response.json()
            assert data["phase_type"] == "linear"
            assert data["latency_warning"] is not None
            assert "latency" in data["latency_warning"].lower()

    def test_get_phase_type_daemon_error(self):
        """Test error handling when daemon communication fails."""
        with patch("web.routers.daemon.get_daemon_client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.send_command_v2.return_value = DaemonResponse(
                success=False,
                error=DaemonError(
                    error_code="IPC_DAEMON_NOT_RUNNING",
                    message="Daemon not responding",
                ),
            )
            mock_client.return_value = mock_instance

            response = client.get("/daemon/phase-type")

            assert response.status_code == 503
            assert "Daemon not responding" in response.json()["detail"]


class TestPhaseTypeSet:
    """Tests for PUT /daemon/phase-type endpoint."""

    def test_set_phase_type_minimum(self):
        """Test setting phase type to minimum."""
        with patch("web.routers.daemon.get_daemon_client") as mock_client, patch(
            "web.routers.daemon.save_phase_type"
        ) as mock_save_phase, patch(
            "web.routers.daemon.load_partitioned_convolution_settings"
        ) as mock_load_partition, patch(
            "web.routers.daemon.save_partitioned_convolution_settings"
        ) as mock_save_partition:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.send_command_v2.return_value = DaemonResponse(
                success=True, data={"phase_type": "minimum"}
            )
            mock_client.return_value = mock_instance
            mock_load_partition.return_value = PartitionedConvolutionSettings(enabled=False)

            response = client.put(
                "/daemon/phase-type",
                json={"phase_type": "minimum"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["phase_type"] == "minimum"
            mock_save_phase.assert_called_once_with("minimum")
            mock_save_partition.assert_not_called()

    def test_set_phase_type_linear(self):
        """Test setting phase type to linear."""
        with patch("web.routers.daemon.get_daemon_client") as mock_client, patch(
            "web.routers.daemon.save_phase_type"
        ) as mock_save_phase, patch(
            "web.routers.daemon.load_partitioned_convolution_settings"
        ) as mock_load_partition, patch(
            "web.routers.daemon.save_partitioned_convolution_settings"
        ) as mock_save_partition:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.send_command_v2.return_value = DaemonResponse(
                success=True, data={"phase_type": "linear"}
            )
            mock_client.return_value = mock_instance
            mock_load_partition.return_value = PartitionedConvolutionSettings(
                enabled=True,
                fast_partition_taps=49152,
                min_partition_taps=8192,
                max_partitions=6,
                tail_fft_multiple=4,
            )

            response = client.put(
                "/daemon/phase-type",
                json={"phase_type": "linear"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["phase_type"] == "linear"
            assert data["data"]["partition_disabled"] is True
            mock_save_phase.assert_called_once_with("linear")
            mock_save_partition.assert_called_once()

    def test_set_phase_type_invalid(self):
        """Test setting invalid phase type returns 422 (Pydantic Literal validation)."""
        response = client.put(
            "/daemon/phase-type",
            json={"phase_type": "invalid"},
        )

        assert response.status_code == 422
        # Pydantic returns structured validation error
        data = response.json()
        assert "detail" in data
        assert "error_code" in data
        assert data["error_code"] == "VALIDATION_ERROR"

    def test_set_phase_type_daemon_error(self):
        """Test error handling when daemon returns error."""
        with patch("web.routers.daemon.get_daemon_client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.send_command_v2.return_value = DaemonResponse(
                success=False,
                error=DaemonError(
                    error_code="IPC_DAEMON_NOT_RUNNING",
                    message="Quad-phase mode not enabled",
                ),
            )
            mock_client.return_value = mock_instance

            response = client.put(
                "/daemon/phase-type",
                json={"phase_type": "linear"},
            )

            assert response.status_code == 503
            assert "Quad-phase mode not enabled" in response.json()["detail"]


class TestDaemonClientPhaseType:
    """Tests for DaemonClient phase type methods."""

    def test_get_phase_type_parses_json(self):
        """Test that get_phase_type correctly parses JSON response."""
        from web.services.daemon_client import DaemonClient, DaemonResponse

        client = DaemonClient()
        with patch.object(client, "send_command_v2") as mock_send:
            mock_send.return_value = DaemonResponse(
                success=True, data={"phase_type": "minimum"}
            )

            success, result = client.get_phase_type()

            assert success is True
            assert result == {"phase_type": "minimum"}

    def test_get_phase_type_invalid_json(self):
        """Test that get_phase_type handles daemon error."""
        from web.services.daemon_client import DaemonClient, DaemonError, DaemonResponse

        client = DaemonClient()
        with patch.object(client, "send_command_v2") as mock_send:
            mock_send.return_value = DaemonResponse(
                success=False,
                error=DaemonError(
                    error_code="IPC_TIMEOUT", message="Daemon not responding"
                ),
            )

            success, result = client.get_phase_type()

            assert success is False
            assert "Daemon not responding" in result

    def test_set_phase_type_validates_input(self):
        """Test that set_phase_type validates input."""
        from web.services.daemon_client import DaemonClient

        client = DaemonClient()

        success, message = client.set_phase_type("invalid")

        assert success is False
        assert "Invalid phase type" in message

    def test_set_phase_type_sends_correct_command(self):
        """Test that set_phase_type sends correct command."""
        from web.services.daemon_client import DaemonClient

        client = DaemonClient()
        with patch.object(client, "send_command") as mock_send:
            mock_send.return_value = (True, "Phase type set")

            client.set_phase_type("linear")

            mock_send.assert_called_once_with("PHASE_TYPE_SET:linear")
