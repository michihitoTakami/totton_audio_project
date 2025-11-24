"""E2E tests for error propagation from C++ Daemon to REST API.

Tests that errors are correctly propagated through the stack:
C++ Audio Engine → ZeroMQ IPC → Python/FastAPI → HTTP Response (RFC 9457)

Issue #211: https://github.com/michihitoTakami/michy_os/issues/211
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from web.error_codes import ErrorCode
from web.exceptions import PROBLEM_JSON_MEDIA_TYPE
from web.main import app
from web.services.daemon_client import DaemonError, DaemonResponse


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


def create_error_response(
    error_code: str, message: str, inner_error: dict | None = None
) -> DaemonResponse:
    """Helper to create DaemonResponse with error."""
    error = DaemonError(
        error_code=error_code,
        message=message,
        inner_error=inner_error,
    )
    return DaemonResponse(success=False, error=error)


def create_success_response(data: dict | None = None) -> DaemonResponse:
    """Helper to create successful DaemonResponse."""
    return DaemonResponse(success=True, data=data)


class TestIPCErrorPropagation:
    """Tests for IPC layer (ZeroMQ) error propagation."""

    def test_ipc_timeout_returns_504(self, client):
        """IPC_TIMEOUT should return 504 Gateway Timeout."""
        error_response = create_error_response(
            error_code=ErrorCode.IPC_TIMEOUT.value,
            message="Daemon not responding (timeout)",
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 504
        assert response.headers["content-type"] == PROBLEM_JSON_MEDIA_TYPE
        data = response.json()
        assert data["error_code"] == "IPC_TIMEOUT"
        assert data["category"] == "ipc_zeromq"
        assert data["title"] == "Daemon Timeout"

    def test_ipc_daemon_not_running_returns_503(self, client):
        """IPC_DAEMON_NOT_RUNNING should return 503 Service Unavailable."""
        error_response = create_error_response(
            error_code=ErrorCode.IPC_DAEMON_NOT_RUNNING.value,
            message="Daemon is not running",
            inner_error={"zmq_errno": 111},  # ECONNREFUSED
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 503
        data = response.json()
        assert data["error_code"] == "IPC_DAEMON_NOT_RUNNING"
        assert data["category"] == "ipc_zeromq"
        assert data["inner_error"]["zmq_errno"] == 111

    def test_ipc_protocol_error_returns_500(self, client):
        """IPC_PROTOCOL_ERROR should return 500 Internal Server Error."""
        error_response = create_error_response(
            error_code=ErrorCode.IPC_PROTOCOL_ERROR.value,
            message="Invalid JSON response from daemon",
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 500
        data = response.json()
        assert data["error_code"] == "IPC_PROTOCOL_ERROR"
        assert data["category"] == "ipc_zeromq"

    def test_ipc_connection_failed_returns_503(self, client):
        """IPC_CONNECTION_FAILED should return 503 Service Unavailable."""
        error_response = create_error_response(
            error_code=ErrorCode.IPC_CONNECTION_FAILED.value,
            message="Failed to connect to daemon",
            inner_error={"zmq_errno": 2},
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 503
        data = response.json()
        assert data["error_code"] == "IPC_CONNECTION_FAILED"


class TestDACErrorPropagation:
    """Tests for DAC layer (ALSA) error propagation."""

    def test_dac_rate_not_supported_returns_422(self, client):
        """DAC_RATE_NOT_SUPPORTED should return 422 with ALSA errno."""
        error_response = create_error_response(
            error_code=ErrorCode.DAC_RATE_NOT_SUPPORTED.value,
            message="Sample rate 1000000 is not supported by DAC",
            inner_error={
                "cpp_code": "0x2004",
                "alsa_errno": -22,
                "alsa_func": "snd_pcm_hw_params_set_rate_near",
            },
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 422
        data = response.json()
        assert data["error_code"] == "DAC_RATE_NOT_SUPPORTED"
        assert data["category"] == "dac_alsa"
        assert data["title"] == "DAC Rate Not Supported"
        assert data["inner_error"]["cpp_code"] == "0x2004"
        assert data["inner_error"]["alsa_errno"] == -22
        assert data["inner_error"]["alsa_func"] == "snd_pcm_hw_params_set_rate_near"

    def test_dac_device_not_found_returns_404(self, client):
        """DAC_DEVICE_NOT_FOUND should return 404 Not Found."""
        error_response = create_error_response(
            error_code=ErrorCode.DAC_DEVICE_NOT_FOUND.value,
            message="DAC device hw:99 not found",
            inner_error={"cpp_code": "0x2001", "alsa_func": "snd_pcm_open"},
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 404
        data = response.json()
        assert data["error_code"] == "DAC_DEVICE_NOT_FOUND"
        assert data["category"] == "dac_alsa"

    def test_dac_busy_returns_409(self, client):
        """DAC_BUSY should return 409 Conflict."""
        error_response = create_error_response(
            error_code=ErrorCode.DAC_BUSY.value,
            message="DAC device is busy",
            inner_error={"alsa_errno": -16},  # EBUSY
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 409
        data = response.json()
        assert data["error_code"] == "DAC_BUSY"
        assert data["inner_error"]["alsa_errno"] == -16


class TestGPUErrorPropagation:
    """Tests for GPU layer (CUDA) error propagation."""

    def test_gpu_memory_error_returns_500(self, client):
        """GPU_MEMORY_ERROR should return 500 with CUDA error details."""
        error_response = create_error_response(
            error_code=ErrorCode.GPU_MEMORY_ERROR.value,
            message="GPU memory allocation failed",
            inner_error={
                "cpp_code": "0x4003",
                "cuda_error": "cudaErrorMemoryAllocation",
            },
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 500
        data = response.json()
        assert data["error_code"] == "GPU_MEMORY_ERROR"
        assert data["category"] == "gpu_cuda"
        assert data["title"] == "GPU Memory Error"
        assert data["inner_error"]["cuda_error"] == "cudaErrorMemoryAllocation"

    def test_gpu_cufft_error_returns_500(self, client):
        """GPU_CUFFT_ERROR should return 500 Internal Server Error."""
        error_response = create_error_response(
            error_code=ErrorCode.GPU_CUFFT_ERROR.value,
            message="cuFFT plan creation failed",
            inner_error={"cpp_code": "0x4006"},
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 500
        data = response.json()
        assert data["error_code"] == "GPU_CUFFT_ERROR"
        assert data["category"] == "gpu_cuda"


class TestValidationErrorPropagation:
    """Tests for validation error propagation."""

    def test_validation_path_traversal_returns_400(self, client):
        """VALIDATION_PATH_TRAVERSAL should return 400 Bad Request."""
        error_response = create_error_response(
            error_code=ErrorCode.VALIDATION_PATH_TRAVERSAL.value,
            message="Path traversal detected: ../../../etc/passwd",
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == "VALIDATION_PATH_TRAVERSAL"
        assert data["category"] == "validation"


class TestRFC9457Compliance:
    """Tests for RFC 9457 Problem Details compliance."""

    def test_content_type_is_problem_json(self, client):
        """Error responses must use application/problem+json content type."""
        error_response = create_error_response(
            error_code=ErrorCode.IPC_TIMEOUT.value,
            message="Timeout",
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.headers["content-type"] == PROBLEM_JSON_MEDIA_TYPE

    def test_required_fields_present(self, client):
        """Error responses must have all required RFC 9457 fields."""
        error_response = create_error_response(
            error_code=ErrorCode.DAC_RATE_NOT_SUPPORTED.value,
            message="Rate not supported",
            inner_error={"cpp_code": "0x2004"},
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        data = response.json()

        # RFC 9457 required fields
        assert "type" in data
        assert "title" in data
        assert "status" in data
        assert "detail" in data

        # Application-specific extensions
        assert "error_code" in data
        assert "category" in data

    def test_type_field_is_uri(self, client):
        """type field should be a URI reference."""
        error_response = create_error_response(
            error_code=ErrorCode.GPU_MEMORY_ERROR.value,
            message="Memory error",
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        data = response.json()
        assert data["type"].startswith("/errors/")
        assert data["type"] == "/errors/gpu-memory-error"

    def test_status_matches_http_code(self, client):
        """status field should match HTTP status code."""
        error_response = create_error_response(
            error_code=ErrorCode.IPC_TIMEOUT.value,
            message="Timeout",
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        data = response.json()
        assert data["status"] == response.status_code
        assert data["status"] == 504

    def test_inner_error_propagated(self, client):
        """inner_error should be propagated when present."""
        inner = {
            "cpp_code": "0x2004",
            "cpp_message": "ALSA rate negotiation failed",
            "alsa_errno": -22,
            "alsa_func": "snd_pcm_hw_params_set_rate_near",
            "cuda_error": None,
        }
        error_response = create_error_response(
            error_code=ErrorCode.DAC_RATE_NOT_SUPPORTED.value,
            message="Rate not supported",
            inner_error=inner,
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        data = response.json()
        assert "inner_error" in data
        assert data["inner_error"]["cpp_code"] == "0x2004"
        assert data["inner_error"]["alsa_errno"] == -22


class TestUnknownErrorCodeFallback:
    """Tests for unknown error code handling."""

    def test_unknown_error_code_returns_500(self, client):
        """Unknown error codes should return 500 Internal Server Error."""
        error_response = create_error_response(
            error_code="UNKNOWN_FUTURE_ERROR",
            message="Some future error type",
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 500
        data = response.json()
        assert data["error_code"] == "UNKNOWN_FUTURE_ERROR"
        assert data["category"] == "internal"
        assert data["title"] == "Internal Error"


class TestAudioErrorPropagation:
    """Tests for audio processing error propagation."""

    def test_audio_buffer_overflow_returns_500(self, client):
        """AUDIO_BUFFER_OVERFLOW should return 500."""
        error_response = create_error_response(
            error_code=ErrorCode.AUDIO_BUFFER_OVERFLOW.value,
            message="Audio buffer overflow detected",
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 500
        data = response.json()
        assert data["error_code"] == "AUDIO_BUFFER_OVERFLOW"
        assert data["category"] == "audio_processing"

    def test_audio_filter_not_found_returns_404(self, client):
        """AUDIO_FILTER_NOT_FOUND should return 404."""
        error_response = create_error_response(
            error_code=ErrorCode.AUDIO_FILTER_NOT_FOUND.value,
            message="Filter file not found: coefficients/missing.bin",
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 404
        data = response.json()
        assert data["error_code"] == "AUDIO_FILTER_NOT_FOUND"


class TestSuccessResponse:
    """Tests to ensure success responses work correctly alongside error handling."""

    def test_success_response_not_affected(self, client):
        """Success responses should work normally."""
        success_response = create_success_response({"phase_type": "minimum"})

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = success_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.get("/daemon/phase-type")

        assert response.status_code == 200
        data = response.json()
        assert data["phase_type"] == "minimum"


class TestPhaseTypeEndpointErrors:
    """Tests specific to phase-type endpoint error handling."""

    def test_put_phase_type_error(self, client):
        """PUT /daemon/phase-type should propagate errors."""
        error_response = create_error_response(
            error_code=ErrorCode.IPC_TIMEOUT.value,
            message="Daemon not responding",
        )

        with patch("web.routers.daemon.get_daemon_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.send_command_v2.return_value = error_response
            mock_factory.return_value.__enter__.return_value = mock_client

            response = client.put("/daemon/phase-type", json={"phase_type": "minimum"})

        assert response.status_code == 504
        data = response.json()
        assert data["error_code"] == "IPC_TIMEOUT"
