"""Tests for DaemonClient error handling.

Tests the DaemonError exception class and JSON error response parsing
in web/services/daemon_client.py.
"""

import pytest

from web.error_codes import ErrorCode
from web.services.daemon_client import DaemonClient, DaemonError, DaemonResponse


class TestDaemonError:
    """Tests for DaemonError exception class."""

    def test_basic_creation(self):
        """Test basic DaemonError creation."""
        error = DaemonError(
            error_code="DAC_RATE_NOT_SUPPORTED",
            message="Sample rate not supported",
        )
        assert error.error_code == "DAC_RATE_NOT_SUPPORTED"
        assert error.message == "Sample rate not supported"
        assert error.inner_error is None

    def test_with_inner_error(self):
        """Test DaemonError with inner_error details."""
        inner = {
            "cpp_code": "0x2004",
            "alsa_errno": -22,
            "alsa_func": "snd_pcm_hw_params_set_rate_near",
        }
        error = DaemonError(
            error_code="DAC_RATE_NOT_SUPPORTED",
            message="Sample rate not supported",
            inner_error=inner,
        )
        assert error.inner_error == inner
        assert error.inner_error["cpp_code"] == "0x2004"

    def test_str_representation(self):
        """Test string representation of DaemonError."""
        error = DaemonError(
            error_code="IPC_TIMEOUT",
            message="Daemon not responding",
        )
        assert str(error) == "[IPC_TIMEOUT] Daemon not responding"

    def test_http_status_property(self):
        """Test http_status property maps correctly."""
        error = DaemonError(
            error_code="DAC_RATE_NOT_SUPPORTED",
            message="Test",
        )
        assert error.http_status == 422

        error = DaemonError(
            error_code="IPC_TIMEOUT",
            message="Test",
        )
        assert error.http_status == 504

        error = DaemonError(
            error_code="GPU_MEMORY_ERROR",
            message="Test",
        )
        assert error.http_status == 500

    def test_category_property(self):
        """Test category property maps correctly."""
        error = DaemonError(
            error_code="DAC_DEVICE_NOT_FOUND",
            message="Test",
        )
        assert error.category == "dac_alsa"

        error = DaemonError(
            error_code="IPC_CONNECTION_FAILED",
            message="Test",
        )
        assert error.category == "ipc_zeromq"

    def test_title_property(self):
        """Test title property maps correctly."""
        error = DaemonError(
            error_code="AUDIO_FILTER_NOT_FOUND",
            message="Test",
        )
        assert error.title == "Filter Not Found"

    def test_unknown_error_code(self):
        """Test handling of unknown error codes."""
        error = DaemonError(
            error_code="UNKNOWN_CODE",
            message="Unknown error",
        )
        # Should fall back to internal error defaults
        assert error.http_status == 500
        assert error.category == "internal"

    def test_is_exception(self):
        """Verify DaemonError is a proper Exception subclass."""
        error = DaemonError(
            error_code="IPC_TIMEOUT",
            message="Test",
        )
        assert isinstance(error, Exception)

        # Can be raised
        with pytest.raises(DaemonError) as exc_info:
            raise error
        assert exc_info.value.error_code == "IPC_TIMEOUT"


class TestDaemonResponse:
    """Tests for DaemonResponse dataclass."""

    def test_success_response(self):
        """Test successful response creation."""
        response = DaemonResponse(success=True, data={"phase_type": "minimum"})
        assert response.success is True
        assert response.data == {"phase_type": "minimum"}
        assert response.error is None

    def test_error_response(self):
        """Test error response creation."""
        error = DaemonError(
            error_code="IPC_TIMEOUT",
            message="Timeout",
        )
        response = DaemonResponse(success=False, error=error)
        assert response.success is False
        assert response.data is None
        assert response.error is not None
        assert response.error.error_code == "IPC_TIMEOUT"


class TestDaemonClientJsonParsing:
    """Tests for DaemonClient JSON response parsing."""

    def test_parse_json_success(self):
        """Test parsing successful JSON response."""
        client = DaemonClient()
        response = client._parse_json_response(
            '{"status": "ok", "data": {"value": 42}}'
        )
        assert response.success is True
        assert response.data == {"value": 42}

    def test_parse_json_error(self):
        """Test parsing error JSON response."""
        client = DaemonClient()
        json_str = """{
            "status": "error",
            "error_code": "DAC_RATE_NOT_SUPPORTED",
            "message": "Rate not supported",
            "inner_error": {"cpp_code": "0x2004", "alsa_errno": -22}
        }"""
        response = client._parse_json_response(json_str)
        assert response.success is False
        assert response.error is not None
        assert response.error.error_code == "DAC_RATE_NOT_SUPPORTED"
        assert response.error.message == "Rate not supported"
        assert response.error.inner_error["cpp_code"] == "0x2004"

    def test_parse_json_error_without_inner_error(self):
        """Test parsing error response without inner_error."""
        client = DaemonClient()
        json_str = """{
            "status": "error",
            "error_code": "IPC_TIMEOUT",
            "message": "Daemon timeout"
        }"""
        response = client._parse_json_response(json_str)
        assert response.success is False
        assert response.error.error_code == "IPC_TIMEOUT"
        assert response.error.inner_error is None

    def test_parse_json_unknown_status(self):
        """Test parsing response with unknown status."""
        client = DaemonClient()
        response = client._parse_json_response('{"status": "unknown"}')
        assert response.success is False
        assert response.error.error_code == ErrorCode.IPC_PROTOCOL_ERROR.value

    def test_parse_json_missing_error_code(self):
        """Test parsing error response with missing error_code."""
        client = DaemonClient()
        response = client._parse_json_response(
            '{"status": "error", "message": "Something"}'
        )
        assert response.success is False
        # Should default to IPC_PROTOCOL_ERROR
        assert response.error.error_code == "IPC_PROTOCOL_ERROR"


class TestDaemonClientLegacyParsing:
    """Tests for DaemonClient legacy text response parsing."""

    def test_parse_legacy_ok(self):
        """Test parsing legacy OK response."""
        client = DaemonClient()
        response = client._parse_legacy_response("OK")
        assert response.success is True
        assert response.data == "Command executed"

    def test_parse_legacy_ok_with_data(self):
        """Test parsing legacy OK:data response."""
        client = DaemonClient()
        response = client._parse_legacy_response("OK:some data here")
        assert response.success is True
        assert response.data == "some data here"

    def test_parse_legacy_err(self):
        """Test parsing legacy ERR response."""
        client = DaemonClient()
        response = client._parse_legacy_response("ERR:Device not found")
        assert response.success is False
        assert response.error is not None
        assert response.error.message == "Device not found"
        assert response.error.error_code == ErrorCode.IPC_PROTOCOL_ERROR.value

    def test_parse_legacy_unknown(self):
        """Test parsing unknown legacy response."""
        client = DaemonClient()
        response = client._parse_legacy_response("SOMETHING_ELSE")
        assert response.success is False
        assert "Unexpected response" in response.error.message

    def test_fallback_to_legacy(self):
        """Test fallback to legacy parsing for non-JSON response."""
        client = DaemonClient()
        # Non-JSON string should fall back to legacy parsing
        response = client._parse_json_response("OK:data")
        assert response.success is True
        assert response.data == "data"


class TestDaemonClientErrorCodes:
    """Tests for DaemonClient error code mapping."""

    @pytest.mark.parametrize(
        "error_code,expected_http,expected_category",
        [
            ("AUDIO_INVALID_INPUT_RATE", 400, "audio_processing"),
            ("DAC_RATE_NOT_SUPPORTED", 422, "dac_alsa"),
            ("IPC_TIMEOUT", 504, "ipc_zeromq"),
            ("IPC_DAEMON_NOT_RUNNING", 503, "ipc_zeromq"),
            ("GPU_MEMORY_ERROR", 500, "gpu_cuda"),
            ("VALIDATION_PATH_TRAVERSAL", 400, "validation"),
        ],
    )
    def test_error_code_to_http_mapping(
        self, error_code, expected_http, expected_category
    ):
        """Test error codes map to correct HTTP status and category."""
        error = DaemonError(error_code=error_code, message="Test")
        assert error.http_status == expected_http
        assert error.category == expected_category


class TestDaemonClientInnerError:
    """Tests for inner_error handling."""

    def test_full_inner_error(self):
        """Test handling of complete inner_error structure."""
        client = DaemonClient()
        json_str = """{
            "status": "error",
            "error_code": "DAC_OPEN_FAILED",
            "message": "Failed to open DAC",
            "inner_error": {
                "cpp_code": "0x2002",
                "cpp_message": "snd_pcm_open failed",
                "alsa_errno": -16,
                "alsa_func": "snd_pcm_open",
                "cuda_error": null
            }
        }"""
        response = client._parse_json_response(json_str)
        assert response.error.inner_error is not None
        inner = response.error.inner_error
        assert inner["cpp_code"] == "0x2002"
        assert inner["alsa_errno"] == -16
        assert inner["alsa_func"] == "snd_pcm_open"
        assert inner["cuda_error"] is None

    def test_cuda_error_in_inner_error(self):
        """Test handling of CUDA errors in inner_error."""
        client = DaemonClient()
        json_str = """{
            "status": "error",
            "error_code": "GPU_MEMORY_ERROR",
            "message": "GPU memory allocation failed",
            "inner_error": {
                "cpp_code": "0x4003",
                "cuda_error": "cudaErrorMemoryAllocation"
            }
        }"""
        response = client._parse_json_response(json_str)
        inner = response.error.inner_error
        assert inner["cuda_error"] == "cudaErrorMemoryAllocation"
