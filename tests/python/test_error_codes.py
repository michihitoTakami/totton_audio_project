"""Tests for error code system.

Tests the error code enums, mappings, and utility functions
defined in web/error_codes.py.
"""

import pytest

from web.error_codes import (
    ERROR_MAPPINGS,
    ErrorCategory,
    ErrorCode,
    ErrorMapping,
    get_category_from_code,
    get_error_mapping,
)


class TestErrorCodeEnum:
    """Tests for ErrorCode enum."""

    def test_all_codes_defined(self):
        """Verify all error codes are defined (30 base + 4 crossfeed + 1 output)."""
        assert len(ErrorCode) == 35

    def test_no_duplicate_values(self):
        """Verify no duplicate enum values."""
        values = [code.value for code in ErrorCode]
        assert len(values) == len(set(values))

    def test_audio_processing_codes(self):
        """Verify Audio Processing category codes (6 codes)."""
        audio_codes = [c for c in ErrorCode if c.value.startswith("AUDIO_")]
        assert len(audio_codes) == 6
        assert ErrorCode.AUDIO_INVALID_INPUT_RATE in audio_codes
        assert ErrorCode.AUDIO_XRUN_DETECTED in audio_codes

    def test_dac_codes(self):
        """Verify DAC/ALSA category codes (6 codes)."""
        dac_codes = [c for c in ErrorCode if c.value.startswith("DAC_")]
        assert len(dac_codes) == 6
        assert ErrorCode.DAC_DEVICE_NOT_FOUND in dac_codes
        assert ErrorCode.DAC_BUSY in dac_codes

    def test_ipc_codes(self):
        """Verify IPC/ZeroMQ category codes (6 codes)."""
        ipc_codes = [c for c in ErrorCode if c.value.startswith("IPC_")]
        assert len(ipc_codes) == 6
        assert ErrorCode.IPC_TIMEOUT in ipc_codes
        assert ErrorCode.IPC_DAEMON_NOT_RUNNING in ipc_codes

    def test_gpu_codes(self):
        """Verify GPU/CUDA category codes (6 codes)."""
        gpu_codes = [c for c in ErrorCode if c.value.startswith("GPU_")]
        assert len(gpu_codes) == 6
        assert ErrorCode.GPU_MEMORY_ERROR in gpu_codes
        assert ErrorCode.GPU_CUFFT_ERROR in gpu_codes

    def test_validation_codes(self):
        """Verify Validation category codes (6 codes)."""
        validation_codes = [c for c in ErrorCode if c.value.startswith("VALIDATION_")]
        assert len(validation_codes) == 6
        assert ErrorCode.VALIDATION_PATH_TRAVERSAL in validation_codes
        assert ErrorCode.VALIDATION_INVALID_HEADPHONE in validation_codes

    def test_crossfeed_codes(self):
        """Verify Crossfeed/HRTF category codes (4 codes)."""
        crossfeed_codes = [c for c in ErrorCode if c.value.startswith("CROSSFEED_")]
        assert len(crossfeed_codes) == 4
        assert ErrorCode.CROSSFEED_NOT_INITIALIZED in crossfeed_codes
        assert ErrorCode.CROSSFEED_INVALID_RATE_FAMILY in crossfeed_codes
        assert ErrorCode.CROSSFEED_NOT_IMPLEMENTED in crossfeed_codes
        assert ErrorCode.CROSSFEED_INVALID_FILTER_SIZE in crossfeed_codes


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_all_categories_defined(self):
        """Verify all categories are defined (6 base + 1 crossfeed)."""
        assert len(ErrorCategory) == 7

    def test_category_values(self):
        """Verify category string values."""
        assert ErrorCategory.AUDIO_PROCESSING.value == "audio_processing"
        assert ErrorCategory.DAC_ALSA.value == "dac_alsa"
        assert ErrorCategory.IPC_ZEROMQ.value == "ipc_zeromq"
        assert ErrorCategory.GPU_CUDA.value == "gpu_cuda"
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.CROSSFEED.value == "crossfeed"
        assert ErrorCategory.INTERNAL.value == "internal"


class TestErrorMappings:
    """Tests for ERROR_MAPPINGS dictionary."""

    def test_all_codes_have_mappings(self):
        """Verify all error codes have mappings (30 base + 4 crossfeed + 1 output)."""
        assert len(ERROR_MAPPINGS) == 35
        for code in ErrorCode:
            assert code in ERROR_MAPPINGS, f"Missing mapping for {code}"

    def test_mapping_structure(self):
        """Verify mapping structure is correct."""
        for code, mapping in ERROR_MAPPINGS.items():
            assert isinstance(mapping, ErrorMapping)
            assert isinstance(mapping.http_status, int)
            assert isinstance(mapping.category, ErrorCategory)
            assert isinstance(mapping.title, str)

    def test_http_status_ranges(self):
        """Verify HTTP status codes are in valid ranges."""
        valid_statuses = {400, 404, 409, 422, 500, 501, 503, 504}
        for code, mapping in ERROR_MAPPINGS.items():
            assert (
                mapping.http_status in valid_statuses
            ), f"{code} has invalid HTTP status {mapping.http_status}"

    @pytest.mark.parametrize(
        "error_code,expected_status",
        [
            (ErrorCode.AUDIO_INVALID_INPUT_RATE, 400),
            (ErrorCode.AUDIO_FILTER_NOT_FOUND, 404),
            (ErrorCode.DAC_DEVICE_NOT_FOUND, 404),
            (ErrorCode.DAC_RATE_NOT_SUPPORTED, 422),
            (ErrorCode.DAC_BUSY, 409),
            (ErrorCode.IPC_TIMEOUT, 504),
            (ErrorCode.IPC_DAEMON_NOT_RUNNING, 503),
            (ErrorCode.GPU_MEMORY_ERROR, 500),
            (ErrorCode.VALIDATION_PATH_TRAVERSAL, 400),
            (ErrorCode.VALIDATION_FILE_NOT_FOUND, 404),
            (ErrorCode.VALIDATION_PROFILE_EXISTS, 409),
        ],
    )
    def test_specific_status_mappings(self, error_code, expected_status):
        """Test specific error code to HTTP status mappings."""
        assert ERROR_MAPPINGS[error_code].http_status == expected_status

    @pytest.mark.parametrize(
        "error_code,expected_category",
        [
            (ErrorCode.AUDIO_BUFFER_OVERFLOW, ErrorCategory.AUDIO_PROCESSING),
            (ErrorCode.DAC_OPEN_FAILED, ErrorCategory.DAC_ALSA),
            (ErrorCode.IPC_CONNECTION_FAILED, ErrorCategory.IPC_ZEROMQ),
            (ErrorCode.GPU_INIT_FAILED, ErrorCategory.GPU_CUDA),
            (ErrorCode.VALIDATION_INVALID_CONFIG, ErrorCategory.VALIDATION),
        ],
    )
    def test_category_mappings(self, error_code, expected_category):
        """Test error code to category mappings."""
        assert ERROR_MAPPINGS[error_code].category == expected_category


class TestGetErrorMapping:
    """Tests for get_error_mapping() function."""

    def test_known_error_code(self):
        """Test lookup of known error code."""
        mapping = get_error_mapping("DAC_RATE_NOT_SUPPORTED")
        assert mapping.http_status == 422
        assert mapping.category == ErrorCategory.DAC_ALSA
        assert mapping.title == "DAC Rate Not Supported"

    def test_unknown_error_code(self):
        """Test fallback for unknown error code."""
        mapping = get_error_mapping("UNKNOWN_ERROR_CODE")
        assert mapping.http_status == 500
        assert mapping.category == ErrorCategory.INTERNAL
        assert mapping.title == "Internal Error"

    def test_empty_string(self):
        """Test fallback for empty string."""
        mapping = get_error_mapping("")
        assert mapping.http_status == 500
        assert mapping.category == ErrorCategory.INTERNAL

    def test_case_sensitivity(self):
        """Test that error codes are case-sensitive."""
        # Lowercase should not match
        mapping = get_error_mapping("dac_rate_not_supported")
        assert mapping.http_status == 500  # Falls back to default


class TestGetCategoryFromCode:
    """Tests for get_category_from_code() function."""

    @pytest.mark.parametrize(
        "error_code,expected_category",
        [
            ("AUDIO_SOMETHING", ErrorCategory.AUDIO_PROCESSING),
            ("DAC_WHATEVER", ErrorCategory.DAC_ALSA),
            ("IPC_ANYTHING", ErrorCategory.IPC_ZEROMQ),
            ("GPU_ERROR", ErrorCategory.GPU_CUDA),
            ("VALIDATION_ISSUE", ErrorCategory.VALIDATION),
        ],
    )
    def test_prefix_detection(self, error_code, expected_category):
        """Test category detection from error code prefix."""
        assert get_category_from_code(error_code) == expected_category

    def test_unknown_prefix(self):
        """Test fallback for unknown prefix."""
        assert get_category_from_code("UNKNOWN_ERROR") == ErrorCategory.INTERNAL
        assert get_category_from_code("") == ErrorCategory.INTERNAL


class TestErrorMappingDataclass:
    """Tests for ErrorMapping dataclass."""

    def test_frozen(self):
        """Verify ErrorMapping is immutable (frozen)."""
        mapping = ErrorMapping(
            http_status=400,
            category=ErrorCategory.VALIDATION,
            title="Test Error",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            mapping.http_status = 500

    def test_equality(self):
        """Test ErrorMapping equality."""
        mapping1 = ErrorMapping(400, ErrorCategory.VALIDATION, "Test")
        mapping2 = ErrorMapping(400, ErrorCategory.VALIDATION, "Test")
        assert mapping1 == mapping2

    def test_hashable(self):
        """Test ErrorMapping is hashable (can be used in sets/dicts)."""
        mapping = ErrorMapping(400, ErrorCategory.VALIDATION, "Test")
        # Should not raise
        hash(mapping)
        {mapping}  # Can be added to set
