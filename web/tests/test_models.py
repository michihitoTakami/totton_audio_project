"""Unit tests for Pydantic models."""

import pytest

from web.models import (
    ApiResponse,
    CrossfeedSettings,
    DaemonStatus,
    EqProfileInfo,
    PhaseTypeResponse,
    PhaseTypeUpdateRequest,
    Settings,
    Status,
    TcpInputConfigUpdate,
    TcpInputSettings,
    TcpInputTelemetry,
)


class TestApiResponse:
    """Test ApiResponse model."""

    def test_create_success_response(self):
        """Test creating a successful API response."""
        response = ApiResponse(success=True, message="Operation successful")
        assert response.success is True
        assert response.message == "Operation successful"
        assert response.restart_required is False

    def test_create_error_response(self):
        """Test creating an error API response."""
        response = ApiResponse(success=False, message="Error occurred")
        assert response.success is False
        assert response.message == "Error occurred"

    def test_api_response_with_data(self):
        """Test API response with data field."""
        response = ApiResponse(success=True, message="Success", data={"key": "value"})
        assert response.data == {"key": "value"}


class TestStatus:
    """Test Status model."""

    def test_status_with_settings(self):
        """Test Status with settings."""
        settings = Settings(
            alsa_device="default",
            upsample_ratio=8,
            eq_enabled=True,
            eq_profile="HD650",
        )
        status = Status(
            settings=settings,
            daemon_running=True,
            eq_active=True,
            input_rate=44100,
            output_rate=352800,
        )
        assert status.daemon_running is True
        assert status.eq_active is True
        assert status.settings.eq_profile == "HD650"

    def test_status_default_values(self):
        """Test Status default values."""
        settings = Settings()
        status = Status(settings=settings)
        assert status.daemon_running is False
        assert status.eq_active is False
        assert status.alsa_connected is False


class TestDaemonStatus:
    """Test DaemonStatus model."""

    def test_daemon_status_running(self):
        """Test daemon running status."""
        status = DaemonStatus(
            running=True,
            pid=12345,
            pid_file="/tmp/daemon.pid",
            binary_path="/usr/bin/daemon",
        )
        assert status.running is True
        assert status.pid == 12345

    def test_daemon_status_stopped(self):
        """Test daemon stopped status."""
        status = DaemonStatus(
            running=False,
            pid=None,
            pid_file="/tmp/daemon.pid",
            binary_path="/usr/bin/daemon",
        )
        assert status.running is False
        assert status.pid is None


class TestEqProfileInfo:
    """Test EqProfileInfo model."""

    def test_eq_profile_info_with_all_fields(self):
        """Test EqProfileInfo with all fields."""
        profile = EqProfileInfo(
            name="HD650",
            filename="hd650.txt",
            path="/data/EQ/hd650.txt",
            size=1024,
            modified=1234567890.0,
            type="opra",
            filter_count=10,
        )
        assert profile.name == "HD650"
        assert profile.filename == "hd650.txt"
        assert profile.type == "opra"
        assert profile.filter_count == 10


class TestCrossfeedSettings:
    """Test CrossfeedSettings model."""

    def test_crossfeed_settings_default(self):
        """Test CrossfeedSettings with defaults."""
        settings = CrossfeedSettings()
        assert settings.enabled is False
        assert settings.head_size == "m"

    def test_crossfeed_settings_custom(self):
        """Test CrossfeedSettings with custom values."""
        settings = CrossfeedSettings(
            enabled=True, head_size="l", hrtf_path="/custom/path"
        )
        assert settings.enabled is True
        assert settings.head_size == "l"


class TestPhaseType:
    """Test PhaseType models."""

    def test_phase_type_response(self):
        """Test PhaseTypeResponse model."""
        response = PhaseTypeResponse(phase_type="minimum")
        assert response.phase_type == "minimum"
        assert response.latency_warning is None

    def test_phase_type_response_with_warning(self):
        """Test PhaseTypeResponse with warning."""
        response = PhaseTypeResponse(
            phase_type="linear",
            latency_warning="~6.7ms alignment above 150 Hz",
        )
        assert response.phase_type == "linear"
        assert response.latency_warning is not None

    def test_phase_type_update_request(self):
        """Test PhaseTypeUpdateRequest model."""
        request = PhaseTypeUpdateRequest(phase_type="linear")
        assert request.phase_type == "linear"


class TestModelSerialization:
    """Test model serialization/deserialization."""

    def test_status_to_dict(self):
        """Test Status serialization to dict."""
        settings = Settings(eq_enabled=True, eq_profile="HD650")
        status = Status(settings=settings, daemon_running=True, eq_active=True)
        data = status.model_dump()
        assert isinstance(data, dict)
        assert data["daemon_running"] is True
        assert data["settings"]["eq_profile"] == "HD650"

    def test_eq_profile_info_json_serialization(self):
        """Test EqProfileInfo JSON serialization."""
        profile = EqProfileInfo(
            name="HD650",
            filename="hd650.txt",
            path="/data/EQ/hd650.txt",
            size=1024,
            modified=1234567890.0,
            type="opra",
            filter_count=10,
        )
        json_str = profile.model_dump_json()
        assert isinstance(json_str, str)
        assert "HD650" in json_str
        assert "hd650.txt" in json_str

    def test_phase_type_request_serialization(self):
        """Test PhaseTypeUpdateRequest serialization."""
        request = PhaseTypeUpdateRequest(phase_type="linear")
        data = request.model_dump()
        assert data["phase_type"] == "linear"


class TestTcpInputModels:
    """Tests for TCP input related models."""

    def test_tcp_input_settings_defaults(self):
        """Settings should provide safe defaults."""
        settings = TcpInputSettings()
        assert settings.enabled is True
        assert settings.bind_address == "0.0.0.0"
        assert settings.port == 46001
        assert settings.buffer_size_bytes == 262_144
        assert settings.connection_mode == "single"
        assert settings.priority_clients == []

    def test_tcp_input_config_update_requires_fields(self):
        """Empty update should be rejected."""
        with pytest.raises(ValueError):
            TcpInputConfigUpdate()

    def test_tcp_input_config_update_aliases(self):
        """CamelCase aliases should be accepted."""
        update = TcpInputConfigUpdate(
            bindAddress="127.0.0.1",
            bufferSizeBytes=131072,
            port=5001,
        )
        assert update.bind_address == "127.0.0.1"
        assert update.buffer_size_bytes == 131072
        assert update.port == 5001

    def test_tcp_input_telemetry_defaults_and_types(self):
        """Telemetry defaults should be zero/None friendly."""
        telemetry = TcpInputTelemetry()
        assert telemetry.listening is False
        assert telemetry.bound_port is None
        assert telemetry.connection_mode == "single"
        assert telemetry.last_header is None
