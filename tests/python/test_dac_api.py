"""
Integration tests for DAC capability API endpoints.

Tests the following endpoints:
- GET /dac/capabilities - Get DAC capabilities
- GET /dac/devices - List devices with capabilities
- GET /dac/supported-rates - Get supported rates for a rate family
- GET /dac/max-ratio - Get maximum upsampling ratio
- GET /dac/validate-config - Validate DAC configuration
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from web.main import app  # noqa: E402
from web.services.dac import DacCapability  # noqa: E402
from web.services.daemon_client import DaemonError, DaemonResponse  # noqa: E402


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_valid_capability():
    """Mock valid DAC capability."""
    return DacCapability(
        device_name="hw:0",
        min_sample_rate=44100,
        max_sample_rate=768000,
        supported_rates=[
            44100,
            48000,
            88200,
            96000,
            176400,
            192000,
            352800,
            384000,
            705600,
            768000,
        ],
        max_channels=2,
        is_valid=True,
        error_message=None,
    )


@pytest.fixture
def mock_invalid_capability():
    """Mock invalid DAC capability (device not found)."""
    return DacCapability(
        device_name="hw:99",
        min_sample_rate=0,
        max_sample_rate=0,
        supported_rates=[],
        max_channels=0,
        is_valid=False,
        error_message="Device not found",
    )


class TestCapabilitiesEndpoint:
    """Tests for GET /dac/capabilities endpoint."""

    def test_capabilities_valid_device(self, client, mock_valid_capability):
        """Valid device should return capabilities."""
        with patch(
            "web.routers.dac.scan_dac_capability", return_value=mock_valid_capability
        ):
            response = client.get("/dac/capabilities?device=hw:0")

        assert response.status_code == 200
        data = response.json()
        assert data["device_name"] == "hw:0"
        assert data["is_valid"] is True
        assert 44100 in data["supported_rates"]
        assert 768000 in data["supported_rates"]
        assert data["max_channels"] == 2

    def test_capabilities_invalid_device_format(self, client):
        """Invalid device name format should return 400."""
        response = client.get("/dac/capabilities?device=../etc/passwd")

        assert response.status_code == 400
        assert "Invalid device name format" in response.json()["detail"]

    def test_capabilities_invalid_device_name_injection(self, client):
        """Path traversal attempt should return 400."""
        response = client.get("/dac/capabilities?device=hw:0/../../../etc")

        assert response.status_code == 400

    def test_capabilities_device_not_found(self, client, mock_invalid_capability):
        """Non-existent device should return valid response with is_valid=False."""
        with patch(
            "web.routers.dac.scan_dac_capability", return_value=mock_invalid_capability
        ):
            response = client.get("/dac/capabilities?device=hw:99")

        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert data["error_message"] == "Device not found"

    def test_capabilities_default_device(self, client, mock_valid_capability):
        """Default device parameter should work."""
        with patch(
            "web.routers.dac.scan_dac_capability", return_value=mock_valid_capability
        ):
            response = client.get("/dac/capabilities")

        assert response.status_code == 200


class TestDevicesEndpoint:
    """Tests for GET /dac/devices endpoint."""

    def test_devices_list(self, client, mock_valid_capability):
        """Should return list of devices."""
        mock_devices = [
            {"id": "default", "name": "Default", "description": "System default"},
            {"id": "hw:0", "name": "USB DAC", "description": "SMSL USB Audio"},
        ]

        with patch("web.routers.dac.get_alsa_devices", return_value=mock_devices):
            with patch(
                "web.routers.dac.scan_dac_capability",
                return_value=mock_valid_capability,
            ):
                response = client.get("/dac/devices")

        assert response.status_code == 200
        data = response.json()
        assert "devices" in data
        assert len(data["devices"]) == 2

    def test_devices_default_no_capability(self, client, mock_valid_capability):
        """Default device should not have capability info."""
        mock_devices = [
            {"id": "default", "name": "Default", "description": "System default"},
        ]

        with patch("web.routers.dac.get_alsa_devices", return_value=mock_devices):
            response = client.get("/dac/devices")

        assert response.status_code == 200
        data = response.json()
        assert data["devices"][0]["id"] == "default"
        assert data["devices"][0]["capabilities"] is None


class TestSupportedRatesEndpoint:
    """Tests for GET /dac/supported-rates endpoint."""

    def test_supported_rates_44k_family(self, client, mock_valid_capability):
        """44k family should return 44.1k multiples."""
        with patch(
            "web.routers.dac.get_supported_output_rates",
            return_value=[44100, 88200, 176400, 352800, 705600],
        ):
            response = client.get("/dac/supported-rates?device=hw:0&family=44k")

        assert response.status_code == 200
        data = response.json()
        assert data["family"] == "44k"
        assert 44100 in data["supported_rates"]
        assert 48000 not in data["supported_rates"]

    def test_supported_rates_48k_family(self, client, mock_valid_capability):
        """48k family should return 48k multiples."""
        with patch(
            "web.routers.dac.get_supported_output_rates",
            return_value=[48000, 96000, 192000, 384000, 768000],
        ):
            response = client.get("/dac/supported-rates?device=hw:0&family=48k")

        assert response.status_code == 200
        data = response.json()
        assert data["family"] == "48k"
        assert 48000 in data["supported_rates"]
        assert 44100 not in data["supported_rates"]

    def test_supported_rates_invalid_family(self, client):
        """Invalid family should return 400."""
        response = client.get("/dac/supported-rates?device=hw:0&family=invalid")

        assert response.status_code == 400
        assert "family must be '44k' or '48k'" in response.json()["detail"]

    def test_supported_rates_invalid_device(self, client):
        """Invalid device name should return 400."""
        response = client.get("/dac/supported-rates?device=../etc/passwd&family=44k")

        assert response.status_code == 400


class TestMaxRatioEndpoint:
    """Tests for GET /dac/max-ratio endpoint."""

    def test_max_ratio_44100(self, client):
        """44100 input should return correct max ratio."""
        with patch("web.routers.dac.get_max_upsample_ratio", return_value=16):
            response = client.get("/dac/max-ratio?device=hw:0&input_rate=44100")

        assert response.status_code == 200
        data = response.json()
        assert data["input_rate"] == 44100
        assert data["max_ratio"] == 16
        assert data["max_output_rate"] == 705600

    def test_max_ratio_48000(self, client):
        """48000 input should return correct max ratio."""
        with patch("web.routers.dac.get_max_upsample_ratio", return_value=16):
            response = client.get("/dac/max-ratio?device=hw:0&input_rate=48000")

        assert response.status_code == 200
        data = response.json()
        assert data["input_rate"] == 48000
        assert data["max_ratio"] == 16
        assert data["max_output_rate"] == 768000

    def test_max_ratio_invalid_input_rate(self, client):
        """Negative input rate should return 400."""
        response = client.get("/dac/max-ratio?device=hw:0&input_rate=-1")

        assert response.status_code == 400
        assert "input_rate must be positive" in response.json()["detail"]

    def test_max_ratio_zero_input_rate(self, client):
        """Zero input rate should return 400."""
        response = client.get("/dac/max-ratio?device=hw:0&input_rate=0")

        assert response.status_code == 400

    def test_max_ratio_invalid_device(self, client):
        """Invalid device name should return 400."""
        response = client.get("/dac/max-ratio?device=../etc/passwd&input_rate=44100")

        assert response.status_code == 400


class TestValidateConfigEndpoint:
    """Tests for GET /dac/validate-config endpoint."""

    def test_validate_config_valid(self, client, mock_valid_capability):
        """Valid configuration should pass validation."""
        with patch(
            "web.routers.dac.scan_dac_capability", return_value=mock_valid_capability
        ):
            response = client.get(
                "/dac/validate-config?device=hw:0&input_rate=44100&output_rate=705600"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["is_supported"] is True
        assert data["data"]["rate_supported"] is True
        assert data["data"]["ratio_valid"] is True

    def test_validate_config_unsupported_rate(self, client, mock_valid_capability):
        """Unsupported output rate should fail validation."""
        with patch(
            "web.routers.dac.scan_dac_capability", return_value=mock_valid_capability
        ):
            response = client.get(
                "/dac/validate-config?device=hw:0&input_rate=44100&output_rate=500000"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["is_supported"] is False
        assert data["data"]["rate_supported"] is False

    def test_validate_config_invalid_ratio(self, client, mock_valid_capability):
        """Invalid upsampling ratio should fail validation."""
        # 44100 * 3 = 132300 (not a valid ratio)
        with patch(
            "web.routers.dac.scan_dac_capability", return_value=mock_valid_capability
        ):
            response = client.get(
                "/dac/validate-config?device=hw:0&input_rate=44100&output_rate=132300"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["ratio_valid"] is False

    def test_validate_config_device_error(self, client, mock_invalid_capability):
        """Device scan failure should return success=False."""
        with patch(
            "web.routers.dac.scan_dac_capability", return_value=mock_invalid_capability
        ):
            response = client.get(
                "/dac/validate-config?device=hw:99&input_rate=44100&output_rate=705600"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Cannot scan device" in data["message"]

    def test_validate_config_invalid_device(self, client):
        """Invalid device name should return 400."""
        response = client.get(
            "/dac/validate-config?device=../etc/passwd&input_rate=44100&output_rate=705600"
        )

        assert response.status_code == 400


class DummyDaemonClient:
    """Context manager friendly fake ZeroMQ client."""

    def __init__(self, *, status_response=None, select_response=None, rescan_response=None):
        self._status_response = status_response
        self._select_response = select_response
        self._rescan_response = rescan_response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def dac_status(self):
        return self._status_response

    def dac_select(self, device: str):
        return self._select_response

    def dac_rescan(self):
        return self._rescan_response


class TestRuntimeDacEndpoints:
    """Tests for new runtime DAC management endpoints."""

    def test_dac_state_success(self, client):
        """Should return daemon state payload."""
        payload = {
            "requested_device": "hw:1",
            "selected_device": "hw:1",
            "active_device": "hw:1",
            "change_pending": False,
            "device_count": 2,
            "devices": [
                {"id": "default", "is_active": False},
                {"id": "hw:1", "is_active": True},
            ],
            "capability": {
                "device": "hw:1",
                "is_valid": True,
                "min_rate": 44100,
                "max_rate": 768000,
                "supported_rates": [44100, 705600],
                "max_channels": 2,
            },
            "output_rate": 705600,
        }
        fake_client = DummyDaemonClient(
            status_response=DaemonResponse(success=True, data=payload)
        )

        with patch("web.routers.dac.get_daemon_client", return_value=fake_client):
            response = client.get("/dac/state")

        assert response.status_code == 200
        data = response.json()
        assert data["active_device"] == "hw:1"
        assert data["capability"]["is_valid"] is True

    def test_dac_state_error_propagates(self, client):
        """Daemon errors should propagate to HTTP layer."""
        fake_client = DummyDaemonClient(
            status_response=DaemonResponse(
                success=False,
                error=DaemonError(
                    error_code="IPC_TIMEOUT",
                    message="Timeout",
                ),
            )
        )

        with patch("web.routers.dac.get_daemon_client", return_value=fake_client):
            response = client.get("/dac/state")

        # DaemonError propagates its mapped HTTP status (timeout -> 504).
        assert response.status_code == 504

    def test_dac_select_persist_success(self, client):
        """Selecting a device with persist flag updates config."""
        fake_client = DummyDaemonClient(
            select_response=DaemonResponse(success=True, data={"selected_device": "hw:2"})
        )
        fake_settings = SimpleNamespace(alsa_device="hw:0")

        with patch("web.routers.dac.get_daemon_client", return_value=fake_client):
            with patch("web.routers.dac.load_config", return_value=fake_settings) as mock_load:
                with patch("web.routers.dac.save_config", return_value=True) as mock_save:
                    response = client.post(
                        "/dac/select", json={"device": "hw:2", "persist": True}
                    )

        assert response.status_code == 200
        mock_load.assert_called_once()
        mock_save.assert_called_once()
        assert fake_settings.alsa_device == "hw:2"

    def test_dac_select_invalid_device(self, client):
        """Invalid device names should be rejected."""
        response = client.post("/dac/select", json={"device": "../etc/passwd"})
        assert response.status_code == 400

    def test_dac_select_persist_failure(self, client):
        """Persist failure should return 500."""
        fake_client = DummyDaemonClient(
            select_response=DaemonResponse(success=True, data={"selected_device": "hw:3"})
        )
        fake_settings = SimpleNamespace(alsa_device="hw:0")

        with patch("web.routers.dac.get_daemon_client", return_value=fake_client):
            with patch("web.routers.dac.load_config", return_value=fake_settings):
                with patch("web.routers.dac.save_config", return_value=False):
                    response = client.post(
                        "/dac/select", json={"device": "hw:3", "persist": True}
                    )

        assert response.status_code == 500

    def test_dac_rescan(self, client):
        """Rescan endpoint should proxy response."""
        fake_client = DummyDaemonClient(
            rescan_response=DaemonResponse(
                success=True, data={"message": "rescan scheduled"}
            )
        )

        with patch("web.routers.dac.get_daemon_client", return_value=fake_client):
            response = client.post("/dac/rescan")

        assert response.status_code == 200
        assert response.json()["data"]["message"] == "rescan scheduled"


class TestDeviceNameValidation:
    """Tests for device name validation."""

    @pytest.mark.parametrize(
        "device",
        [
            "hw:0",
            "hw:0,0",
            "hw:1,1",
            "hw:99",
            "plughw:0",
            "plughw:0,0",
            "default",
            "sysdefault",
            "sysdefault:CARD=PCH",
        ],
    )
    def test_valid_device_names(self, client, mock_valid_capability, device):
        """Valid device names should be accepted."""
        with patch(
            "web.routers.dac.scan_dac_capability", return_value=mock_valid_capability
        ):
            response = client.get(f"/dac/capabilities?device={device}")

        assert response.status_code == 200

    @pytest.mark.parametrize(
        "device",
        [
            "../etc/passwd",
            "hw:0/../../../etc",
            "/dev/sda",
            "hw:0; rm -rf /",
            "hw:0`id`",
            "hw:0$(whoami)",
            "hw:100",  # Card number > 99
            "hw:0,100",  # Device number > 99
        ],
    )
    def test_invalid_device_names(self, client, device):
        """Invalid device names should be rejected."""
        response = client.get(f"/dac/capabilities?device={device}")

        assert response.status_code == 400


class TestDacServiceUnit:
    """Unit tests for DAC service functions."""

    def test_is_safe_device_name_valid(self):
        """Valid device names should pass."""
        from web.services.dac import is_safe_device_name

        assert is_safe_device_name("hw:0") is True
        assert is_safe_device_name("hw:0,0") is True
        assert is_safe_device_name("plughw:0") is True
        assert is_safe_device_name("default") is True
        assert is_safe_device_name("sysdefault:CARD=PCH") is True

    def test_is_safe_device_name_invalid(self):
        """Invalid device names should fail."""
        from web.services.dac import is_safe_device_name

        assert is_safe_device_name("../etc/passwd") is False
        assert is_safe_device_name("hw:0/../../../etc") is False
        assert is_safe_device_name("/dev/sda") is False

    def test_parse_device_name(self):
        """Device name parsing should extract card/device numbers."""
        from web.services.dac import _parse_device_name

        assert _parse_device_name("hw:0") == (0, None)
        assert _parse_device_name("hw:0,0") == (0, 0)
        assert _parse_device_name("hw:1,2") == (1, 2)
        assert _parse_device_name("plughw:0") == (0, None)
        assert _parse_device_name("plughw:0,1") == (0, 1)
        assert _parse_device_name("default") == (None, None)

    def test_capability_cache(self, mock_valid_capability):
        """Cache should prevent repeated scans."""
        from web.services.dac import (
            _capability_cache,
            _get_cached_capability,
            _set_cached_capability,
        )

        # Clear cache
        _capability_cache.clear()

        # First call should return None
        assert _get_cached_capability("hw:0") is None

        # Set cache
        _set_cached_capability("hw:0", mock_valid_capability)

        # Second call should return cached value
        cached = _get_cached_capability("hw:0")
        assert cached is not None
        assert cached.device_name == "hw:0"
        assert cached.is_valid is True

        # Clear cache for other tests
        _capability_cache.clear()


class TestProcParsing:
    """Tests for /proc/asound parsing logic."""

    def test_playback_only_parsing(self, tmp_path):
        """Should only parse Playback section, ignoring Capture rates."""
        from web.services.dac import _scan_from_proc

        # Create a mock /proc/asound/card0/stream0 file
        # Playback: 44100-192000, Capture: 44100-384000
        proc_content = """USB DAC at usb-0000:00:14.0-3, high speed : USB Audio

Playback:
  Status: Stop
  Interface 1
    Altset 1
    Format: S32_LE
    Channels: 2
    Rates: 44100, 48000, 88200, 96000, 176400, 192000

Capture:
  Status: Stop
  Interface 2
    Altset 1
    Format: S32_LE
    Channels: 2
    Rates: 44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000
"""
        # Mock the proc path by patching
        with patch("web.services.dac.Path") as mock_path:
            mock_stream = mock_path.return_value
            mock_stream.exists.return_value = True
            mock_stream.read_text.return_value = proc_content

            result = _scan_from_proc(0)

        assert result is not None
        assert result.is_valid is True
        # Should only have Playback rates (up to 192000)
        assert 192000 in result.supported_rates
        assert 352800 not in result.supported_rates  # Capture-only rate
        assert 384000 not in result.supported_rates  # Capture-only rate
        assert result.max_channels == 2

    def test_capture_only_device_returns_none(self, tmp_path):
        """Capture-only device should return None (no playback capability)."""
        from web.services.dac import _scan_from_proc

        # A capture-only device (like a microphone)
        proc_content = """Microphone at usb-0000:00:14.0-3, high speed : USB Audio

Capture:
  Status: Stop
  Interface 1
    Altset 1
    Format: S16_LE
    Channels: 1
    Rates: 44100, 48000
"""
        with patch("web.services.dac.Path") as mock_path:
            mock_stream = mock_path.return_value
            mock_stream.exists.return_value = True
            mock_stream.read_text.return_value = proc_content

            result = _scan_from_proc(0)

        # Should return None because there's no Playback section
        assert result is None

    def test_mono_dac_channels(self):
        """Mono DAC should report 1 channel, not default to 2."""
        from web.services.dac import _scan_from_proc

        # A hypothetical mono DAC
        proc_content = """Mono DAC at usb-0000:00:14.0-3, high speed : USB Audio

Playback:
  Status: Stop
  Interface 1
    Altset 1
    Format: S32_LE
    Channels: 1
    Rates: 44100, 48000, 96000
"""
        with patch("web.services.dac.Path") as mock_path:
            mock_stream = mock_path.return_value
            mock_stream.exists.return_value = True
            mock_stream.read_text.return_value = proc_content

            result = _scan_from_proc(0)

        assert result is not None
        assert result.max_channels == 1  # Should be 1, not 2

    def test_multiple_altsets_max_channels(self):
        """Should report maximum channels across all altsets."""
        from web.services.dac import _scan_from_proc

        # DAC with multiple altsets, different channel counts
        proc_content = """Multi-channel DAC at usb-0000:00:14.0-3, high speed : USB Audio

Playback:
  Status: Stop
  Interface 1
    Altset 1
    Format: S32_LE
    Channels: 2
    Rates: 44100, 48000
  Interface 1
    Altset 2
    Format: S32_LE
    Channels: 8
    Rates: 44100, 48000, 96000
"""
        with patch("web.services.dac.Path") as mock_path:
            mock_stream = mock_path.return_value
            mock_stream.exists.return_value = True
            mock_stream.read_text.return_value = proc_content

            result = _scan_from_proc(0)

        assert result is not None
        assert result.max_channels == 8  # Should be max across altsets
