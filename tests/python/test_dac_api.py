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
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from web.main import app  # noqa: E402
from web.services.dac import DacCapability  # noqa: E402


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
