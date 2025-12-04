"""Tests for REST API design improvements (Issue #129).

This module tests:
- Unified error response format (ErrorResponse)
- Deprecated endpoint behavior
- OpenAPI schema validation
"""

import pytest
from fastapi.testclient import TestClient

from web.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app, raise_server_exceptions=False)


class TestUnifiedErrorResponse:
    """Test that all error responses use the ErrorResponse format."""

    def test_404_error_has_unified_format(self, client: TestClient):
        """404 errors should return ErrorResponse format."""
        # DELETE a non-existent profile
        response = client.delete("/eq/profiles/nonexistent_profile_xyz_123")
        assert response.status_code == 404
        data = response.json()
        # Should have ErrorResponse fields
        assert "detail" in data
        assert "error_code" in data
        assert data["error_code"] == "HTTP_404"

    def test_400_error_has_unified_format(self, client: TestClient):
        """400 errors should return ErrorResponse format."""
        # Try to activate a profile with path traversal (dotdot)
        response = client.post("/eq/activate/..test")
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "error_code" in data
        assert data["error_code"] == "HTTP_400"

    def test_validation_error_has_unified_format(self, client: TestClient):
        """Validation errors should return ErrorResponse format with VALIDATION_ERROR code."""
        # Send invalid settings update (wrong type)
        response = client.post(
            "/settings",
            json={"upsample_ratio": "not_a_number"},
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "error_code" in data
        assert data["error_code"] == "VALIDATION_ERROR"

    def test_error_detail_can_be_string(self, client: TestClient):
        """Error detail should preserve string format."""
        response = client.delete("/eq/profiles/nonexistent_profile")
        assert response.status_code == 404
        data = response.json()
        # String detail should remain a string
        assert isinstance(data["detail"], str)
        assert "nonexistent_profile" in data["detail"]


class TestDeprecatedEndpoint:
    """Test deprecated /restart endpoint."""

    def test_deprecated_restart_endpoint_works(self, client: TestClient):
        """Deprecated /restart should still work but forward to /daemon/restart."""
        response = client.post("/restart")
        # Should return ApiResponse (may fail if daemon not running, but format is correct)
        data = response.json()
        assert "success" in data
        assert "message" in data

    def test_deprecated_restart_in_openapi_schema(self, client: TestClient):
        """Deprecated /restart should be marked deprecated in OpenAPI schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()

        # Find /restart endpoint
        restart_path = schema["paths"].get("/restart", {})
        post_op = restart_path.get("post", {})

        # Should be marked as deprecated
        assert post_op.get("deprecated") is True
        assert "legacy" in post_op.get("tags", [])


class TestOpenAPISchema:
    """Test OpenAPI schema correctness."""

    def test_openapi_schema_accessible(self, client: TestClient):
        """OpenAPI schema should be accessible at /openapi.json."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

    def test_openapi_has_tags(self, client: TestClient):
        """OpenAPI schema should have tag descriptions."""
        response = client.get("/openapi.json")
        schema = response.json()
        tags = {tag["name"]: tag for tag in schema.get("tags", [])}

        # Should have all expected tags
        expected_tags = [
            "status",
            "daemon",
            "eq",
            "opra",
            "input-mode",
            "output",
            "legacy",
        ]
        for tag_name in expected_tags:
            assert tag_name in tags, f"Missing tag: {tag_name}"
            assert (
                "description" in tags[tag_name]
            ), f"Tag {tag_name} missing description"

    def test_all_endpoints_have_response_models(self, client: TestClient):
        """All GET/POST endpoints should have response schemas defined."""
        response = client.get("/openapi.json")
        schema = response.json()

        # Endpoints that should have response schemas
        endpoints_to_check = [
            ("/status", "get"),
            ("/devices", "get"),
            ("/daemon/status", "get"),
            ("/daemon/zmq/ping", "get"),
            ("/api/input-mode/switch", "post"),
            ("/api/output/mode", "get"),
            ("/api/output/mode", "post"),
            ("/eq/profiles", "get"),
            ("/eq/active", "get"),
        ]

        for path, method in endpoints_to_check:
            path_schema = schema["paths"].get(path, {})
            method_schema = path_schema.get(method, {})
            responses = method_schema.get("responses", {})
            success_response = responses.get("200", {})

            # Should have content with schema
            assert (
                "content" in success_response
            ), f"{method.upper()} {path} missing response content"
            json_content = success_response["content"].get("application/json", {})
            assert (
                "schema" in json_content
            ), f"{method.upper()} {path} missing response schema"


class TestResponseModels:
    """Test that response models are correctly applied."""

    def test_status_response_model(self, client: TestClient):
        """GET /status should return Status model fields."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()

        # Should have Status model fields
        required_fields = [
            "settings",
            "pipewire_connected",
            "daemon_running",
            "eq_active",
            "clip_rate",
            "clip_count",
            "total_samples",
            "input_mode",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_devices_response_model(self, client: TestClient):
        """GET /devices should return DevicesResponse model."""
        response = client.get("/devices")
        # May return 500 if ALSA is not available (CI environment)
        if response.status_code == 500:
            pytest.skip("ALSA not available in this environment")
        assert response.status_code == 200
        data = response.json()

        # Should have DevicesResponse model fields
        assert "devices" in data
        assert isinstance(data["devices"], list)

        # If there are devices, check they have AlsaDevice structure
        if len(data["devices"]) > 0:
            device = data["devices"][0]
            assert "id" in device, "AlsaDevice should have 'id' field"
            assert "name" in device, "AlsaDevice should have 'name' field"
            # description is optional

    def test_daemon_status_response_model(self, client: TestClient):
        """GET /daemon/status should return DaemonStatus model."""
        response = client.get("/daemon/status")
        assert response.status_code == 200
        data = response.json()

        # Should have DaemonStatus model fields
        required_fields = [
            "running",
            "pid_file",
            "binary_path",
            "pipewire_connected",
            "input_mode",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_eq_profiles_response_model(self, client: TestClient):
        """GET /eq/profiles should return EqProfilesResponse model."""
        response = client.get("/eq/profiles")
        assert response.status_code == 200
        data = response.json()

        # Should have EqProfilesResponse model fields
        assert "profiles" in data
        assert isinstance(data["profiles"], list)

        # If there are profiles, check structure
        if len(data["profiles"]) > 0:
            # Each profile should have EqProfileInfo fields
            profile_data = data["profiles"][0]
            required_fields = [
                "name",
                "filename",
                "path",
                "size",
                "modified",
                "type",
                "filter_count",
            ]
            for field in required_fields:
                assert field in profile_data, f"Missing profile field: {field}"

            # Check types
            assert isinstance(profile_data["name"], str)
            assert isinstance(profile_data["size"], int)
            assert isinstance(profile_data["filter_count"], int)
            assert profile_data["type"] in ["opra", "custom"]

    def test_zmq_ping_response_model(self, client: TestClient):
        """GET /daemon/zmq/ping should return ZmqPingResponse model."""
        response = client.get("/daemon/zmq/ping")
        assert response.status_code == 200
        data = response.json()

        # Should have ZmqPingResponse model fields
        required_fields = ["success", "daemon_running"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


class TestPhaseTypeEndpoints:
    """Test phase type API endpoints (Issue #197)."""

    def test_get_phase_type_response_model(self, client: TestClient):
        """GET /daemon/phase-type should return PhaseTypeResponse model."""
        response = client.get("/daemon/phase-type")
        # May return 503/504 if daemon not running or timeout
        if response.status_code in (503, 504):
            pytest.skip("Daemon not running or timeout")
        assert response.status_code == 200
        data = response.json()

        # Should have PhaseTypeResponse model fields
        assert "phase_type" in data, "Missing field: phase_type"
        assert data["phase_type"] in ["minimum", "linear"], "Invalid phase_type value"
        # latency_warning is optional (null for minimum phase)
        assert "latency_warning" in data

    def test_get_phase_type_in_openapi_schema(self, client: TestClient):
        """GET /daemon/phase-type should be in OpenAPI schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()

        # Check endpoint exists
        phase_type_path = schema["paths"].get("/daemon/phase-type", {})
        assert "get" in phase_type_path, "GET /daemon/phase-type missing in OpenAPI"
        assert "put" in phase_type_path, "PUT /daemon/phase-type missing in OpenAPI"

        # Check response schema
        get_op = phase_type_path["get"]
        responses = get_op.get("responses", {})
        success_response = responses.get("200", {})
        assert (
            "content" in success_response
        ), "GET /daemon/phase-type missing response content"

    def test_put_phase_type_validation_error(self, client: TestClient):
        """PUT /daemon/phase-type with invalid value should return 422 (Pydantic Literal validation)."""
        response = client.put(
            "/daemon/phase-type",
            json={"phase_type": "invalid_value"},
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "error_code" in data
        assert data["error_code"] == "VALIDATION_ERROR"

    def test_put_phase_type_missing_field(self, client: TestClient):
        """PUT /daemon/phase-type with missing field should return 422."""
        response = client.put(
            "/daemon/phase-type",
            json={},
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "error_code" in data
        assert data["error_code"] == "VALIDATION_ERROR"
