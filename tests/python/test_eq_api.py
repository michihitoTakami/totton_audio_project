"""
Integration tests for EQ profile API endpoints.

Tests the following endpoints:
- POST /eq/validate - Validate uploaded file
- POST /eq/import - Import profile
- POST /eq/import-text - Import profile from pasted text
- GET /eq/profiles - List profiles
- POST /eq/activate/{name} - Activate profile
- POST /eq/deactivate - Deactivate EQ
- DELETE /eq/profiles/{name} - Delete profile
- GET /eq/active - Get active profile
"""

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from web.main import app  # noqa: E402
from web.services.config import load_config  # noqa: E402
from web.constants import MAX_EQ_FILE_SIZE  # noqa: E402


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def valid_eq_content():
    """Valid EQ profile content."""
    return """Preamp: -6.5 dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
Filter 2: ON PK Fc 1000 Hz Gain 3.5 dB Q 2.0
Filter 3: ON PK Fc 5000 Hz Gain -1.5 dB Q 1.5
"""


@pytest.fixture
def invalid_eq_content():
    """Invalid EQ profile content (missing Preamp)."""
    return """Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
"""


@pytest.fixture
def eq_profile_dir(tmp_path, monkeypatch):
    """Create temporary EQ profile directory and patch constants."""
    eq_dir = tmp_path / "data" / "EQ"
    eq_dir.mkdir(parents=True)

    # Patch EQ_PROFILES_DIR in constants and all modules that import it
    monkeypatch.setattr("web.constants.EQ_PROFILES_DIR", eq_dir)
    monkeypatch.setattr("web.routers.eq.EQ_PROFILES_DIR", eq_dir)
    monkeypatch.setattr("web.routers.opra.EQ_PROFILES_DIR", eq_dir)
    monkeypatch.setattr("web.services.config.EQ_PROFILES_DIR", eq_dir)

    return eq_dir


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    """Create temporary config file and patch constants."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"alsaDevice": "default", "upsampleRatio": 8}')

    monkeypatch.setattr("web.constants.CONFIG_PATH", config_file)
    monkeypatch.setattr("web.services.config.CONFIG_PATH", config_file)

    return config_file


@pytest.fixture(autouse=True)
def disable_daemon_reload(monkeypatch):
    """Ensure Eq router never tries to reload a real daemon during tests."""
    monkeypatch.setattr("web.routers.eq.check_daemon_running", lambda: False)


class TestValidateEndpoint:
    """Tests for POST /eq/validate endpoint."""

    def test_validate_valid_file(self, client, valid_eq_content):
        """Valid file should pass validation."""
        files = {"file": ("test_profile.txt", io.BytesIO(valid_eq_content.encode()))}
        response = client.post("/eq/validate", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["filter_count"] == 3
        assert data["preamp_db"] == -6.5
        assert data["filename"] == "test_profile.txt"
        assert len(data["errors"]) == 0

    def test_validate_invalid_file(self, client, invalid_eq_content):
        """Invalid file should fail validation."""
        files = {"file": ("test_profile.txt", io.BytesIO(invalid_eq_content.encode()))}
        response = client.post("/eq/validate", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_validate_empty_file(self, client):
        """Empty file should fail validation."""
        files = {"file": ("empty.txt", io.BytesIO(b""))}
        response = client.post("/eq/validate", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("Empty" in e for e in data["errors"])

    def test_validate_oversized_file(self, client):
        """File exceeding size limit should be rejected."""
        # Create a file larger than 1MB
        large_content = "x" * (1024 * 1024 + 1)
        files = {"file": ("large.txt", io.BytesIO(large_content.encode()))}
        response = client.post("/eq/validate", files=files)

        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()

    def test_validate_unsafe_filename(self, client, valid_eq_content):
        """Unsafe filename should be sanitized."""
        files = {
            "file": ("../../../etc/passwd.txt", io.BytesIO(valid_eq_content.encode()))
        }
        response = client.post("/eq/validate", files=files)

        assert response.status_code == 200
        data = response.json()
        # Should extract safe basename
        assert data["filename"] == "passwd.txt"

    def test_validate_invalid_extension(self, client, valid_eq_content):
        """Invalid file extension should be rejected."""
        files = {"file": ("profile.exe", io.BytesIO(valid_eq_content.encode()))}
        response = client.post("/eq/validate", files=files)

        assert response.status_code == 400
        assert ".txt" in response.json()["detail"].lower()


class TestImportEndpoint:
    """Tests for POST /eq/import endpoint."""

    def test_import_valid_file(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Valid file should be imported successfully."""
        files = {"file": ("test_profile.txt", io.BytesIO(valid_eq_content.encode()))}
        response = client.post("/eq/import", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "imported" in data["message"].lower()

        # Verify file was created
        assert (eq_profile_dir / "test_profile.txt").exists()

    def test_import_invalid_file_rejected(
        self, client, invalid_eq_content, eq_profile_dir, config_path
    ):
        """Invalid file should be rejected."""
        files = {"file": ("test_profile.txt", io.BytesIO(invalid_eq_content.encode()))}
        response = client.post("/eq/import", files=files)

        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

        # Verify file was NOT created
        assert not (eq_profile_dir / "test_profile.txt").exists()

    def test_import_overwrite_protection(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Existing file should not be overwritten without flag."""
        # Create existing file
        existing_file = eq_profile_dir / "existing.txt"
        existing_file.write_text("original content")

        files = {"file": ("existing.txt", io.BytesIO(valid_eq_content.encode()))}
        response = client.post("/eq/import", files=files)

        assert response.status_code == 409
        assert "exists" in response.json()["detail"].lower()

        # Verify original content unchanged
        assert existing_file.read_text() == "original content"

    def test_import_overwrite_with_flag(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Existing file should be overwritten with flag."""
        # Create existing file
        existing_file = eq_profile_dir / "existing.txt"
        existing_file.write_text("original content")

        files = {"file": ("existing.txt", io.BytesIO(valid_eq_content.encode()))}
        response = client.post("/eq/import?overwrite=true", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify content was updated
        assert existing_file.read_text() == valid_eq_content


class TestImportTextEndpoint:
    """Tests for POST /eq/import-text endpoint."""

    def test_import_text_valid_profile(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Should import text payload successfully."""
        payload = {"name": "text_profile", "content": valid_eq_content}
        response = client.post("/eq/import-text", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert (eq_profile_dir / "text_profile.txt").read_text() == valid_eq_content

    def test_import_text_accepts_name_with_extension(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Name ending with .txt should be normalized."""
        payload = {"name": "text_profile.txt", "content": valid_eq_content}
        response = client.post("/eq/import-text", json=payload)

        assert response.status_code == 200
        assert (eq_profile_dir / "text_profile.txt").exists()

    def test_import_text_invalid_content_rejected(
        self, client, invalid_eq_content, eq_profile_dir, config_path
    ):
        """Invalid EQ content should return HTTP 400."""
        payload = {"name": "bad_profile", "content": invalid_eq_content}
        response = client.post("/eq/import-text", json=payload)

        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

    def test_import_text_empty_content_rejected(
        self, client, eq_profile_dir, config_path
    ):
        """Blank content should be rejected."""
        payload = {"name": "blank_profile", "content": "   "}
        response = client.post("/eq/import-text", json=payload)

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_import_text_overwrite_protection(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Existing file should not be overwritten without flag."""
        (eq_profile_dir / "text_profile.txt").write_text("original")

        payload = {"name": "text_profile", "content": valid_eq_content}
        response = client.post("/eq/import-text", json=payload)

        assert response.status_code == 409
        assert (eq_profile_dir / "text_profile.txt").read_text() == "original"

    def test_import_text_overwrite_with_flag(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Existing file should overwrite when flag is true."""
        (eq_profile_dir / "text_profile.txt").write_text("original")

        payload = {"name": "text_profile", "content": valid_eq_content}
        response = client.post("/eq/import-text?overwrite=true", json=payload)

        assert response.status_code == 200
        assert (eq_profile_dir / "text_profile.txt").read_text() == valid_eq_content

    def test_import_text_invalid_name_rejected(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Unsafe profile names should be rejected."""
        payload = {"name": "..bad", "content": valid_eq_content}
        response = client.post("/eq/import-text", json=payload)

        assert response.status_code == 400
        assert ".." in response.json()["detail"]

    def test_import_text_too_large(self, client, eq_profile_dir, config_path):
        """Payload larger than MAX_EQ_FILE_SIZE should be rejected."""
        payload = {
            "name": "huge",
            "content": "x" * (MAX_EQ_FILE_SIZE + 1),
        }
        response = client.post("/eq/import-text", json=payload)

        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()


class TestProfilesEndpoint:
    """Tests for GET /eq/profiles endpoint."""

    def test_list_empty_profiles(self, client, eq_profile_dir, config_path):
        """Empty directory should return empty list."""
        response = client.get("/eq/profiles")

        assert response.status_code == 200
        data = response.json()
        assert data["profiles"] == []

    def test_list_profiles(self, client, valid_eq_content, eq_profile_dir, config_path):
        """Should list all profile files."""
        # Create some profile files
        (eq_profile_dir / "profile1.txt").write_text(valid_eq_content)
        (eq_profile_dir / "profile2.txt").write_text(valid_eq_content)

        response = client.get("/eq/profiles")

        assert response.status_code == 200
        data = response.json()
        assert len(data["profiles"]) == 2

        names = [p["name"] for p in data["profiles"]]
        assert "profile1" in names
        assert "profile2" in names

    def test_list_profiles_with_metadata(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Profile listing should include metadata."""
        (eq_profile_dir / "test.txt").write_text(valid_eq_content)

        response = client.get("/eq/profiles")

        assert response.status_code == 200
        profile = response.json()["profiles"][0]

        assert "name" in profile
        assert "filename" in profile
        assert "size" in profile
        assert "filter_count" in profile
        assert "type" in profile

    def test_list_profiles_excludes_opra_profiles(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """OPRA-generated profiles should be hidden from the saved list."""
        opra_content = """# OPRA: Test Headphone
# Author: test
Preamp: -6 dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
"""
        (eq_profile_dir / "opra_test.txt").write_text(opra_content)
        (eq_profile_dir / "custom.txt").write_text(valid_eq_content)

        response = client.get("/eq/profiles")

        assert response.status_code == 200
        data = response.json()
        names = [p["name"] for p in data["profiles"]]
        assert "custom" in names
        assert "opra_test" not in names


class TestActivateEndpoint:
    """Tests for POST /eq/activate/{name} endpoint."""

    def test_activate_existing_profile(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Should activate existing profile."""
        (eq_profile_dir / "test.txt").write_text(valid_eq_content)

        response = client.post("/eq/activate/test")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["restart_required"] is False
        assert data["data"]["daemon_running"] is False
        assert data["data"]["daemon_reloaded"] is False

    def test_activate_triggers_reload_when_daemon_online(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Activation should reload the daemon when it is running."""
        (eq_profile_dir / "test.txt").write_text(valid_eq_content)

        with patch(
            "web.routers.eq.check_daemon_running",
            return_value=True,
        ) as mock_running, patch(
            "web.routers.eq.get_daemon_client",
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.reload_config.return_value = (True, "Command executed")
            mock_get_client.return_value = mock_client

            response = client.post("/eq/activate/test")

        mock_running.assert_called_once()
        mock_get_client.assert_called_once()
        mock_client.reload_config.assert_called_once()

        assert response.status_code == 200
        body = response.json()
        assert body["restart_required"] is False
        assert body["data"]["daemon_running"] is True
        assert body["data"]["daemon_reloaded"] is True

    def test_activate_restart_required_when_reload_fails(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """If the reload command fails we still flag restart_required."""
        (eq_profile_dir / "test.txt").write_text(valid_eq_content)

        with patch(
            "web.routers.eq.check_daemon_running",
            return_value=True,
        ), patch(
            "web.routers.eq.get_daemon_client",
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.reload_config.return_value = (False, "Reload failure")
            mock_get_client.return_value = mock_client

            response = client.post("/eq/activate/test")

        body = response.json()
        assert response.status_code == 200
        assert body["restart_required"] is True
        assert body["data"]["daemon_running"] is True
        assert body["data"]["daemon_reloaded"] is False
        assert body["data"]["reload_error"] == "Reload failure"

    def test_activate_nonexistent_profile(self, client, eq_profile_dir, config_path):
        """Should return 404 for nonexistent profile."""
        response = client.post("/eq/activate/nonexistent")

        assert response.status_code == 404


class TestDeactivateEndpoint:
    """Tests for POST /eq/deactivate endpoint."""

    def test_deactivate_eq(self, client, config_path):
        """Should deactivate EQ."""
        response = client.post("/eq/deactivate")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["restart_required"] is False
        assert data["data"]["daemon_running"] is False
        assert data["data"]["daemon_reloaded"] is False


class TestEqConfigPersistence:
    """Tests ensuring EQ fields are persisted in config for the daemon."""

    def test_activate_sets_eq_enabled_and_path(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Activation should write eqEnabled and eqProfilePath."""
        (eq_profile_dir / "test.txt").write_text(valid_eq_content)

        response = client.post("/eq/activate/test")
        assert response.status_code == 200

        config_data = json.loads(config_path.read_text())
        assert config_data["eqEnabled"] is True
        assert config_data["eqProfile"] == "test"
        assert config_data["eqProfilePath"] == str(eq_profile_dir / "test.txt")

    def test_deactivate_clears_eq_fields(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Deactivation should clear eqEnabled/eqProfilePath."""
        config_path.write_text(
            json.dumps(
                {
                    "alsaDevice": "default",
                    "upsampleRatio": 8,
                    "eqEnabled": True,
                    "eqProfile": "active",
                    "eqProfilePath": str(eq_profile_dir / "active.txt"),
                }
            )
        )

        response = client.post("/eq/deactivate")
        assert response.status_code == 200

        config_data = json.loads(config_path.read_text())
        assert config_data["eqEnabled"] is False
        assert config_data["eqProfile"] is None
        assert config_data["eqProfilePath"] is None

    def test_status_eq_active_true_when_enabled_with_path(
        self, client, eq_profile_dir, config_path
    ):
        """Status should show eq_active when enabled and path present."""
        config_path.write_text(
            json.dumps(
                {
                    "alsaDevice": "default",
                    "upsampleRatio": 8,
                    "eqEnabled": True,
                    "eqProfile": "present",
                    "eqProfilePath": str(eq_profile_dir / "present.txt"),
                }
            )
        )

        response = client.get("/status")
        assert response.status_code == 200
        assert response.json()["eq_active"] is True

    def test_status_eq_active_false_without_path(self, client, config_path):
        """Status should show eq_active false when path is missing."""
        config_path.write_text(
            json.dumps(
                {
                    "alsaDevice": "default",
                    "upsampleRatio": 8,
                    "eqEnabled": True,
                    "eqProfile": "no_path",
                }
            )
        )

        response = client.get("/status")
        assert response.status_code == 200
        assert response.json()["eq_active"] is False

    def test_get_active_missing_file_returns_error(
        self, client, eq_profile_dir, config_path
    ):
        """Missing profile file should return active with error message."""
        missing_path = eq_profile_dir / "missing.txt"
        config_path.write_text(
            json.dumps(
                {
                    "alsaDevice": "default",
                    "upsampleRatio": 8,
                    "eqEnabled": True,
                    "eqProfile": "missing",
                    "eqProfilePath": str(missing_path),
                }
            )
        )

        response = client.get("/eq/active")
        assert response.status_code == 200
        data = response.json()
        assert data["active"] is True
        assert "not found" in data["error"].lower()

    def test_load_config_migrates_eq_profile_only(self, eq_profile_dir, config_path):
        """eqProfile alone should migrate to path and enable flag."""
        config_path.write_text(
            json.dumps(
                {
                    "alsaDevice": "default",
                    "upsampleRatio": 8,
                    "eqProfile": "migrate_me",
                }
            )
        )

        cfg = load_config()
        assert cfg.eq_profile == "migrate_me"
        assert cfg.eq_profile_path == str(eq_profile_dir / "migrate_me.txt")
        assert cfg.eq_enabled is True


class TestDeleteEndpoint:
    """Tests for DELETE /eq/profiles/{name} endpoint."""

    def test_delete_existing_profile(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Should delete existing profile."""
        profile_file = eq_profile_dir / "to_delete.txt"
        profile_file.write_text(valid_eq_content)

        response = client.delete("/eq/profiles/to_delete")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert not profile_file.exists()

    def test_delete_nonexistent_profile(self, client, eq_profile_dir, config_path):
        """Should return 404 for nonexistent profile."""
        response = client.delete("/eq/profiles/nonexistent")

        assert response.status_code == 404

    def test_delete_active_profile_deactivates(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Deleting active profile should deactivate EQ first."""
        # Create and activate a profile
        profile_file = eq_profile_dir / "active_profile.txt"
        profile_file.write_text(valid_eq_content)
        client.post("/eq/activate/active_profile")

        # Verify it's active
        active_response = client.get("/eq/active")
        assert active_response.json()["active"] is True
        assert active_response.json()["name"] == "active_profile"

        # Delete the active profile
        response = client.delete("/eq/profiles/active_profile")
        assert response.status_code == 200

        # Verify EQ is now deactivated
        active_response = client.get("/eq/active")
        assert active_response.json()["active"] is False
        assert active_response.json()["name"] is None


class TestActiveEndpoint:
    """Tests for GET /eq/active endpoint."""

    def test_get_active_when_none(self, client, config_path):
        """Should return inactive state when no EQ active."""
        response = client.get("/eq/active")

        assert response.status_code == 200
        data = response.json()
        assert data["active"] is False
        assert data["name"] is None

    def test_get_active_profile(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Should return active profile info."""
        # Create and activate a profile
        (eq_profile_dir / "active_test.txt").write_text(valid_eq_content)
        client.post("/eq/activate/active_test")

        response = client.get("/eq/active")

        assert response.status_code == 200
        data = response.json()
        assert data["active"] is True
        assert data["name"] == "active_test"


class TestSecurityFeatures:
    """Tests for security features.

    Security is enforced at multiple layers:
    1. FastAPI routing: Paths with slashes (even URL-encoded) don't match {name} parameter
    2. validate_profile_name(): Rejects '..' sequences and names starting with '.'
    3. sanitize_filename(): Rejects unsafe characters in uploaded filenames
    """

    def test_path_traversal_with_slashes_blocked_by_routing(
        self, client, eq_profile_dir, config_path
    ):
        """Path traversal with slashes is blocked at routing layer (404)."""
        # FastAPI routing treats slashes as path separators, so these don't reach our handler
        response = client.post("/eq/activate/../../../etc/passwd")
        assert response.status_code == 404  # Route not found

        response = client.delete("/eq/profiles/../../../etc/passwd")
        assert response.status_code == 404  # Route not found

    def test_dotdot_in_profile_name_rejected(self, client, eq_profile_dir, config_path):
        """Profile name containing '..' should be rejected at validation layer."""
        response = client.post("/eq/activate/valid..name")

        assert response.status_code == 400
        assert ".." in response.json()["detail"]

    def test_dotdot_in_delete_rejected(self, client, eq_profile_dir, config_path):
        """Delete with '..' in name should be rejected."""
        response = client.delete("/eq/profiles/valid..name")

        assert response.status_code == 400
        assert ".." in response.json()["detail"]

    def test_hidden_file_profile_name_rejected(
        self, client, eq_profile_dir, config_path
    ):
        """Profile name starting with '.' should be rejected."""
        response = client.post("/eq/activate/.hidden")

        assert response.status_code == 400
        assert "." in response.json()["detail"]

    def test_hidden_file_delete_rejected(self, client, eq_profile_dir, config_path):
        """Delete profile starting with '.' should be rejected."""
        response = client.delete("/eq/profiles/.hidden")

        assert response.status_code == 400

    def test_traversal_does_not_access_outside_files(
        self, client, valid_eq_content, eq_profile_dir, config_path, tmp_path
    ):
        """Even if a file exists outside EQ dir, traversal patterns are rejected."""
        # Create a file outside EQ directory
        outside_file = tmp_path / "target.txt"
        outside_file.write_text(valid_eq_content)

        # Try various traversal patterns - all should be rejected before file access
        # Pattern with '..' (no slashes) - reaches handler, rejected by validation
        response = client.post("/eq/activate/..target")
        assert response.status_code == 400

        response = client.delete("/eq/profiles/..target")
        assert response.status_code == 400

        # File should still exist (not accessed or deleted)
        assert outside_file.exists()

    def test_special_characters_in_filename(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Special characters in uploaded filename should be rejected."""
        files = {"file": ("test;rm -rf /.txt", io.BytesIO(valid_eq_content.encode()))}
        response = client.post("/eq/validate", files=files)

        # Should reject or sanitize
        assert (
            response.status_code == 400
            or response.json()["filename"] != "test;rm -rf /.txt"
        )

    def test_valid_profile_name_allowed(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Valid profile names with safe characters should be allowed."""
        # Create profile with valid name
        (eq_profile_dir / "my-profile_v2.0.txt").write_text(valid_eq_content)

        response = client.post("/eq/activate/my-profile_v2.0")

        assert response.status_code == 200


class TestSettingsSecurityFeatures:
    """Tests for settings endpoint security.

    Validates that path traversal attacks via settings are blocked.
    """

    def test_settings_rejects_traversal_eq_profile(self, client, config_path):
        """Settings should reject eq_profile with path traversal."""
        response = client.post("/settings", json={"eq_profile": "../../../etc/passwd"})

        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

    def test_settings_rejects_dotdot_eq_profile(self, client, config_path):
        """Settings should reject eq_profile containing '..'."""
        response = client.post("/settings", json={"eq_profile": "valid..name"})

        assert response.status_code == 400

    def test_settings_rejects_hidden_eq_profile(self, client, config_path):
        """Settings should reject eq_profile starting with '.'."""
        response = client.post("/settings", json={"eq_profile": ".hidden"})

        assert response.status_code == 400

    def test_settings_accepts_valid_eq_profile(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Settings should accept valid eq_profile names."""
        # Create the profile file
        (eq_profile_dir / "my-valid-profile.txt").write_text(valid_eq_content)

        response = client.post("/settings", json={"eq_profile": "my-valid-profile"})

        assert response.status_code == 200

    def test_get_active_rejects_tampered_config(self, client, config_path, tmp_path):
        """get_active should handle tampered config with unsafe profile name."""
        # Directly write unsafe value to config (simulating tampering)
        # Note: config uses camelCase (eqProfile, not eq_profile)
        config_path.write_text('{"eqProfile": "../../../etc/passwd"}')

        response = client.get("/eq/active")

        # Should return error, not read the file
        assert response.status_code == 200
        data = response.json()
        assert data["active"] is True
        assert "error" in data
        assert "invalid" in data["error"].lower()
