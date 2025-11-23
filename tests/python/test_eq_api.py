"""
Integration tests for EQ profile API endpoints.

Tests the following endpoints:
- POST /eq/validate - Validate uploaded file
- POST /eq/import - Import profile
- GET /eq/profiles - List profiles
- POST /eq/activate/{name} - Activate profile
- POST /eq/deactivate - Deactivate EQ
- DELETE /eq/profiles/{name} - Delete profile
- GET /eq/active - Get active profile
"""

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from web.main import app  # noqa: E402


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

    return eq_dir


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    """Create temporary config file and patch constants."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"alsaDevice": "default", "upsampleRatio": 8}')

    monkeypatch.setattr("web.constants.CONFIG_PATH", config_file)
    monkeypatch.setattr("web.services.config.CONFIG_PATH", config_file)

    return config_file


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

    def test_list_profiles_detects_opra_type(self, client, eq_profile_dir, config_path):
        """Should detect OPRA profile type."""
        opra_content = """# OPRA: Test Headphone
# Author: test
Preamp: -6 dB
Filter 1: ON PK Fc 100 Hz Gain -2.0 dB Q 1.0
"""
        (eq_profile_dir / "opra_test.txt").write_text(opra_content)

        response = client.get("/eq/profiles")

        assert response.status_code == 200
        profile = response.json()["profiles"][0]
        assert profile["type"] == "opra"


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
        assert data["restart_required"] is True

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
        assert data["restart_required"] is True


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
    """Tests for security features."""

    def test_path_traversal_in_activate(self, client, eq_profile_dir, config_path):
        """Path traversal in activate should be prevented."""
        response = client.post("/eq/activate/../../../etc/passwd")

        # Should return 404 (file not found in safe directory)
        assert response.status_code == 404

    def test_path_traversal_in_delete(self, client, eq_profile_dir, config_path):
        """Path traversal in delete should be prevented."""
        response = client.delete("/eq/profiles/../../../etc/passwd")

        # Should return 404
        assert response.status_code == 404

    def test_special_characters_in_filename(
        self, client, valid_eq_content, eq_profile_dir, config_path
    ):
        """Special characters in filename should be rejected."""
        files = {"file": ("test;rm -rf /.txt", io.BytesIO(valid_eq_content.encode()))}
        response = client.post("/eq/validate", files=files)

        # Should reject or sanitize
        assert (
            response.status_code == 400
            or response.json()["filename"] != "test;rm -rf /.txt"
        )
