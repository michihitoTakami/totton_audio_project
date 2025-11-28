"""
Tests for the admin UI HTML template (Issue #354).

Focus:
- `/admin` renders successfully
- Partitioned convolution controls exist in the DOM
- JavaScript references the new API helpers
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure `web` package can be imported when running pytest directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from web.main import app  # noqa: E402


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app, raise_server_exceptions=False)


class TestAdminPartitionUi:
    """Partitioned convolution UI regression tests."""

    def test_partition_section_exists(self, client):
        """Section heading and form fields should be rendered."""
        response = client.get("/admin")
        assert response.status_code == 200
        html = response.text
        assert "低遅延パーティション" in html
        for element_id in [
            "partitionToggle",
            "fastPartitionTaps",
            "minPartitionTaps",
            "maxPartitions",
            "tailFftMultiple",
            "partitionSaveBtn",
            "partitionResetBtn",
        ]:
            assert f'id="{element_id}"' in html

    def test_javascript_references_partition_api(self, client):
        """Client-side helpers should call the new API endpoints."""
        html = client.get("/admin").text
        assert "/partitioned-convolution" in html
        assert "fetchPartitionSettings" in html
        assert "savePartitionSettings" in html

