"""
Regression tests to ensure legacy/admin pages stay removed (Issue #718).
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


def test_admin_route_removed(client):
    """`/admin` should be gone."""
    response = client.get("/admin")
    assert response.status_code == 404


def test_legacy_route_removed(client):
    """`/legacy` should be gone."""
    response = client.get("/legacy")
    assert response.status_code == 404
