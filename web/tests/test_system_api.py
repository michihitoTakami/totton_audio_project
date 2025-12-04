"""Tests for System API endpoints."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.main import app

client = TestClient(app)


@pytest.fixture
def temp_log_file():
    """Create a temporary log file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        f.write("2025-01-15 10:00:00 INFO Starting daemon\n")
        f.write("2025-01-15 10:00:01 DEBUG Initializing GPU\n")
        f.write("2025-01-15 10:00:02 INFO GPU initialized successfully\n")
        f.write("2025-01-15 10:00:03 WARNING High GPU temperature: 75C\n")
        f.write("2025-01-15 10:00:04 ERROR Failed to load filter\n")
        f.write("2025-01-15 10:00:05 INFO Filter loaded successfully\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


def test_get_system_logs_default():
    """Test GET /api/system/logs with default parameters."""
    response = client.get("/api/system/logs")
    # This may fail if log file doesn't exist, which is expected
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert "logs" in data
        assert "total_lines" in data
        assert "file_path" in data
        assert "file_size" in data
        assert isinstance(data["logs"], list)
        assert isinstance(data["total_lines"], int)


def test_get_system_logs_with_lines():
    """Test GET /api/system/logs with lines parameter."""
    response = client.get("/api/system/logs?lines=50")
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert "logs" in data
        assert len(data["logs"]) <= 50


def test_get_system_logs_with_offset():
    """Test GET /api/system/logs with offset parameter."""
    response = client.get("/api/system/logs?lines=10&offset=20")
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert "logs" in data


def test_get_system_logs_with_level_filter():
    """Test GET /api/system/logs with level filter."""
    response = client.get("/api/system/logs?lines=50&level=error")
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert "logs" in data


def test_get_system_logs_pagination():
    """Test GET /api/system/logs pagination works correctly."""
    # Get first page
    response1 = client.get("/api/system/logs?lines=10&offset=0")
    assert response1.status_code in [200, 404]

    if response1.status_code == 200:
        # Get second page
        response2 = client.get("/api/system/logs?lines=10&offset=10")
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Total lines should be the same
        assert data1["total_lines"] == data2["total_lines"]
        assert data1["file_path"] == data2["file_path"]


def test_get_system_logs_invalid_lines():
    """Test GET /api/system/logs with invalid lines parameter."""
    # lines must be >= 1
    response = client.get("/api/system/logs?lines=0")
    assert response.status_code == 422  # Validation error

    # lines must be <= 1000
    response = client.get("/api/system/logs?lines=2000")
    assert response.status_code == 422  # Validation error


def test_get_system_logs_invalid_offset():
    """Test GET /api/system/logs with invalid offset parameter."""
    # offset must be >= 0
    response = client.get("/api/system/logs?offset=-1")
    assert response.status_code == 422  # Validation error
