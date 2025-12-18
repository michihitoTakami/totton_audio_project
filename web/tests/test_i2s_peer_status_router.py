from fastapi.testclient import TestClient

from web.main import create_app


def test_i2s_peer_status_accepts_first_update(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(
        "MAGICBOX_I2S_PEER_STATUS_PATH", str(tmp_path / "peer-status.json")
    )
    app = create_app()
    client = TestClient(app)

    payload = {
        "running": True,
        "mode": "capture",
        "sample_rate": 48000,
        "format": "S32_LE",
        "channels": 2,
        "generation": 1,
        "updated_at_unix_ms": 1000,
        "note": "test",
    }
    resp = client.post("/i2s/peer-status", json=payload)
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    get_resp = client.get("/i2s/peer-status")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["generation"] == 1
    assert data["sample_rate"] == 48000


def test_i2s_peer_status_ignores_stale_generation(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(
        "MAGICBOX_I2S_PEER_STATUS_PATH", str(tmp_path / "peer-status.json")
    )
    app = create_app()
    client = TestClient(app)

    resp1 = client.post(
        "/i2s/peer-status",
        json={
            "running": True,
            "mode": "capture",
            "sample_rate": 44100,
            "format": "S24_3LE",
            "channels": 2,
            "generation": 5,
            "updated_at_unix_ms": 5000,
        },
    )
    assert resp1.status_code == 200

    # generation が小さい=stale
    resp2 = client.post(
        "/i2s/peer-status",
        json={
            "running": True,
            "mode": "capture",
            "sample_rate": 48000,
            "format": "S32_LE",
            "channels": 2,
            "generation": 4,
            "updated_at_unix_ms": 6000,
        },
    )
    assert resp2.status_code == 200

    data = client.get("/i2s/peer-status").json()
    assert data["generation"] == 5
    assert data["sample_rate"] == 44100


def test_i2s_peer_status_accepts_same_generation_newer_timestamp(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "MAGICBOX_I2S_PEER_STATUS_PATH", str(tmp_path / "peer-status.json")
    )
    app = create_app()
    client = TestClient(app)

    client.post(
        "/i2s/peer-status",
        json={
            "running": True,
            "mode": "capture",
            "sample_rate": 44100,
            "format": "S24_3LE",
            "channels": 2,
            "generation": 2,
            "updated_at_unix_ms": 2000,
        },
    )

    client.post(
        "/i2s/peer-status",
        json={
            "running": True,
            "mode": "capture",
            "sample_rate": 48000,
            "format": "S32_LE",
            "channels": 2,
            "generation": 2,
            "updated_at_unix_ms": 3000,
        },
    )

    data = client.get("/i2s/peer-status").json()
    assert data["generation"] == 2
    assert data["sample_rate"] == 48000
