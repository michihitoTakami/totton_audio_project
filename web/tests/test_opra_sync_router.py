from pathlib import Path

from fastapi.testclient import TestClient

from scripts.integration.opra_cache import OpraCacheManager, compute_sha256
from scripts.integration.opra_downloader import OpraDownloadResult
from web.main import create_app


def _write_sample_db(path: Path) -> None:
    path.write_text('{"type":"vendor","id":"v1","data":{"name":"Test"}}\n')


def test_opra_sync_status_returns_metadata(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GPU_OS_DATA_DIR", str(tmp_path))

    manager = OpraCacheManager()
    db_path = tmp_path / "database_v1.jsonl"
    _write_sample_db(db_path)

    metadata = manager.build_metadata(
        commit_sha="abc1234",
        source="manual",
        source_url="file://local",
        database_path=db_path,
        stats={"vendor": 1},
    )
    manager.install_version("abc1234", db_path, metadata)
    manager.activate_version("abc1234")

    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/opra/sync/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_commit"] == "abc1234"
    assert body["current_metadata"]["source"] == "manual"


def test_opra_sync_update_starts_job(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GPU_OS_DATA_DIR", str(tmp_path))

    def fake_download_opra_database(*, target, source, temp_dir, **_kwargs):
        db_path = temp_dir / "database_v1.jsonl"
        _write_sample_db(db_path)
        return OpraDownloadResult(
            commit_sha="abc1234",
            source=source,
            source_url="https://example.com/opra.jsonl",
            database_path=db_path,
            sha256=compute_sha256(db_path),
            size_bytes=db_path.stat().st_size,
            stats={"vendor": 1},
        )

    monkeypatch.setattr(
        "web.services.opra_sync.download_opra_database", fake_download_opra_database
    )

    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/api/opra/sync/update",
        json={"target": "latest", "source": "github_raw"},
    )
    assert resp.status_code == 202

    manager = OpraCacheManager()
    state = manager.load_state()
    assert state.current_commit == "abc1234"
