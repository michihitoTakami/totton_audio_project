from pathlib import Path


from scripts.integration.opra_cache import OpraCacheManager
from scripts.integration.opra_downloader import OpraDownloadError
from web.services import opra_sync


def _write_db(path: Path) -> None:
    path.write_text(
        '{"type":"vendor","id":"v1","data":{"name":"Test"}}\n',
        encoding="utf-8",
    )


def test_run_update_job_failure_keeps_current(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPU_OS_DATA_DIR", str(tmp_path))

    manager = OpraCacheManager()
    db_path = tmp_path / "database_v1.jsonl"
    _write_db(db_path)

    metadata = manager.build_metadata(
        commit_sha="1a2b3c4",
        source="manual",
        source_url="file://local",
        database_path=db_path,
    )
    manager.install_version("1a2b3c4", db_path, metadata)
    manager.activate_version("1a2b3c4")

    job_id = opra_sync.start_update_job(target="latest", source="github_raw")

    def _fail_download(**_kwargs):
        raise OpraDownloadError("network failure")

    monkeypatch.setattr("web.services.opra_sync.download_opra_database", _fail_download)

    opra_sync.run_update_job(job_id=job_id, target="latest", source="github_raw")

    state = manager.load_state()
    assert state.status == "failed"
    assert state.current_commit == "1a2b3c4"
    assert manager.get_current_commit() == "1a2b3c4"


def test_run_rollback_job_switches_to_previous(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPU_OS_DATA_DIR", str(tmp_path))

    manager = OpraCacheManager()
    db_path = tmp_path / "database_v1.jsonl"
    _write_db(db_path)

    first_sha = "aaa1111"
    second_sha = "bbb2222"

    metadata = manager.build_metadata(
        commit_sha=first_sha,
        source="manual",
        source_url="file://local",
        database_path=db_path,
    )
    manager.install_version(first_sha, db_path, metadata)
    manager.activate_version(first_sha)

    metadata = manager.build_metadata(
        commit_sha=second_sha,
        source="manual",
        source_url="file://local",
        database_path=db_path,
    )
    manager.install_version(second_sha, db_path, metadata)
    manager.activate_version(second_sha)

    job_id = opra_sync.start_rollback_job()
    opra_sync.run_rollback_job(job_id=job_id)

    state = manager.load_state()
    assert state.status == "ready"
    assert state.current_commit == first_sha
    assert state.previous_commit == second_sha
