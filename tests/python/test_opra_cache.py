from pathlib import Path

from scripts.integration.opra_cache import (
    DATABASE_FILENAME,
    METADATA_FILENAME,
    OpraCacheManager,
)


def _write_db(path: Path) -> None:
    path.write_text(
        '{"type":"vendor","id":"test","data":{"name":"Test"}}\n', encoding="utf-8"
    )


def test_install_and_activate_opra_version(tmp_path: Path) -> None:
    opra_dir = tmp_path / "opra"
    manager = OpraCacheManager(opra_dir=opra_dir, max_versions=3)
    db_path = tmp_path / "database_v1.jsonl"
    _write_db(db_path)

    commit_sha = "abcdef1"
    metadata = manager.build_metadata(
        commit_sha=commit_sha,
        source="manual",
        source_url="file://local",
        database_path=db_path,
    )
    version_dir = manager.install_version(commit_sha, db_path, metadata)

    assert (version_dir / DATABASE_FILENAME).exists()
    assert (version_dir / METADATA_FILENAME).exists()

    manager.activate_version(commit_sha)
    assert manager.current_path.is_symlink()
    assert manager.current_path.resolve() == version_dir

    state = manager.load_state()
    assert state.current_commit == commit_sha


def test_rollback_to_previous_version(tmp_path: Path) -> None:
    opra_dir = tmp_path / "opra"
    manager = OpraCacheManager(opra_dir=opra_dir, max_versions=3)
    db_path = tmp_path / "database_v1.jsonl"
    _write_db(db_path)

    first_sha = "1111111"
    second_sha = "2222222"

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

    manager.rollback()
    state = manager.load_state()
    assert state.current_commit == first_sha
    assert state.previous_commit == second_sha


def test_prune_keeps_latest_versions(tmp_path: Path) -> None:
    opra_dir = tmp_path / "opra"
    manager = OpraCacheManager(opra_dir=opra_dir, max_versions=2)
    db_path = tmp_path / "database_v1.jsonl"
    _write_db(db_path)

    shas = ["aaaaaaa", "bbbbbbb", "ccccccc"]
    for commit_sha in shas:
        metadata = manager.build_metadata(
            commit_sha=commit_sha,
            source="manual",
            source_url="file://local",
            database_path=db_path,
        )
        manager.install_version(commit_sha, db_path, metadata)
        manager.activate_version(commit_sha)

    remaining = {path.name for path in (opra_dir / "versions").iterdir()}
    assert remaining == {shas[-1], shas[-2]}
