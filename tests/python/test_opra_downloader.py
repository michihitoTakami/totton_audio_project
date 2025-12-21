from pathlib import Path

import pytest

from scripts.integration.opra_downloader import (
    OPRA_CLOUDFLARE_URL,
    OPRA_RAW_BASE,
    OpraDownloadError,
    download_opra_database,
    resolve_latest_commit_sha,
    validate_database,
)


class FetchRecorder:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, url: str, _timeout: float) -> bytes:
        self.calls.append(url)
        if url not in self.responses:
            raise AssertionError(f"Unexpected URL: {url}")
        return self.responses[url]


def _sample_jsonl() -> bytes:
    lines = [
        '{"type":"vendor","id":"v1","data":{"name":"Test"}}',
        '{"type":"product","id":"p1","data":{"name":"Headphone"}}',
        '{"type":"eq","id":"e1","data":{"name":"EQ"}}',
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_resolve_latest_commit_sha() -> None:
    api_url = "https://api.github.com/repos/opra-project/OPRA/commits/main"
    expected_sha = "abcdef123"  # pragma: allowlist secret
    fetcher = FetchRecorder({api_url: b'{"sha":"ABCDEF123"}'})
    assert resolve_latest_commit_sha(fetcher=fetcher) == expected_sha
    assert fetcher.calls == [api_url]


def test_download_latest_uses_raw_sha(tmp_path: Path) -> None:
    api_url = "https://api.github.com/repos/opra-project/OPRA/commits/main"
    resolved_sha = "abc1234"  # pragma: allowlist secret
    raw_url = f"{OPRA_RAW_BASE}/{resolved_sha}/dist/database_v1.jsonl"
    fetcher = FetchRecorder(
        {
            api_url: f'{{"sha":"{resolved_sha}"}}'.encode("utf-8"),
            raw_url: _sample_jsonl(),
        }
    )
    result = download_opra_database(
        target="latest",
        source="github_raw",
        fetcher=fetcher,
        temp_dir=tmp_path,
        sample_lines=3,
    )
    assert result.commit_sha == resolved_sha
    assert result.source_url == raw_url
    assert result.database_path.exists()
    assert result.sha256
    assert result.stats["vendor"] == 1


def test_download_commit_sha_direct(tmp_path: Path) -> None:
    commit_sha = "deadbeef"  # pragma: allowlist secret
    raw_url = f"{OPRA_RAW_BASE}/{commit_sha}/dist/database_v1.jsonl"
    fetcher = FetchRecorder({raw_url: _sample_jsonl()})
    result = download_opra_database(
        target=commit_sha,
        source="github_raw",
        fetcher=fetcher,
        temp_dir=tmp_path,
        sample_lines=2,
    )
    assert result.commit_sha == commit_sha
    assert fetcher.calls == [raw_url]


def test_download_cloudflare_marks_unknown(tmp_path: Path) -> None:
    fetcher = FetchRecorder({OPRA_CLOUDFLARE_URL: _sample_jsonl()})
    result = download_opra_database(
        target="latest",
        source="cloudflare",
        fetcher=fetcher,
        temp_dir=tmp_path,
        sample_lines=3,
    )
    assert result.commit_sha == "unknown"
    assert result.source_url == OPRA_CLOUDFLARE_URL


def test_validation_requires_type(tmp_path: Path) -> None:
    path = tmp_path / "database_v1.jsonl"
    path.write_text('{"id":"no_type"}\n', encoding="utf-8")
    with pytest.raises(OpraDownloadError):
        validate_database(path, sample_lines=1)


def test_download_rejects_broken_jsonl(tmp_path: Path) -> None:
    commit_sha = "badfeed"  # pragma: allowlist secret
    raw_url = f"{OPRA_RAW_BASE}/{commit_sha}/dist/database_v1.jsonl"
    fetcher = FetchRecorder({raw_url: b'{"type":"vendor"\n'})

    with pytest.raises(OpraDownloadError):
        download_opra_database(
            target=commit_sha,
            source="github_raw",
            fetcher=fetcher,
            temp_dir=tmp_path,
            sample_lines=5,
        )
