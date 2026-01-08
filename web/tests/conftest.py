import pytest


@pytest.fixture(autouse=True)
def disable_rtp_autostart(monkeypatch):
    """テスト実行時はRTP自動起動を無効化して外部依存を避ける。"""
    monkeypatch.setenv("TOTTON_AUDIO_RTP_AUTOSTART", "false")
    yield
