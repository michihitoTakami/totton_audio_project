import asyncio

import pytest

from web.models import RtpInputConfigUpdate, RtpInputSettings
from web.services.rtp_input import (
    RtpDriftEstimator,
    RtpReceiverManager,
    build_gst_command,
)


def test_build_gst_command_supports_encodings():
    base = RtpInputSettings()
    l16 = build_gst_command(base.model_copy(update={"encoding": "L16"}))
    l24 = build_gst_command(base.model_copy(update={"encoding": "L24"}))
    l32 = build_gst_command(base.model_copy(update={"encoding": "L32"}))

    assert "rtpL16depay" in l16
    assert "rtpL24depay" in l24
    assert "rtpL32depay" in l32
    assert any("latency=100" in part for part in l24)
    assert "rtpbin" in l24
    assert any("rtcp" in part for part in l24)


def test_config_update_merges_and_validates():
    manager = RtpReceiverManager(settings=RtpInputSettings())
    updated = asyncio.run(
        manager.apply_config(
            RtpInputConfigUpdate(
                port=46000,
                latency_ms=250,
                rtcp_port=46010,
                rtcp_send_port=46011,
                sender_host="192.168.0.10",
            )
        )
    )
    assert updated.port == 46000
    assert updated.latency_ms == 250
    assert updated.rtcp_port == 46010
    assert updated.rtcp_send_port == 46011
    assert updated.sender_host == "192.168.0.10"

    with pytest.raises(Exception):
        asyncio.run(manager.apply_config(RtpInputConfigUpdate(latency_ms=1)))


class _DummyProcess:
    def __init__(self):
        self.returncode = None
        self.pid = 1234
        self._terminated = False

    def terminate(self):
        self._terminated = True

    async def wait(self):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


async def _dummy_runner(_cmd):
    return _DummyProcess()


def test_start_stop_uses_runner():
    manager = RtpReceiverManager(
        settings=RtpInputSettings(), process_runner=_dummy_runner
    )
    asyncio.run(manager.start())
    status = asyncio.run(manager.status())
    assert status.running is True
    assert status.pid == 1234

    asyncio.run(manager.stop())
    status_after = asyncio.run(manager.status())
    assert status_after.running is False


def test_rate_monitor_restarts_on_change():
    calls: list[str] = []

    async def _runner(_cmd):
        calls.append("start")
        return _DummyProcess()

    rates = [44100, 44100, 48000, 48000]
    idx = 0

    async def _probe():
        nonlocal idx
        rate = rates[min(idx, len(rates) - 1)]
        idx += 1
        return rate

    manager = RtpReceiverManager(
        settings=RtpInputSettings(sample_rate=44100), process_runner=_runner
    )

    async def _run():
        await manager.start()
        await manager.start_rate_monitor(_probe, interval_sec=0.01)
        await asyncio.sleep(0.05)
        await manager.stop_rate_monitor()
        return await manager.status()

    status = asyncio.run(_run())
    assert status.settings.sample_rate == 48000
    # 最初の起動とレート変更後の再起動で2回
    assert calls.count("start") >= 2


def test_rate_monitor_continues_when_restart_fails():
    calls: list[str] = []

    async def _runner(_cmd):
        calls.append("start")
        if len(calls) == 1:
            return _DummyProcess()
        raise RuntimeError("boom")

    rates = [44100, 48000, 48000]
    idx = 0

    async def _probe():
        nonlocal idx
        rate = rates[min(idx, len(rates) - 1)]
        idx += 1
        return rate

    manager = RtpReceiverManager(
        settings=RtpInputSettings(sample_rate=44100), process_runner=_runner
    )

    async def _run():
        await manager.start()
        await manager.start_rate_monitor(_probe, interval_sec=0.01)
        await asyncio.sleep(0.05)
        await manager.stop_rate_monitor()
        return await manager.status()

    status = asyncio.run(_run())
    # 再起動失敗後も監視は止まらず、last_error が記録される
    assert status.running is False
    assert status.settings.sample_rate == 48000
    assert calls.count("start") == 2
    assert status.last_error and "restart failed" in status.last_error


def test_rate_monitor_does_not_autostart_when_stopped():
    async def _runner(_cmd):
        return _DummyProcess()

    rates = [44100, 96000, 96000]
    idx = 0

    async def _probe():
        nonlocal idx
        rate = rates[min(idx, len(rates) - 1)]
        idx += 1
        return rate

    manager = RtpReceiverManager(
        settings=RtpInputSettings(sample_rate=44100), process_runner=_runner
    )

    async def _run():
        await manager.start_rate_monitor(_probe, interval_sec=0.01)
        await asyncio.sleep(0.05)
        await manager.stop_rate_monitor()
        return await manager.status()

    status = asyncio.run(_run())
    # 停止中は自動起動せず、設定のみ更新される
    assert status.running is False
    assert status.settings.sample_rate == 96000


def test_drift_estimator_tracks_ppm():
    estimator = RtpDriftEstimator(sample_rate=48000, window=32)
    base_ts = 0
    base_arrival = 0.0
    estimator.observe(base_ts, 0)

    # 10ms 区間で +50ppm の遅れを持つ到着を5回記録
    for _ in range(5):
        base_ts += 480  # 10ms worth of samples
        base_arrival += 0.0100005  # 50ppm drift
        stats = estimator.observe(base_ts, int(base_arrival * 1e9))

    assert 40 <= stats.drift_ppm <= 60
    assert stats.average_jitter_ms < 0.2
