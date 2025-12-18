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
    # depay直後のraw capsを明示して交渉揺れを抑制する
    assert any(part.startswith("audio/x-raw,format=S16BE,") for part in l16)
    assert any(part.startswith("audio/x-raw,format=S24BE,") for part in l24)
    assert any(part.startswith("audio/x-raw,format=S32BE,") for part in l32)
    assert any(f"latency={base.latency_ms}" in part for part in l24)
    assert any("quality=8" in part for part in l24)
    assert "max-size-time=60000000" in l24
    assert "rtpbin" in l24
    assert any("rtcp" in part for part in l24)
    l24_str = " ".join(l24)
    # sink -> src を '!' で連結している
    assert "rtpbin.recv_rtp_sink_0" in l24_str
    assert "rtpbin. !" in l24_str


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
    def __init__(self, wait_result: int = 0):
        self.returncode = None
        self.pid = 1234
        self._terminated = False
        self._wait_result = wait_result
        self._wait_future: asyncio.Future[int] | None = None

    def terminate(self):
        self._terminated = True
        if self._wait_future and not self._wait_future.done():
            self._wait_future.set_result(self._wait_result)

    async def wait(self):
        if self._wait_future is None:
            # Immediate exit if no external trigger
            self.returncode = self._wait_result
            return self._wait_result
        self.returncode = await self._wait_future
        return self.returncode

    def kill(self):
        self.returncode = -9
        if self._wait_future and not self._wait_future.done():
            self._wait_future.set_result(-9)

    def finish(self, code: int = 0):
        if self._wait_future and not self._wait_future.done():
            self._wait_future.set_result(code)

    def enable_deferred_wait(self):
        if self._wait_future is None:
            self._wait_future = asyncio.get_event_loop().create_future()


async def _dummy_runner(_cmd):
    proc = _DummyProcess()
    proc.enable_deferred_wait()
    return proc


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
        proc = _DummyProcess()
        proc.enable_deferred_wait()
        return proc

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
            proc = _DummyProcess()
            proc.enable_deferred_wait()
            return proc
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
    assert status.last_error and (
        "restart failed" in status.last_error
        or "rtp process exited" in status.last_error
    )


def test_rate_monitor_does_not_autostart_when_stopped():
    async def _runner(_cmd):
        proc = _DummyProcess()
        proc.enable_deferred_wait()
        return proc

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


def test_rate_probe_timeout_does_not_hang():
    async def _runner(_cmd):
        proc = _DummyProcess()
        proc.enable_deferred_wait()
        return proc

    async def _probe_hang():
        await asyncio.sleep(0.1)  # longer than timeout
        return 48000

    manager = RtpReceiverManager(
        settings=RtpInputSettings(sample_rate=44100), process_runner=_runner
    )

    async def _run():
        await manager.start()
        await manager.start_rate_monitor(
            _probe_hang, interval_sec=0.01, timeout_sec=0.02
        )
        await asyncio.sleep(0.05)
        await manager.stop_rate_monitor()
        return await manager.status()

    status = asyncio.run(_run())
    # タイムアウト後もフリーズせず last_error が設定され、動作は継続
    assert status.running is True
    assert status.last_error and "rate_probe error" in status.last_error


def test_apply_config_and_restart_records_error_on_failure():
    async def _runner(_cmd):
        raise RuntimeError("fail-start")

    manager = RtpReceiverManager(
        settings=RtpInputSettings(sample_rate=44100), process_runner=_runner
    )

    with pytest.raises(RuntimeError):
        asyncio.run(
            manager.apply_config_and_restart(
                RtpInputConfigUpdate(sample_rate=48000, latency_ms=200)
            )
        )
    status = asyncio.run(manager.status())
    assert status.last_error and "restart failed" in status.last_error


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


def test_auto_restart_when_process_exits():
    calls: list[_DummyProcess] = []

    async def _runner(_cmd):
        proc = _DummyProcess()
        proc.enable_deferred_wait()
        calls.append(proc)
        return proc

    manager = RtpReceiverManager(
        settings=RtpInputSettings(),
        process_runner=_runner,
        restart_delay_sec=0.01,
        restart_max_delay_sec=0.05,
    )

    async def _run():
        await manager.start()
        # 1回目のプロセスを終了させる
        calls[0].finish(1)
        await asyncio.sleep(0.2)
        return await manager.status()

    status = asyncio.run(_run())
    # 再起動が行われ、2回以上startされている
    assert len(calls) >= 2
    assert status.running is True
    assert status.pid == calls[-1].pid


def test_start_retries_after_initial_failure():
    attempts = 0

    async def _runner(_cmd):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("alsa-missing")
        proc = _DummyProcess()
        proc.enable_deferred_wait()
        return proc

    manager = RtpReceiverManager(
        settings=RtpInputSettings(),
        process_runner=_runner,
        restart_delay_sec=0.01,
        restart_max_delay_sec=0.05,
    )

    async def _run():
        with pytest.raises(RuntimeError):
            await manager.start()
        await asyncio.sleep(0.05)
        return await manager.status()

    status = asyncio.run(_run())
    assert attempts >= 2  # 自動リトライが発生
    assert status.running is True
