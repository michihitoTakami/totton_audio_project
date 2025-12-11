"""RTP入力(GStreamer)のプロセス管理."""

from __future__ import annotations

import asyncio
import os
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Sequence

from ..models import (
    RtpInputConfigUpdate,
    RtpInputSettings,
    RtpInputStatus,
)

DEFAULT_PORT = 46000
DEFAULT_LATENCY_MS = 100
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
DEFAULT_ENCODING = "L24"
DEFAULT_DEVICE = "hw:Loopback,0,0"
DEFAULT_QUALITY = 8
DEFAULT_RTCP_PORT = DEFAULT_PORT + 1
DEFAULT_RTCP_SEND_PORT = DEFAULT_PORT + 2
DEFAULT_SENDER_HOST = "raspberrypi.local"
DEFAULT_MONITOR_INTERVAL_SEC = 1.0
DEFAULT_RATE_PROBE_TIMEOUT_SEC = 1.0

# Issue #762 要件に合わせたサポートレート（44.1k/48k系を網羅）
SUPPORTED_SAMPLE_RATES = {
    44100,
    88200,
    176400,
    352800,
    705600,
    48000,
    96000,
    192000,
    384000,
    768000,
}

_ENCODING_TO_DEPAY_AND_FORMAT = {
    # RFC3551: L16/L24/L32 RTP はネットワークバイトオーダー（BE）
    "L16": ("rtpL16depay", "S16BE"),
    "L24": ("rtpL24depay", "S24BE"),
    "L32": ("rtpL32depay", "S32BE"),
}

ProcessRunner = Callable[[Sequence[str]], Awaitable[asyncio.subprocess.Process]]
_logger = logging.getLogger(__name__)


def _env_int(
    name: str, default: int, minimum: int | None = None, maximum: int | None = None
) -> int:
    """環境変数をintとして解釈し、範囲をクランプして返す."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_str(name: str, default: str, allowed: Iterable[str] | None = None) -> str:
    """環境変数を取得し、許可リストがあればバリデート."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if allowed is None:
        return value or default
    return value if value in allowed else default


def load_default_settings() -> RtpInputSettings:
    """環境変数を考慮したデフォルト設定を返す."""
    encoding = _env_str(
        "MAGICBOX_RTP_ENCODING", DEFAULT_ENCODING, _ENCODING_TO_DEPAY_AND_FORMAT.keys()
    )
    base_port = _env_int("MAGICBOX_RTP_PORT", DEFAULT_PORT, minimum=1024, maximum=65535)
    sample_rate = _env_int(
        "MAGICBOX_RTP_SAMPLE_RATE",
        DEFAULT_SAMPLE_RATE,
        minimum=8000,
        maximum=768000,
    )
    # サポート外レートは安全側でデフォルトへフォールバック
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        sample_rate = DEFAULT_SAMPLE_RATE
    return RtpInputSettings(
        port=base_port,
        sample_rate=sample_rate,
        channels=_env_int(
            "MAGICBOX_RTP_CHANNELS", DEFAULT_CHANNELS, minimum=1, maximum=8
        ),
        latency_ms=_env_int(
            "MAGICBOX_RTP_LATENCY_MS", DEFAULT_LATENCY_MS, minimum=10, maximum=5000
        ),
        encoding=encoding,  # type: ignore[arg-type]
        device=os.getenv("MAGICBOX_RTP_DEVICE", DEFAULT_DEVICE),
        resample_quality=_env_int(
            "MAGICBOX_RTP_QUALITY", DEFAULT_QUALITY, minimum=0, maximum=10
        ),
        rtcp_port=_env_int(
            "MAGICBOX_RTP_RTCP_PORT", DEFAULT_RTCP_PORT, minimum=1024, maximum=65535
        ),
        rtcp_send_port=_env_int(
            "MAGICBOX_RTP_RTCP_SEND_PORT",
            DEFAULT_RTCP_SEND_PORT,
            minimum=1024,
            maximum=65535,
        ),
        sender_host=os.getenv("MAGICBOX_RTP_SENDER_HOST", DEFAULT_SENDER_HOST),
    )


def build_gst_command(settings: RtpInputSettings) -> list[str]:
    """設定からgst-launch-1.0コマンドを構築 (RTCP同期付き)."""
    depay, raw_format = _ENCODING_TO_DEPAY_AND_FORMAT.get(
        settings.encoding, ("rtpL24depay", "S24BE")
    )
    # ALSAに渡す直前はLE/S32で揃え、フォーマット不一致による not-linked を防ぐ
    sink_format = "S32LE"

    caps = (
        f"application/x-rtp,media=audio,clock-rate={settings.sample_rate},"
        f"encoding-name={settings.encoding},payload=96,channels={settings.channels}"
    )

    # RTP/RTCP (rtpbin) を用い、送信側のタイムスタンプに同期。RTCPは port+1(受信) / port+2(送信) を使用。
    # rtpbin.* のsinkとsrcは別チェーンに明示的に '!' で接続する。
    return [
        "gst-launch-1.0",
        "-e",
        "rtpbin",
        "name=rtpbin",
        f"latency={settings.latency_ms * 2}",  # add extra headroom to avoid underruns
        "ntp-sync=true",
        # RTP (payload)
        "udpsrc",
        f"port={settings.port}",
        f"caps={caps}",
        "!",
        "rtpbin.recv_rtp_sink_0",
        "rtpbin.",
        "!",
        depay,
        "!",
        "audioconvert",
        "!",
        "audioresample",
        f"quality={settings.resample_quality}",
        "!",
        # フォーマットは変換可能な範囲でシンクに合わせる
        f"audio/x-raw,format={sink_format},rate={settings.sample_rate},channels={settings.channels}",
        "!",
        "queue",
        "max-size-time=300000000",  # 300ms safety buffer (latency trade-off)
        "!",
        "alsasink",
        f"device={settings.device}",
        "sync=true",
        # RTCP (recv)
        "udpsrc",
        f"port={settings.rtcp_port}",
        "!",
        "rtpbin.recv_rtcp_sink_0",
        # RTCP (send back to sender host)
        "rtpbin.send_rtcp_src_0",
        "!",
        "udpsink",
        f"host={settings.sender_host}",
        f"port={settings.rtcp_send_port}",
        "sync=false",
        "async=false",
    ]


class RtpReceiverManager:
    """RTP受信パイプラインのライフサイクル管理."""

    def __init__(
        self,
        settings: RtpInputSettings | None = None,
        process_runner: ProcessRunner | None = None,
        restart_delay_sec: float = 5.0,
        restart_max_delay_sec: float = 30.0,
        auto_restart: bool = True,
        log_min_interval_sec: float = 30.0,
    ):
        self._settings = settings or load_default_settings()
        self._process: asyncio.subprocess.Process | None = None
        self._last_error: str | None = None
        self._lock = asyncio.Lock()
        self._process_runner: ProcessRunner = (
            process_runner or self._default_process_runner
        )
        self._monitor_task: asyncio.Task | None = None
        self._monitor_stop = asyncio.Event()
        self._watchdog_task: asyncio.Task | None = None
        self._watchdog_stop = asyncio.Event()
        self._restart_delay = max(0.1, restart_delay_sec)
        self._restart_max_delay = max(self._restart_delay, restart_max_delay_sec)
        self._auto_restart = auto_restart
        self._log_interval = max(1.0, log_min_interval_sec)
        self._log_next_ts = 0.0
        self._log_last_msg: str | None = None

    @staticmethod
    async def _default_process_runner(cmd: Sequence[str]) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(*cmd)

    def _is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def _validate_rate(self, sample_rate: int) -> int:
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")
        return sample_rate

    def _update_settings(self, update: RtpInputConfigUpdate) -> RtpInputSettings:
        """既存設定に更新をマージし、新しい設定を返す."""
        data = self._settings.model_dump()
        update_data = update.model_dump(exclude_none=True)
        data.update(update_data)
        # レートは必ずサポートリストにクランプ
        if "sample_rate" in data:
            data["sample_rate"] = self._validate_rate(int(data["sample_rate"]))
        # バリデーションはPydanticに任せる
        self._settings = RtpInputSettings.model_validate(data)
        return self._settings

    async def start(self) -> None:
        """パイプラインを起動（既に動作中なら何もしない）."""
        async with self._lock:
            self._watchdog_stop.clear()
            self._ensure_watchdog()
            if self._is_running():
                return
            try:
                await self._start_process_unlocked()
            except Exception as exc:  # noqa: BLE001
                # 自動再起動ループがバックグラウンドで再試行する
                self._last_error = str(exc)
                raise

    async def stop(self) -> None:
        """パイプラインを停止."""
        async with self._lock:
            self._watchdog_stop.set()
            await self._stop_unlocked(for_restart=False)
        await self._stop_watchdog()
        async with self._lock:
            # 次回のstartで再利用できるようフラグのみクリア
            self._watchdog_stop.clear()

    async def _start_unlocked(self) -> None:
        if self._is_running():
            return
        await self._start_process_unlocked()

    def _ensure_watchdog(self) -> None:
        if self._watchdog_task and not self._watchdog_task.done():
            return
        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(), name="rtp_process_watchdog"
        )

    async def _start_process_unlocked(self) -> None:
        cmd = build_gst_command(self._settings)
        try:
            self._process = await self._process_runner(cmd)
            self._last_error = None
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            raise

    async def _stop_unlocked(self, for_restart: bool = False) -> None:
        if not self._process:
            return
        proc = self._process
        self._process = None
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except asyncio.CancelledError:
            return
        except asyncio.TimeoutError:
            proc.kill()

        if not for_restart:
            # 再起動目的ではない停止時のみウォッチドッグ停止
            self._watchdog_stop.set()

    async def _stop_watchdog(self) -> None:
        if not self._watchdog_task:
            return
        task = self._watchdog_task
        self._watchdog_task = None
        try:
            await task
        except asyncio.CancelledError:
            return

    async def _watchdog_loop(self) -> None:
        """プロセス終了や起動失敗からの自動復旧ループ."""
        backoff = self._restart_delay
        while not self._watchdog_stop.is_set():
            async with self._lock:
                proc = self._process

            # プロセスが無ければ自動起動を試みる
            if proc is None:
                if not self._auto_restart:
                    await asyncio.sleep(0.2)
                    continue
                try:
                    async with self._lock:
                        await self._start_process_unlocked()
                    backoff = self._restart_delay
                    continue
                except Exception as exc:  # noqa: BLE001
                    self._last_error = f"autostart failed: {exc}"
                    self._warn_throttled(self._last_error)
                    await asyncio.sleep(backoff)
                    backoff = min(self._restart_max_delay, backoff * 2)
                    continue

            # プロセス終了を待機
            try:
                rc = await proc.wait()
            except Exception as exc:  # noqa: BLE001
                rc = None
                self._last_error = f"rtp process wait failed: {exc}"

            if self._watchdog_stop.is_set():
                break

            async with self._lock:
                if self._process is proc:
                    self._process = None

            if self._watchdog_stop.is_set() or not self._auto_restart:
                break

            # 予期せぬ終了は自動再起動する
            self._last_error = (
                f"rtp process exited with code {rc}"
                if rc is not None
                else "rtp process exited"
            )
            self._warn_throttled(self._last_error)
            await asyncio.sleep(backoff)
            backoff = min(self._restart_max_delay, backoff * 2)

    async def _restart_locked(self) -> None:
        await self._stop_unlocked(for_restart=True)
        await self._start_unlocked()

    def _warn_throttled(self, message: str) -> None:
        """連続失敗時にログスパムしないよう一定間隔でのみ出力."""
        now = time.monotonic()
        if message != self._log_last_msg or now >= self._log_next_ts:
            _logger.warning(message)
            self._log_last_msg = message
            self._log_next_ts = now + self._log_interval

    async def apply_config_and_restart(
        self, update: RtpInputConfigUpdate
    ) -> RtpInputSettings:
        """設定更新を反映し、稼働中なら最小ダウンタイムで再起動."""
        async with self._lock:
            new_settings = self._update_settings(update)
            try:
                await self._restart_locked()
            except Exception as exc:  # noqa: BLE001
                # API 経由で利用されるのでエラーは通知するが、last_error も残す
                self._last_error = f"restart failed: {exc}"
                raise
            return new_settings

    async def apply_config(self, update: RtpInputConfigUpdate) -> RtpInputSettings:
        """設定を更新し、適用後の設定を返す."""
        async with self._lock:
            new_settings = self._update_settings(update)
            return new_settings

    async def status(self) -> RtpInputStatus:
        """現在の状態を返す."""
        pid = self._process.pid if self._process and self._is_running() else None
        return RtpInputStatus(
            running=self._is_running(),
            pid=pid,
            last_error=self._last_error,
            settings=self._settings,
        )

    async def start_rate_monitor(
        self,
        rate_probe: Callable[[], Awaitable[int]],
        interval_sec: float = DEFAULT_MONITOR_INTERVAL_SEC,
        timeout_sec: float = DEFAULT_RATE_PROBE_TIMEOUT_SEC,
    ) -> None:
        """サンプルレート変化を監視し、変化時に自動再構築する."""
        if self._monitor_task and not self._monitor_task.done():
            # パラメータ変更時は一度止めて作り直す
            await self.stop_rate_monitor()
        self._monitor_stop.clear()
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(rate_probe, interval_sec, timeout_sec),
            name="rtp_rate_monitor",
        )

    async def stop_rate_monitor(self) -> None:
        """レート監視タスクを停止."""
        if not self._monitor_task:
            return
        self._monitor_stop.set()
        await asyncio.wait({self._monitor_task})
        if exc := self._monitor_task.exception():
            self._last_error = f"monitor stopped: {exc}"
        self._monitor_task = None

    async def _monitor_loop(
        self,
        rate_probe: Callable[[], Awaitable[int]],
        interval_sec: float,
        timeout_sec: float,
    ) -> None:
        while not self._monitor_stop.is_set():
            try:
                new_rate = await asyncio.wait_for(rate_probe(), timeout=timeout_sec)
            except Exception as exc:  # noqa: BLE001
                # 監視エラーはログに残しつつ継続
                self._last_error = f"rate_probe error: {exc}"
                await asyncio.sleep(interval_sec)
                continue

            try:
                validated_rate = self._validate_rate(new_rate)
            except ValueError as exc:
                self._last_error = str(exc)
                await asyncio.sleep(interval_sec)
                continue

            if validated_rate != self._settings.sample_rate:
                async with self._lock:
                    self._settings = self._settings.model_copy(
                        update={"sample_rate": validated_rate}
                    )
                    if self._is_running():
                        try:
                            await self._restart_locked()
                        except Exception as exc:  # noqa: BLE001
                            # 失敗しても監視は継続し、再試行に任せる
                            self._last_error = f"restart failed: {exc}"
                    else:
                        # 停止中は設定のみ更新し、意図しない自動起動を避ける
                        self._last_error = None
            await asyncio.sleep(interval_sec)

    async def shutdown(self) -> None:
        """監視とパイプラインをまとめて停止."""
        await self.stop_rate_monitor()
        await self.stop()


@dataclass
class RtpDriftStats:
    """RTPジッタ／ドリフト統計."""

    drift_ppm: float
    average_jitter_ms: float
    sample_count: int


class RtpDriftEstimator:
    """到着時刻とRTPタイムスタンプからドリフトを推定."""

    def __init__(self, sample_rate: int, window: int = 128):
        self.sample_rate = sample_rate
        self.window = max(2, window)
        self._observations: deque[tuple[int, float, float]] = deque(
            maxlen=self.window
        )  # (rtp_ts, arrival_s, deviation_s)

    def observe(self, rtp_timestamp: int, arrival_time_ns: int) -> RtpDriftStats:
        arrival_s = arrival_time_ns / 1_000_000_000
        deviation_s = 0.0
        if self._observations:
            last_rtp_ts, last_arrival_s, _ = self._observations[-1]
            expected_interval = (rtp_timestamp - last_rtp_ts) / self.sample_rate
            actual_interval = arrival_s - last_arrival_s
            if expected_interval > 0:
                deviation_s = actual_interval - expected_interval
            else:
                deviation_s = 0.0
        self._observations.append((rtp_timestamp, arrival_s, deviation_s))
        return self.stats()

    def stats(self) -> RtpDriftStats:
        if len(self._observations) < 2:
            return RtpDriftStats(drift_ppm=0.0, average_jitter_ms=0.0, sample_count=0)

        # ドリフトは最新観測値の偏差を用いる
        _, _, latest_dev = self._observations[-1]
        last, prev = self._observations[-1], self._observations[-2]
        expected_interval = (last[0] - prev[0]) / self.sample_rate
        drift_ppm = 0.0
        if expected_interval > 0:
            drift_ppm = (latest_dev / expected_interval) * 1_000_000

        # ジッタは絶対偏差の移動平均
        deviations = [abs(dev) for _, _, dev in self._observations if dev != 0.0]
        avg_jitter_ms = (
            (sum(deviations) / len(deviations)) * 1000 if deviations else 0.0
        )
        return RtpDriftStats(
            drift_ppm=drift_ppm,
            average_jitter_ms=avg_jitter_ms,
            sample_count=len(self._observations),
        )


# シングルトン的に利用するマネージャ
rtp_receiver_manager = RtpReceiverManager()


def get_rtp_receiver_manager() -> RtpReceiverManager:
    """FastAPI依存解決用の取得関数."""
    return rtp_receiver_manager


__all__ = [
    "DEFAULT_LATENCY_MS",
    "DEFAULT_PORT",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
    "DEFAULT_ENCODING",
    "DEFAULT_DEVICE",
    "DEFAULT_QUALITY",
    "DEFAULT_MONITOR_INTERVAL_SEC",
    "SUPPORTED_SAMPLE_RATES",
    "build_gst_command",
    "load_default_settings",
    "RtpReceiverManager",
    "RtpDriftEstimator",
    "RtpDriftStats",
    "rtp_receiver_manager",
    "get_rtp_receiver_manager",
]
