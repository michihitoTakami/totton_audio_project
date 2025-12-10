"""RTP入力(GStreamer)のプロセス管理."""

from __future__ import annotations

import asyncio
import os
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
DEFAULT_QUALITY = 10
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
    "L16": ("rtpL16depay", "S16BE"),
    "L24": ("rtpL24depay", "S24BE"),
    "L32": ("rtpL32depay", "S32BE"),
}

ProcessRunner = Callable[[Sequence[str]], Awaitable[asyncio.subprocess.Process]]


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
    """設定からgst-launch-1.0コマンドを構築 (シンプルなRTP受信)."""
    depay, raw_format = _ENCODING_TO_DEPAY_AND_FORMAT.get(
        settings.encoding, ("rtpL24depay", "S24BE")
    )
    caps = (
        f"application/x-rtp,media=audio,clock-rate={settings.sample_rate},"
        f"encoding-name={settings.encoding},payload=96,channels={settings.channels}"
    )

    # シンプルなRTP受信パイプライン（rtpbin不使用）
    return [
        "gst-launch-1.0",
        "-e",
        "udpsrc",
        f"port={settings.port}",
        f"caps={caps}",
        "!",
        depay,
        "!",
        "audioconvert",
        "!",
        "audioresample",
        f"quality={settings.resample_quality}",
        "!",
        f"audio/x-raw,format={raw_format},rate={settings.sample_rate},channels={settings.channels}",
        "!",
        "queue",
        "max-size-time=200000000",  # 200ms safety buffer
        "!",
        "alsasink",
        f"device={settings.device}",
        "sync=true",
    ]


class RtpReceiverManager:
    """RTP受信パイプラインのライフサイクル管理."""

    def __init__(
        self,
        settings: RtpInputSettings | None = None,
        process_runner: ProcessRunner | None = None,
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
            if self._is_running():
                return
            await self._start_unlocked()

    async def stop(self) -> None:
        """パイプラインを停止."""
        async with self._lock:
            await self._stop_unlocked()

    async def _start_unlocked(self) -> None:
        if self._is_running():
            return
        cmd = build_gst_command(self._settings)
        try:
            self._process = await self._process_runner(cmd)
            self._last_error = None
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            raise

    async def _stop_unlocked(self) -> None:
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
        except asyncio.TimeoutError:
            proc.kill()

    async def _restart_locked(self) -> None:
        await self._stop_unlocked()
        await self._start_unlocked()

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
