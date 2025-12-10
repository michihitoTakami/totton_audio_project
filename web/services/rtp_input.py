"""RTP入力(GStreamer)のプロセス管理."""

from __future__ import annotations

import asyncio
import os
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

_ENCODING_TO_DEPAY_AND_FORMAT = {
    "L16": ("rtpL16depay", "S16LE"),
    "L24": ("rtpL24depay", "S24LE"),
    "L32": ("rtpL32depay", "S32LE"),
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
    return RtpInputSettings(
        port=_env_int("MAGICBOX_RTP_PORT", DEFAULT_PORT, minimum=1024, maximum=65535),
        sample_rate=_env_int(
            "MAGICBOX_RTP_SAMPLE_RATE",
            DEFAULT_SAMPLE_RATE,
            minimum=8000,
            maximum=768000,
        ),
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
    )


def build_gst_command(settings: RtpInputSettings) -> list[str]:
    """設定からgst-launch-1.0コマンドを構築."""
    depay, raw_format = _ENCODING_TO_DEPAY_AND_FORMAT.get(
        settings.encoding, ("rtpL24depay", "S24LE")
    )
    caps = (
        f"application/x-rtp,media=audio,clock-rate={settings.sample_rate},"
        f"encoding-name={settings.encoding},channels={settings.channels}"
    )

    # jitterbufferのlatency(ms)は指定値を利用。audioresample qualityはデフォルト10。
    return [
        "gst-launch-1.0",
        "-e",
        "udpsrc",
        f"port={settings.port}",
        f"caps={caps}",
        "!",
        "rtpjitterbuffer",
        f"latency={settings.latency_ms}",
        "!",
        depay,
        "!",
        "audioconvert",
        "!",
        "audioresample",
        f"quality={settings.resample_quality}",
        "!",
        "audio/x-raw",
        f"format={raw_format}",
        f"rate={settings.sample_rate}",
        f"channels={settings.channels}",
        "!",
        "queue",
        "max-size-time=200000000",  # 200ms safety buffer
        "!",
        "alsasink",
        f"device={settings.device}",
        "sync=true",
        "provide-clock=true",
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

    @staticmethod
    async def _default_process_runner(cmd: Sequence[str]) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(*cmd)

    def _is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def _update_settings(self, update: RtpInputConfigUpdate) -> RtpInputSettings:
        """既存設定に更新をマージし、新しい設定を返す."""
        data = self._settings.model_dump()
        update_data = update.model_dump(exclude_none=True)
        data.update(update_data)
        # バリデーションはPydanticに任せる
        self._settings = RtpInputSettings.model_validate(data)
        return self._settings

    async def start(self) -> None:
        """パイプラインを起動（既に動作中なら何もしない）."""
        async with self._lock:
            if self._is_running():
                return
            cmd = build_gst_command(self._settings)
            try:
                self._process = await self._process_runner(cmd)
                self._last_error = None
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                raise

    async def stop(self) -> None:
        """パイプラインを停止."""
        async with self._lock:
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
    "build_gst_command",
    "load_default_settings",
    "RtpReceiverManager",
    "rtp_receiver_manager",
    "get_rtp_receiver_manager",
]
