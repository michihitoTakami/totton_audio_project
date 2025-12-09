"""TCP入力テレメトリのパーサーとポーラー."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from typing import Any, Awaitable, Callable, cast

from ..models import (
    TcpConnectionMode,
    TcpInputStreamFormat,
    TcpInputTelemetry,
)

logger = logging.getLogger(__name__)

POLL_DEFAULT_SEC = 2.0
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FORMAT_CODE_TO_NAME = {
    1: "S16_LE",
    2: "S24_3LE",
    4: "S32_LE",
}


def _env_flag(name: str) -> bool:
    """Return True if an env var looks enabled."""
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def _resolve_poll_interval_seconds(override: float | None = None) -> float:
    """Resolve polling interval with env override and sane minimum."""
    if override is not None:
        return max(0.1, float(override))

    env_value = os.getenv("MAGICBOX_TCP_POLL_INTERVAL_SEC")
    if env_value:
        try:
            return max(0.1, float(env_value))
        except ValueError:
            logger.warning(
                "MAGICBOX_TCP_POLL_INTERVAL_SEC=%s が数値として解釈できません。デフォルト%ssを使用します。",
                env_value,
                POLL_DEFAULT_SEC,
            )
    return POLL_DEFAULT_SEC


def _normalize_format(value: Any) -> str:
    """Normalize PCM format representation to user-friendly string."""
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            try:
                return _FORMAT_CODE_TO_NAME[int(text)]
            except (KeyError, ValueError):
                return text
        upper = text.upper()
        return upper if upper else "unknown"
    if isinstance(value, int):
        return _FORMAT_CODE_TO_NAME.get(value, str(value))
    return "unknown"


def _normalize_connection_mode(value: Any) -> TcpConnectionMode:
    """Normalize connection mode to supported literals."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"single", "takeover", "priority"}:
            return cast(TcpConnectionMode, normalized)
    return cast(TcpConnectionMode, "single")


def _parse_header(raw: Any) -> TcpInputStreamFormat | None:
    """Parse PCM header payload into model."""
    if not isinstance(raw, dict):
        return None
    try:
        return TcpInputStreamFormat(
            sample_rate=int(raw.get("sample_rate", 0) or 0),
            channels=int(raw.get("channels", 0) or 0),
            format=_normalize_format(raw.get("format")),
            version=int(raw.get("version", 1) or 1),
        )
    except Exception:
        logger.debug("PCMヘッダーの解析に失敗しました。raw=%s", raw, exc_info=True)
        return None


def parse_tcp_telemetry(payload: dict[str, Any] | None) -> TcpInputTelemetry:
    """デーモンのステータスペイロードをPydanticモデルへ変換する."""
    if not isinstance(payload, dict):
        return TcpInputTelemetry()

    bound_port = payload.get("bound_port", payload.get("boundPort"))
    try:
        bound_port_int = int(bound_port) if bound_port is not None else None
    except (TypeError, ValueError):
        bound_port_int = None

    priority_clients_raw = payload.get("priority_clients")
    priority_clients: list[str] = []
    if isinstance(priority_clients_raw, list):
        priority_clients = [
            str(client)
            for client in priority_clients_raw
            if isinstance(client, (str, int))
        ]

    return TcpInputTelemetry(
        listening=bool(payload.get("listening", False)),
        bound_port=bound_port_int,
        client_connected=bool(payload.get("client_connected", False)),
        streaming=bool(payload.get("streaming", False)),
        xrun_count=int(payload.get("xrun_count", 0) or 0),
        ring_buffer_frames=int(payload.get("ring_buffer_frames", 0) or 0),
        watermark_frames=int(payload.get("watermark_frames", 0) or 0),
        buffered_frames=int(payload.get("buffered_frames", 0) or 0),
        max_buffered_frames=int(payload.get("max_buffered_frames", 0) or 0),
        dropped_frames=int(payload.get("dropped_frames", 0) or 0),
        disconnect_reason=payload.get("disconnect_reason"),
        connection_mode=_normalize_connection_mode(payload.get("connection_mode")),
        priority_clients=priority_clients,
        last_header=_parse_header(payload.get("last_header")),
        rep_endpoint=payload.get("rep_endpoint"),
        pub_endpoint=payload.get("pub_endpoint"),
    )


class TcpTelemetryStore:
    """スレッドセーフなTCPテレメトリキャッシュ."""

    def __init__(self):
        self._telemetry: TcpInputTelemetry | None = None
        self._updated_at: float | None = None
        self._last_error: str | None = None
        self._lock = asyncio.Lock()

    async def update(self, telemetry: TcpInputTelemetry) -> None:
        """最新テレメトリを保存する."""
        async with self._lock:
            self._telemetry = telemetry
            self._updated_at = time.time()
            self._last_error = None

    async def record_error(self, message: str) -> None:
        """エラー状態を保存する（最新テレメトリは保持）."""
        async with self._lock:
            self._last_error = message
            self._updated_at = time.time()

    async def snapshot(
        self,
    ) -> tuple[TcpInputTelemetry | None, float | None, str | None]:
        """テレメトリとメタ情報のスナップショットを返す."""
        async with self._lock:
            return self._telemetry, self._updated_at, self._last_error

    async def latest(self) -> TcpInputTelemetry:
        """テレメトリを取得（未取得時はデフォルト値）."""
        async with self._lock:
            return self._telemetry or TcpInputTelemetry()


Fetcher = Callable[
    [],
    Awaitable[TcpInputTelemetry | dict[str, Any] | None]
    | TcpInputTelemetry
    | dict[str, Any]
    | None,
]


class TcpTelemetryPoller:
    """非同期TCPテレメトリポーラー."""

    def __init__(
        self,
        fetcher: Fetcher,
        store: TcpTelemetryStore,
        interval_seconds: float | None = None,
    ):
        self._fetcher = fetcher
        self._store = store
        self._interval = _resolve_poll_interval_seconds(interval_seconds)
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._disabled = _env_flag("MAGICBOX_DISABLE_TCP_POLLING")

    @property
    def running(self) -> bool:
        """ポーラーが起動中かを返す."""
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """ポーリングを開始する（環境変数で無効化可能）。"""
        if self._disabled:
            logger.info("TCPテレメトリポーリングは環境変数で無効化されています。")
            return
        if self.running:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="tcp-telemetry-poller")

    async def stop(self) -> None:
        """ポーリングを停止する."""
        if not self._task:
            return
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _awaitable_fetch(self) -> TcpInputTelemetry | dict[str, Any] | None:
        """Fetch helper to support sync/async callables."""
        result = self._fetcher()
        if inspect.isawaitable(result):
            return await result
        return result

    async def _poll_once(self) -> None:
        """Fetch telemetry once and update store."""
        try:
            result = await self._awaitable_fetch()
            if result is None:
                return
            telemetry = (
                result
                if isinstance(result, TcpInputTelemetry)
                else TcpInputTelemetry.model_validate(result)
            )
            await self._store.update(telemetry)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("TCPテレメトリの取得に失敗しました: %s", exc, exc_info=True)
            await self._store.record_error(str(exc))

    async def _run(self) -> None:
        """Background loop."""
        try:
            while not self._stop_event.is_set():
                await self._poll_once()
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._interval
                    )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass


__all__ = [
    "TcpTelemetryStore",
    "TcpTelemetryPoller",
    "parse_tcp_telemetry",
]
