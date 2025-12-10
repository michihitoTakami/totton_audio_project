"""ZeroMQ bridge for RTP receiver control and monitoring.

Web UI から RTP 受信パイプラインを操作するための最小ブリッジ。
- REP/REQ で待ち受け
- 5 秒タイムアウト
- STATUS / SET_LATENCY コマンドをサポート
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, asdict
from typing import Any, Callable

import zmq

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "ipc:///tmp/rtp_receiver.sock"
DEFAULT_TIMEOUT_MS = 5000
_MIN_LATENCY_MS = 10
_MAX_LATENCY_MS = 500


@dataclass
class RtpStats:
    """RTP 統計値."""

    packets_received: int = 0
    packets_lost: int = 0
    jitter_ms: float = 0.0
    clock_drift_ppm: float = 0.0
    sample_rate: int = 0


class RtpStatsStore:
    """スレッドセーフに RTP 統計を保持・更新する."""

    def __init__(self) -> None:
        self._stats = RtpStats()
        self._lock = threading.Lock()

    def update(
        self,
        *,
        packets_received: int | None = None,
        packets_lost: int | None = None,
        jitter_ms: float | None = None,
        clock_drift_ppm: float | None = None,
        sample_rate: int | None = None,
    ) -> None:
        """統計値を更新（指定された項目のみ上書き）。"""
        with self._lock:
            if packets_received is not None:
                self._stats.packets_received = max(0, int(packets_received))
            if packets_lost is not None:
                self._stats.packets_lost = max(0, int(packets_lost))
            if jitter_ms is not None:
                self._stats.jitter_ms = float(jitter_ms)
            if clock_drift_ppm is not None:
                self._stats.clock_drift_ppm = float(clock_drift_ppm)
            if sample_rate is not None:
                self._stats.sample_rate = max(0, int(sample_rate))

    def snapshot(self) -> dict[str, Any]:
        """統計値のスナップショットを辞書として返す."""
        with self._lock:
            return asdict(self._stats)


class RtpZmqBridge:
    """RTP 受信向け ZeroMQ ブリッジ (REP/REQ)."""

    def __init__(
        self,
        stats_store: RtpStatsStore,
        set_latency_ms: Callable[[int], None],
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        self._stats_store = stats_store
        self._set_latency_ms = set_latency_ms
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms

        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_lock = threading.Lock()

    def __enter__(self) -> "RtpZmqBridge":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.stop()

    def start(self) -> None:
        """ZeroMQ REP ソケットで待受を開始."""
        with self._start_lock:
            if self._thread and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REP)
            self._socket.linger = 0
            self._socket.rcvtimeo = self._timeout_ms
            self._socket.sndtimeo = self._timeout_ms
            self._socket.bind(self._endpoint)

            self._thread = threading.Thread(
                target=self._serve,
                name="rtp_zmq_bridge",
                daemon=True,
            )
            self._thread.start()
            logger.info("RTP ZeroMQ bridge started at %s", self._endpoint)

    def stop(self) -> None:
        """待受を停止し、ソケットとスレッドをクリーンアップ."""
        with self._start_lock:
            self._stop_event.set()
            socket, context, thread = self._socket, self._context, self._thread

        if thread is not None:
            thread.join(timeout=max(1.0, self._timeout_ms / 1000 + 0.5))
            if thread.is_alive():
                logger.warning("ZeroMQ bridge thread did not stop within timeout")

        with self._start_lock:
            self._socket = None
            self._context = None
            self._thread = None

        if socket is not None:
            try:
                socket.close(0)
            except zmq.ZMQError:
                logger.warning("Failed to close ZMQ socket cleanly", exc_info=True)

        if context is not None:
            try:
                context.term()
            except zmq.ZMQError:
                logger.warning("Failed to terminate ZMQ context cleanly", exc_info=True)

    def _serve(self) -> None:
        socket = self._socket
        if socket is None:
            return

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        # ポーリング間隔は停止を検知しやすいよう短めにする
        poll_timeout = min(self._timeout_ms, 500)

        while not self._stop_event.is_set():
            try:
                events = dict(poller.poll(poll_timeout))
            except zmq.ZMQError:
                break

            if socket not in events:
                continue

            if events[socket] & zmq.POLLIN:
                try:
                    request = socket.recv_json(flags=0)
                except zmq.Again:
                    continue
                except (ValueError, json.JSONDecodeError):
                    self._safe_send(
                        {"status": "error", "message": "invalid json payload"}
                    )
                    continue
                except zmq.ZMQError as exc:
                    logger.warning("ZeroMQ receive failed: %s", exc)
                    continue

                response = self._handle_request(request)
                self._safe_send(response)

        poller.unregister(socket)

    def _safe_send(self, payload: dict[str, Any]) -> None:
        if self._socket is None:
            return
        try:
            self._socket.send_json(payload)
        except zmq.ZMQError as exc:
            logger.warning("ZeroMQ send failed: %s", exc)

    def _handle_request(self, message: Any) -> dict[str, Any]:
        if not isinstance(message, dict):
            return {"status": "error", "message": "request must be a JSON object"}

        cmd = str(message.get("cmd", "")).upper()
        if cmd == "STATUS":
            return {"status": "ok", "data": self._stats_store.snapshot()}
        if cmd == "SET_LATENCY":
            return self._handle_set_latency(message)

        return {"status": "error", "message": f"unknown command: {cmd or '<empty>'}"}

    def _handle_set_latency(self, message: dict[str, Any]) -> dict[str, Any]:
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return {"status": "error", "message": "params must be an object"}

        if "latency_ms" not in params:
            return {"status": "error", "message": "latency_ms is required"}

        try:
            latency = int(params["latency_ms"])
        except (TypeError, ValueError):
            return {"status": "error", "message": "latency_ms must be an integer"}

        if latency < _MIN_LATENCY_MS or latency > _MAX_LATENCY_MS:
            return {
                "status": "error",
                "message": f"latency_ms must be between {_MIN_LATENCY_MS} and {_MAX_LATENCY_MS}",
            }

        try:
            self._set_latency_ms(latency)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to apply latency change")
            return {"status": "error", "message": f"failed to set latency: {exc}"}

        return {"status": "ok", "data": {"latency_ms": latency}}
