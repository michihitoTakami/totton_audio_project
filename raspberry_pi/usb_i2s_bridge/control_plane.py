"""ZeroMQ ベースの I2S 制御プレーン同期ユーティリティ."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import zmq

logger = logging.getLogger(__name__)

# 44.1k/48k 系 x 1/2/4/8/16
ALLOWED_SAMPLE_RATES = {
    44_100,
    88_200,
    176_400,
    352_800,
    705_600,
    48_000,
    96_000,
    192_000,
    384_000,
    768_000,
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _is_supported_rate(rate: int) -> bool:
    return int(rate) in ALLOWED_SAMPLE_RATES


@dataclass
class ControlStatus:
    """共有する状態."""

    running: bool
    mode: str  # capture / silence / none
    sample_rate: int
    fmt: str
    channels: int
    generation: int = 0
    updated_at_ms: int = field(default_factory=_now_ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "running": bool(self.running),
            "mode": str(self.mode),
            "sample_rate": int(self.sample_rate),
            "format": str(self.fmt),
            "channels": int(self.channels),
            "generation": int(self.generation),
            "updated_at_ms": int(self.updated_at_ms),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlStatus":
        return cls(
            running=bool(data.get("running", False)),
            mode=str(data.get("mode", "none")),
            sample_rate=int(data.get("sample_rate", 0)),
            fmt=str(data.get("format", "")),
            channels=int(data.get("channels", 0)),
            generation=int(data.get("generation", 0)),
            updated_at_ms=int(data.get("updated_at_ms", _now_ms())),
        )

    def is_supported(self) -> bool:
        return _is_supported_rate(self.sample_rate) and self.channels > 0


class ControlPlaneServer:
    """REQ/REP の REP 側."""

    def __init__(
        self,
        *,
        endpoint: str,
        status_provider: Callable[[], ControlStatus],
        on_peer_status: Optional[Callable[[ControlStatus], None]] = None,
        timeout_ms: int = 2000,
    ) -> None:
        self._endpoint = endpoint
        self._status_provider = status_provider
        self._on_peer_status = on_peer_status
        self._timeout_ms = max(500, int(timeout_ms))
        self._ctx: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._ctx = zmq.Context()
        sock = self._ctx.socket(zmq.REP)
        sock.linger = 0
        sock.rcvtimeo = self._timeout_ms
        sock.sndtimeo = self._timeout_ms
        sock.bind(self._endpoint)
        self._socket = sock
        self._thread = threading.Thread(
            target=self._serve, name="i2s_control_rep", daemon=True
        )
        self._thread.start()
        logger.info("I2S control-plane server started at %s", self._endpoint)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=(self._timeout_ms / 1000) + 1)
        if self._socket:
            try:
                self._socket.close(0)
            except zmq.ZMQError:
                pass
        if self._ctx:
            try:
                self._ctx.term()
            except zmq.ZMQError:
                pass
        self._socket = None
        self._ctx = None
        self._thread = None

    def _serve(self) -> None:
        assert self._socket is not None
        sock = self._socket
        while not self._stop.is_set():
            try:
                req = sock.recv_json(flags=0)
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            except ValueError:
                continue

            resp = self._handle(req if isinstance(req, dict) else {})
            try:
                sock.send_json(resp)
            except zmq.ZMQError:
                logger.warning("I2S control-plane: send failed")
                continue

    def _handle(self, req: Dict[str, Any]) -> Dict[str, Any]:
        cmd = str(req.get("cmd", "")).upper()
        if cmd not in {"STATUS", "SYNC"}:
            return {"status": "error", "message": f"unknown cmd: {cmd or '<empty>'}"}

        peer = req.get("peer")
        if peer and isinstance(peer, dict) and self._on_peer_status:
            try:
                self._on_peer_status(ControlStatus.from_dict(peer))
            except Exception as exc:  # noqa: BLE001
                logger.debug("failed to parse peer status: %s", exc)

        status = self._status_provider()
        return {"status": "ok", "data": status.to_dict()}


class ControlPlaneSync:
    """双方向同期と capture 許可判定を担う."""

    def __init__(
        self,
        *,
        endpoint: str,
        peer_endpoint: Optional[str],
        require_peer: bool = True,
        poll_interval_sec: float = 1.0,
        timeout_ms: int = 2000,
    ) -> None:
        self._local = ControlStatus(
            running=False, mode="none", sample_rate=0, fmt="", channels=0, generation=0
        )
        self._peer: ControlStatus | None = None
        self._peer_endpoint = peer_endpoint
        self._require_peer = require_peer
        self._poll_interval = max(0.2, float(poll_interval_sec))
        self._timeout_ms = max(500, int(timeout_ms))
        self._lock = threading.Lock()
        self._synced = False
        self._server = ControlPlaneServer(
            endpoint=endpoint,
            status_provider=self._get_local,
            on_peer_status=self._set_peer,
            timeout_ms=timeout_ms,
        )
        self._ctx: zmq.Context | None = None
        self._client: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # --- lifecycle ---
    def start(self) -> None:
        self._server.start()
        if self._peer_endpoint:
            self._stop.clear()
            self._ctx = zmq.Context()
            self._client = None
            self._thread = threading.Thread(
                target=self._poll_peer, name="i2s_control_req", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self._poll_interval + 0.5)
        self._reset_client()
        if self._ctx:
            try:
                self._ctx.term()
            except zmq.ZMQError:
                pass
        self._client = None
        self._ctx = None
        self._thread = None
        self._server.stop()

    # --- state ops ---
    def update_local(
        self,
        *,
        running: bool,
        mode: str,
        sample_rate: int,
        fmt: str,
        channels: int,
    ) -> None:
        """ローカル状態を更新し、生成番号を進める."""
        with self._lock:
            changed = (
                running != self._local.running
                or mode != self._local.mode
                or sample_rate != self._local.sample_rate
                or fmt != self._local.fmt
                or channels != self._local.channels
            )
            if changed:
                self._local.generation += 1
            self._local.running = running
            self._local.mode = mode
            self._local.sample_rate = int(sample_rate)
            self._local.fmt = fmt
            self._local.channels = int(channels)
            self._local.updated_at_ms = _now_ms()
            if changed:
                self._synced = False

    def _get_local(self) -> ControlStatus:
        with self._lock:
            return ControlStatus(**self._local.__dict__)

    def _set_peer(self, status: ControlStatus) -> None:
        with self._lock:
            self._peer = status
            self._refresh_synced_locked()

    def peer_status(self) -> ControlStatus | None:
        with self._lock:
            return self._peer

    def is_synced(self) -> bool:
        with self._lock:
            return self._synced

    def capture_allowed(self) -> bool:
        """capture モードを許可するか判定."""
        with self._lock:
            if not _is_supported_rate(self._local.sample_rate):
                return False
            if not self._require_peer:
                return True
            return self._synced

    # --- polling ---
    def _poll_peer(self) -> None:
        while not self._stop.wait(self._poll_interval):
            sock = self._ensure_client()
            if sock is None:
                with self._lock:
                    self._synced = False
                continue
            try:
                sock.send_json({"cmd": "SYNC", "peer": self._get_local().to_dict()})
                resp = sock.recv_json()
            except zmq.Again:
                # REQ/REP は timeout 後に EFSM になり得るため、ソケットを作り直す
                self._reset_client()
                with self._lock:
                    self._synced = False
                continue
            except zmq.ZMQError:
                self._reset_client()
                with self._lock:
                    self._synced = False
                continue
            except ValueError:
                self._reset_client()
                with self._lock:
                    self._synced = False
                continue

            if not isinstance(resp, dict) or resp.get("status") != "ok":
                with self._lock:
                    self._synced = False
                continue

            data = resp.get("data") or {}
            if not isinstance(data, dict):
                # 想定外の型はプロトコル破損扱いとして同期解除＆リセット
                with self._lock:
                    self._synced = False
                self._reset_client()
                continue
            try:
                self._set_peer(ControlStatus.from_dict(data))
            except Exception:  # noqa: BLE001
                with self._lock:
                    self._synced = False
                # 破損した応答の可能性があるためリセット
                self._reset_client()

    def _ensure_client(self) -> zmq.Socket | None:
        if not self._peer_endpoint:
            return None
        if self._ctx is None:
            return None
        if self._client is not None:
            return self._client
        try:
            sock = self._ctx.socket(zmq.REQ)
            sock.linger = 0
            sock.rcvtimeo = self._timeout_ms
            sock.sndtimeo = self._timeout_ms
            sock.connect(self._peer_endpoint)
            self._client = sock
            return sock
        except zmq.ZMQError:
            self._reset_client()
            return None

    def _reset_client(self) -> None:
        if self._client is None:
            return
        try:
            self._client.close(0)
        except zmq.ZMQError:
            pass
        self._client = None

    # --- internal ---
    def _refresh_synced_locked(self) -> None:
        if self._peer is None:
            self._synced = False
            return
        if not (self._local.is_supported() and self._peer.is_supported()):
            self._synced = False
            return
        self._synced = (
            self._peer.sample_rate == self._local.sample_rate
            and self._peer.fmt == self._local.fmt
            and self._peer.channels == self._local.channels
        )


__all__ = [
    "ALLOWED_SAMPLE_RATES",
    "ControlStatus",
    "ControlPlaneServer",
    "ControlPlaneSync",
    "_is_supported_rate",
]
