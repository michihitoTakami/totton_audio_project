"""USB(PC) -> I2S(TX) bridge for Raspberry Pi.

要件:
- USB入力(PC)の rate/format を取得し I2S へ流す（取り急ぎは **パススルー優先**）
- 44.1k系/48k系の切替を検知し安全に再初期化（フェード/ミュート）
- USB切断(PC再起動/抜き差し)や XRUN で落ちても自動復帰
- Jetson側が再起動しても継続運用できるよう、入力が無い時はサイレンスを送出してI2Sを維持

実装方針:
- 「橋渡し」に徹し、系統変換（任意のリサンプル/フォーマット変換/DSP）は行わない
- `arecord|aplay` のみで構成し、再初期化（レート/フォーマット切替）・切断復帰・無入力時サイレンス送出に注力する

将来拡張（Issue #824）:
- LAN制御プレーンで rate/format/ch を相互監視し、Jetson側の再初期化へ同期させる。
  そのため本実装は現在値を status ファイルに書き出せるようにしている（後で ZMQ/UDP/HTTP に置換可能）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional, Tuple

from .control_plane import ControlPlaneSync

_CONFIG_OVERRIDES: dict[str, str] = {}


def _load_config_overrides() -> dict[str, str]:
    path = os.getenv(
        "USB_I2S_CONFIG_PATH", "/var/lib/usb-i2s-bridge/config.env"
    ).strip()
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        text = cfg_path.read_text()
    except OSError:
        return {}
    overrides: dict[str, str] = {}
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "USB_I2S_STATUS_REPORT_URL" and value == "":
            # Allow docker/default env to enable status reporting by default (#1079)
            continue
        overrides[key] = value
    return overrides


_CONFIG_OVERRIDES = _load_config_overrides()

_DEFAULT_CAPTURE_DEVICE = "hw:2,0"  # USB Audio in (typical)
_DEFAULT_PLAYBACK_DEVICE = "hw:0,0"  # I2S out (typical)
_DEFAULT_CHANNELS = 2
# 重要: 44.1k/48k を勝手に変換しないことが最優先。
# hw_params が未確定/closed の間に fallback へ倒れる場合は、
# 48k 固定よりも 44.1k をデフォルトに寄せた方が害が少ない（Issue #949）。
_DEFAULT_FALLBACK_RATE = 44100
_DEFAULT_PREFERRED_FORMAT = "S32_LE"  # 24in32 を推奨

_DEFAULT_ALSA_BUFFER_TIME_US = 200_000
_DEFAULT_ALSA_LATENCY_TIME_US = 20_000
_DEFAULT_QUEUE_TIME_NS = 100_000_000

_DEFAULT_FADE_MS = 80
_DEFAULT_RESTART_BACKOFF_SEC = 0.5
_DEFAULT_POLL_INTERVAL_SEC = 1.0
_DEFAULT_STATUS_PATH = Path("/var/run/usb-i2s-bridge/status.json")

# LAN 制御プレーン（Issue #824）
# NOTE(#950):
# 60100/60101 の ZeroMQ 制御プレーンは「標準構成」では Jetson 側受け口が無いことが多く、
# デフォルト有効だと “同期できない→動かない/不安定” の温床になる。
# そのためデフォルトは無効（空）に寄せ、必要な環境のみ明示的に有効化する。
_DEFAULT_CONTROL_ENDPOINT = os.getenv("USB_I2S_CONTROL_ENDPOINT", "").strip()
_DEFAULT_CONTROL_PEER = os.getenv("USB_I2S_CONTROL_PEER", "").strip()
_DEFAULT_CONTROL_REQUIRE_PEER = os.getenv(
    "USB_I2S_CONTROL_REQUIRE_PEER", "0"
).strip().lower() not in {"0", "false", "no", "off"}
_DEFAULT_CONTROL_POLL_INTERVAL = float(
    os.getenv("USB_I2S_CONTROL_POLL_INTERVAL_SEC", "1.0")
)
_DEFAULT_CONTROL_TIMEOUT_MS = int(os.getenv("USB_I2S_CONTROL_TIMEOUT_MS", "2000"))

# Jetson Web(:80) への状態レポート（#950）
# 空文字の場合もデフォルト値にフォールバック
_DEFAULT_STATUS_REPORT_URL = (
    os.getenv("USB_I2S_STATUS_REPORT_URL", "").strip()
    or "http://192.168.55.1/i2s/peer-status"
)
_DEFAULT_STATUS_REPORT_TIMEOUT_MS = int(
    os.getenv("USB_I2S_STATUS_REPORT_TIMEOUT_MS", "300")
)
_DEFAULT_STATUS_REPORT_MIN_INTERVAL_SEC = float(
    os.getenv("USB_I2S_STATUS_REPORT_MIN_INTERVAL_SEC", "1.0")
)

_ALSA_BYTES_PER_SAMPLE: dict[str, int] = {
    "S16_LE": 2,
    "S24_3LE": 3,
    "S32_LE": 4,
    "S16_BE": 2,
    "S24_3BE": 3,
    "S32_BE": 4,
}

_SUPPORTED_ALSA_FORMATS = sorted(_ALSA_BYTES_PER_SAMPLE.keys())


@dataclass
class UsbI2sBridgeConfig:
    capture_device: str = _DEFAULT_CAPTURE_DEVICE
    playback_device: str = _DEFAULT_PLAYBACK_DEVICE
    channels: int = _DEFAULT_CHANNELS
    fallback_rate: int = _DEFAULT_FALLBACK_RATE
    preferred_format: str = _DEFAULT_PREFERRED_FORMAT
    alsa_buffer_time_us: int = _DEFAULT_ALSA_BUFFER_TIME_US
    alsa_latency_time_us: int = _DEFAULT_ALSA_LATENCY_TIME_US
    queue_time_ns: int = _DEFAULT_QUEUE_TIME_NS
    fade_ms: int = _DEFAULT_FADE_MS
    poll_interval_sec: float = _DEFAULT_POLL_INTERVAL_SEC
    restart_backoff_sec: float = _DEFAULT_RESTART_BACKOFF_SEC
    keep_silence_when_no_capture: bool = True
    status_path: Path | None = _DEFAULT_STATUS_PATH
    status_report_url: str | None = _DEFAULT_STATUS_REPORT_URL or None
    status_report_timeout_ms: int = _DEFAULT_STATUS_REPORT_TIMEOUT_MS
    status_report_min_interval_sec: float = _DEFAULT_STATUS_REPORT_MIN_INTERVAL_SEC
    control_endpoint: str | None = _DEFAULT_CONTROL_ENDPOINT or None
    control_peer: str | None = _DEFAULT_CONTROL_PEER or None
    control_require_peer: bool = _DEFAULT_CONTROL_REQUIRE_PEER
    control_poll_interval_sec: float = _DEFAULT_CONTROL_POLL_INTERVAL
    control_timeout_ms: int = _DEFAULT_CONTROL_TIMEOUT_MS
    dry_run: bool = False

    def validate(self) -> None:
        if self.channels <= 0:
            raise ValueError("channels must be > 0")
        if self.fallback_rate <= 0:
            raise ValueError("fallback_rate must be > 0")
        if self.fade_ms < 0:
            raise ValueError("fade_ms must be >= 0")
        if self.alsa_buffer_time_us <= 0 or self.alsa_latency_time_us <= 0:
            raise ValueError("ALSA buffer/latency must be > 0")
        if self.queue_time_ns <= 0:
            raise ValueError("queue_time_ns must be > 0")
        if self.poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be > 0")
        if self.restart_backoff_sec < 0:
            raise ValueError("restart_backoff_sec must be >= 0")

        if self.status_path is not None and not isinstance(self.status_path, Path):
            raise ValueError("status_path must be a Path or None")
        if self.status_report_timeout_ms <= 0:
            raise ValueError("status_report_timeout_ms must be > 0")
        if self.status_report_min_interval_sec < 0:
            raise ValueError("status_report_min_interval_sec must be >= 0")
        if self.control_poll_interval_sec <= 0:
            raise ValueError("control_poll_interval_sec must be > 0")
        if self.control_timeout_ms <= 0:
            raise ValueError("control_timeout_ms must be > 0")


def _env_int(name: str, default: int) -> int:
    raw = _CONFIG_OVERRIDES.get(name, os.getenv(name))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _CONFIG_OVERRIDES.get(name, os.getenv(name))
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return _CONFIG_OVERRIDES.get(name, os.getenv(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = _CONFIG_OVERRIDES.get(name, os.getenv(name))
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path | None) -> Path | None:
    raw = _CONFIG_OVERRIDES.get(name, os.getenv(name))
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return None
    return Path(raw)


def _resolve_card_index(card_id: str) -> Optional[int]:
    """ALSA card ID (e.g. 'UAC2Gadget') -> card number.

    - /proc/asound/cards の [ID] 部分を優先
    - 読めない/見つからない場合は None
    """
    wanted = card_id.strip()
    if not wanted:
        return None
    try:
        text = Path("/proc/asound/cards").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s+\[(?P<id>[^\]]+)\]", line)
        if not m:
            continue
        idx = int(m.group(1))
        cid = m.group("id").strip()
        if cid == wanted:
            return idx
    return None


def _parse_alsa_hw_device(device: str) -> Optional[tuple[int, int]]:
    """ALSA device string -> (card, pcm).

    Supported:
    - hw:2,0 / plughw:2,0
    - hw:Pi2Jetson,0
    - hw:CARD=UAC2Gadget,DEV=0 (and plughw:...)
    """
    # hw:2,0
    m = re.match(r"^(?:plughw:|hw:)(?P<card>\d+),(?P<pcm>\d+)$", device)
    if m:
        return int(m.group("card")), int(m.group("pcm"))

    # hw:CARD=UAC2Gadget,DEV=0
    m = re.match(
        r"^(?:plughw:|hw:)(?:CARD=)?(?P<card>[^,]+),(?:DEV=)?(?P<pcm>\d+)$",
        device,
    )
    if not m:
        return None
    card_raw = m.group("card").strip()
    pcm = int(m.group("pcm"))
    if card_raw.isdigit():
        return int(card_raw), pcm
    idx = _resolve_card_index(card_raw)
    if idx is None:
        return None
    return idx, pcm


def _hw_params_path(device: str, stream: str) -> Optional[Path]:
    """ALSA device string -> /proc/asound/.../hw_params.

    stream: 'c' for capture, 'p' for playback
    """
    parsed = _parse_alsa_hw_device(device)
    if parsed is None:
        return None
    card, pcm = parsed
    return Path(f"/proc/asound/card{card}/pcm{pcm}{stream}/sub0/hw_params")


def _pcm_device_node(device: str, stream: str) -> Optional[Path]:
    """ALSA device string (hw:2,0) -> /dev/snd/pcmC2D0c のようなデバイスノード.

    電源断復帰やUSB抜き差しなどでは hw_params が "closed" のままでも、
    デバイスノードの出現で「デバイスとして存在する」ことは判定できる。
    """
    parsed = _parse_alsa_hw_device(device)
    if parsed is None:
        return None
    card, pcm = parsed
    suffix = "c" if stream == "c" else "p"
    return Path(f"/dev/snd/pcmC{card}D{pcm}{suffix}")


def _device_present(device: str, stream: str) -> bool:
    node = _pcm_device_node(device, stream)
    if node is None:
        return False
    return node.exists()


def _parse_hw_params_rate(text: str) -> Optional[int]:
    # hw_params が未オープンの場合は "closed" のみが入る。
    # ただし環境によっては行頭に空白が入るので strip して判定する。
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("closed"):
            return None
        break

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("rate:"):
            # e.g. "rate: 44100 (44100/1)"
            for token in s.split():
                if token.isdigit():
                    return int(token)
    return None


def _parse_hw_params_format(text: str) -> Optional[str]:
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("closed"):
            return None
        break

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("format:"):
            # e.g. "format: S32_LE"
            parts = s.split()
            if len(parts) >= 2:
                return parts[1].strip()
    return None


def _probe_capture_params(device: str) -> Tuple[Optional[int], Optional[str]]:
    # 注意:
    # - /proc/asound/.../hw_params は「開いている substream」に紐づくため、
    #   sub0 固定だと常に "closed" を読んでしまう環境がある。
    # - その場合、rate/fmt が None のままになり、fallback_rate + plughw に固定される。
    parsed = _parse_alsa_hw_device(device)
    if parsed is None:
        return None, None
    card, pcm = parsed

    base = Path(f"/proc/asound/card{card}/pcm{pcm}c")
    candidates: list[Path] = []

    # 互換性のため従来の sub0 を最優先で試す
    p0 = base / "sub0" / "hw_params"
    if p0.exists():
        candidates.append(p0)

    # 次に sub* を総当りする（どれか一つが active ならそこに rate/fmt が出る）
    try:
        for p in sorted(base.glob("sub*/hw_params")):
            if p not in candidates and p.exists():
                candidates.append(p)
    except OSError:
        pass

    best_rate: Optional[int] = None
    best_fmt: Optional[str] = None
    for p in candidates:
        try:
            text = p.read_text()
        except OSError:
            continue
        r = _parse_hw_params_rate(text)
        f = _parse_hw_params_format(text)
        if r is not None:
            best_rate = r
        if f is not None:
            best_fmt = f
        if best_rate is not None and best_fmt is not None:
            break

    return best_rate, best_fmt


class StatusReporter:
    """Jetson Web(:80)へ状態を非同期にPOSTする（#950）.

    - 音声パスをブロックしないことを最優先にする
    - 送信失敗は握りつぶし、最新値のみを送る（キューしない）
    """

    def __init__(
        self,
        *,
        url: str,
        timeout_ms: int = 300,
        min_interval_sec: float = 1.0,
    ) -> None:
        self._url = str(url).strip()
        self._timeout_sec = max(0.05, float(timeout_ms) / 1000.0)
        self._min_interval_sec = max(0.0, float(min_interval_sec))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._last_sent = 0.0  # monotonic seconds

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self._url:
            return
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(
            target=self._run, name="usb_i2s_status_reporter", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    def submit(self, payload: dict) -> None:
        """最新payloadを上書きして送信を促す."""
        if not self._url:
            return
        with self._lock:
            self._latest = payload
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self._stop.is_set():
                return

            with self._lock:
                payload = self._latest
                self._latest = None

            if not payload:
                continue

            # 送信間隔制限（連続レート更新等でのスパムを抑制）
            now = time.monotonic()
            wait = (self._last_sent + self._min_interval_sec) - now
            if wait > 0:
                # stopを尊重しつつ待つ
                if self._stop.wait(timeout=wait):
                    return

            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self._url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout_sec):
                    pass
                self._last_sent = time.monotonic()
            except Exception:
                # 送信失敗は握りつぶす（音を止めない）
                self._last_sent = time.monotonic()


def _persist_status(
    path: Path,
    *,
    running: bool,
    mode: str,
    sample_rate: int,
    alsa_format: str,
    channels: int,
    generation: int = 0,
    xruns: int | None = None,
    last_error: str | None = None,
    last_error_at_unix_ms: int | None = None,
    uptime_sec: float | None = None,
    reporter: StatusReporter | None = None,
) -> None:
    """Issue #824 の制御プレーン連携用に、現在値をファイルへ書き出す."""
    payload = {
        "running": running,
        "mode": mode,  # capture / silence / none
        "sample_rate": int(sample_rate),
        "format": str(alsa_format),
        "channels": int(channels),
        "generation": int(generation),
        "updated_at_unix_ms": int(time.time() * 1000),
        "note": "For Issue #824/#950 (rate/format/ch status).",
    }
    if xruns is not None:
        payload["xruns"] = int(xruns)
    if last_error is not None:
        payload["last_error"] = str(last_error)
    if last_error_at_unix_ms is not None:
        payload["last_error_at_unix_ms"] = int(last_error_at_unix_ms)
    if uptime_sec is not None:
        payload["uptime_sec"] = float(uptime_sec)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError:
        # 送出自体は継続する
        pass
    if reporter:
        reporter.submit(payload)


def _build_control_sync(cfg: UsbI2sBridgeConfig) -> ControlPlaneSync | None:
    endpoint = (cfg.control_endpoint or "").strip()
    if not endpoint:
        return None
    return ControlPlaneSync(
        endpoint=endpoint,
        peer_endpoint=(cfg.control_peer or "").strip() or None,
        require_peer=bool(cfg.control_require_peer),
        poll_interval_sec=cfg.control_poll_interval_sec,
        timeout_ms=cfg.control_timeout_ms,
    )


def _alsa_bytes_per_sample(fmt: str) -> int:
    return _ALSA_BYTES_PER_SAMPLE.get(fmt, 4)


def _build_arecord_command(
    cfg: UsbI2sBridgeConfig, *, sample_rate: int, alsa_format: str
) -> list[str]:
    # NOTE: -t raw でヘッダ無しPCM
    return [
        "arecord",
        "-q",
        "-D",
        cfg.capture_device,
        "-t",
        "raw",
        "-c",
        str(cfg.channels),
        "-r",
        str(int(sample_rate)),
        "-f",
        str(alsa_format),
        "--buffer-time",
        str(int(cfg.alsa_buffer_time_us)),
        "--period-time",
        str(int(cfg.alsa_latency_time_us)),
    ]


def _build_aplay_command(
    cfg: UsbI2sBridgeConfig, *, sample_rate: int, alsa_format: str
) -> list[str]:
    return [
        "aplay",
        "-q",
        "-D",
        cfg.playback_device,
        "-t",
        "raw",
        "-c",
        str(cfg.channels),
        "-r",
        str(int(sample_rate)),
        "-f",
        str(alsa_format),
        "--buffer-time",
        str(int(cfg.alsa_buffer_time_us)),
        "--period-time",
        str(int(cfg.alsa_latency_time_us)),
    ]


def _terminate_process(
    proc: subprocess.Popen | None, *, timeout_sec: float = 2.0
) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout_sec)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _run_with_arecord_aplay(
    cfg: UsbI2sBridgeConfig, control: ControlPlaneSync | None = None
) -> None:
    """arecord|aplay パイプでUSB->I2Sを維持しつつ、切断時も自動復旧する.

    - GStreamer を使わず、標準 ALSA ツールでシンプルに構成する（Issue #919）
    - USB切断/復帰や XRUN で落ちても再起動
    - keep_silence=true の場合、入力が無い時はサイレンス送出で I2S を維持
    """
    cfg.validate()

    reporter: StatusReporter | None = None
    if cfg.status_report_url:
        reporter = StatusReporter(
            url=cfg.status_report_url,
            timeout_ms=int(cfg.status_report_timeout_ms),
            min_interval_sec=float(cfg.status_report_min_interval_sec),
        )
        reporter.start()

    # 状態
    last_rate = int(cfg.fallback_rate)
    last_observed_capture_format: Optional[str] = None
    current_mode = "none"  # capture / silence / none
    current_rate = int(cfg.fallback_rate)
    current_fmt = str(cfg.preferred_format)

    start_monotonic = time.monotonic()
    xrun_count = 0
    last_error: str | None = None
    last_error_at_unix_ms: int | None = None
    error_lock = threading.Lock()

    aplay_proc: subprocess.Popen | None = None
    arecord_proc: subprocess.Popen | None = None
    arecord_out: IO[bytes] | None = None
    aplay_in: IO[bytes] | None = None
    aplay_err_thread: threading.Thread | None = None
    arecord_err_thread: threading.Thread | None = None

    status_generation = 0
    last_status_key: tuple[str, int, str, int] | None = None  # (mode, rate, fmt, ch)

    def _write_status(mode: str, rate: int, fmt: str) -> None:
        nonlocal status_generation, last_status_key
        key = (str(mode), int(rate), str(fmt), int(cfg.channels))
        if key != last_status_key:
            status_generation += 1
            last_status_key = key
        if cfg.status_path:
            uptime_sec = max(0.0, time.monotonic() - start_monotonic)
            with error_lock:
                xruns_snapshot = xrun_count
                last_error_snapshot = last_error
                last_error_at_snapshot = last_error_at_unix_ms
            _persist_status(
                cfg.status_path,
                running=(mode != "none"),
                mode=mode if mode != "none" else "none",
                sample_rate=rate,
                alsa_format=fmt,
                channels=cfg.channels,
                generation=status_generation,
                xruns=xruns_snapshot,
                last_error=last_error_snapshot,
                last_error_at_unix_ms=last_error_at_snapshot,
                uptime_sec=uptime_sec,
                reporter=reporter,
            )
        if control:
            control.update_local(
                running=(mode != "none"),
                mode=mode if mode != "none" else "none",
                sample_rate=rate,
                fmt=fmt,
                channels=cfg.channels,
            )

    def _record_error(message: str, *, is_xrun: bool = False) -> None:
        nonlocal xrun_count, last_error, last_error_at_unix_ms, current_mode
        with error_lock:
            mode_snapshot = current_mode
            if is_xrun and mode_snapshot == "capture":
                xrun_count += 1
            last_error = str(message)
            last_error_at_unix_ms = int(time.time() * 1000)

    def _start_error_monitor(
        pipe: IO[bytes] | None, *, name: str
    ) -> threading.Thread | None:
        if pipe is None:
            return None

        def _run() -> None:
            for raw in iter(pipe.readline, b""):
                if not raw:
                    break
                try:
                    line = raw.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if not line:
                    continue
                lowered = line.lower()
                if "xrun" in lowered or "overrun" in lowered or "underrun" in lowered:
                    _record_error(f"{name}: {line}", is_xrun=True)

        thread = threading.Thread(
            target=_run, name=f"usb_i2s_{name}_stderr", daemon=True
        )
        thread.start()
        return thread

    def _restart_aplay(*, rate: int, fmt: str) -> None:
        nonlocal aplay_proc, aplay_in, current_rate, current_fmt, aplay_err_thread
        _terminate_process(aplay_proc)
        aplay_proc = None
        aplay_in = None
        aplay_err_thread = None
        cmd = _build_aplay_command(cfg, sample_rate=rate, alsa_format=fmt)
        if cfg.dry_run:
            print(f"[usb_i2s_bridge] aplay cmd: {' '.join(cmd)}")
            return
        print(f"[usb_i2s_bridge] start aplay rate={rate} fmt={fmt} cmd={' '.join(cmd)}")
        try:
            aplay_proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )
            assert aplay_proc.stdin is not None
            aplay_in = aplay_proc.stdin
            aplay_err_thread = _start_error_monitor(aplay_proc.stderr, name="aplay")
            current_rate = int(rate)
            current_fmt = str(fmt)
        except FileNotFoundError as e:
            print(f"[usb_i2s_bridge] aplay not found: {e}; retrying")
            _record_error(f"aplay not found: {e}")
            aplay_proc = None
            aplay_in = None
        except Exception as e:
            print(f"[usb_i2s_bridge] failed to start aplay: {e}; retrying")
            _record_error(f"aplay start failed: {e}")
            aplay_proc = None
            aplay_in = None

    def _restart_arecord(*, rate: int, fmt: str) -> None:
        nonlocal arecord_proc, arecord_out, arecord_err_thread
        _terminate_process(arecord_proc)
        arecord_proc = None
        arecord_out = None
        arecord_err_thread = None
        cmd = _build_arecord_command(cfg, sample_rate=rate, alsa_format=fmt)
        if cfg.dry_run:
            print(f"[usb_i2s_bridge] arecord cmd: {' '.join(cmd)}")
            return
        print(
            f"[usb_i2s_bridge] start arecord rate={rate} fmt={fmt} cmd={' '.join(cmd)}"
        )
        try:
            arecord_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            assert arecord_proc.stdout is not None
            arecord_out = arecord_proc.stdout
            arecord_err_thread = _start_error_monitor(
                arecord_proc.stderr, name="arecord"
            )
        except FileNotFoundError as e:
            print(f"[usb_i2s_bridge] arecord not found: {e}; retrying")
            _record_error(f"arecord not found: {e}")
            arecord_proc = None
            arecord_out = None
        except Exception as e:
            print(f"[usb_i2s_bridge] failed to start arecord: {e}; retrying")
            _record_error(f"arecord start failed: {e}")
            arecord_proc = None
            arecord_out = None

    def _stop_arecord() -> None:
        nonlocal arecord_proc, arecord_out
        _terminate_process(arecord_proc)
        arecord_proc = None
        arecord_out = None

    def _stop_all() -> None:
        nonlocal aplay_proc, aplay_in
        _stop_arecord()
        _terminate_process(aplay_proc)
        aplay_proc = None
        aplay_in = None

    def _handle_term(signum: int, frame) -> None:  # noqa: ANN001
        _ = signum, frame
        _stop_all()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    try:
        # 初期: keep_silence=true ならサイレンスで I2S を立ち上げる
        if cfg.keep_silence_when_no_capture:
            current_mode = "silence"
            _restart_aplay(rate=current_rate, fmt=current_fmt)
            _write_status(current_mode, current_rate, current_fmt)

        # 送出チャンク（小さめにして、切断検知/再起動の反応性を上げる）
        chunk_ms = 10
        next_poll = 0.0

        while True:
            now = time.monotonic()
            if now >= next_poll:
                capture_present = _device_present(cfg.capture_device, "c")
                rate, fmt = _probe_capture_params(cfg.capture_device)
                capture_allowed = control.capture_allowed() if control else True

                if not capture_present and not cfg.keep_silence_when_no_capture:
                    desired_mode = "none"
                else:
                    desired_mode = (
                        "capture" if capture_present and capture_allowed else "silence"
                    )

                if rate is not None:
                    last_rate = int(rate)
                elif capture_present and desired_mode == "capture":
                    # デバイスはあるが hw_params がまだ取れない場合は一旦フォールバック
                    last_rate = int(cfg.fallback_rate)

                if fmt is not None:
                    last_observed_capture_format = fmt

                desired_rate = int(last_rate)
                desired_fmt = str(
                    fmt or last_observed_capture_format or cfg.preferred_format
                )

                if capture_present and not capture_allowed:
                    print(
                        "[usb_i2s_bridge] peer not synced; holding capture and sending silence"
                    )

                if cfg.dry_run:
                    print(
                        f"[usb_i2s_bridge] mode={desired_mode} rate={desired_rate} fmt={desired_fmt}"
                    )
                    _restart_aplay(rate=desired_rate, fmt=desired_fmt)
                    if desired_mode == "capture":
                        _restart_arecord(rate=desired_rate, fmt=desired_fmt)
                    return

                # aplay は必ず起動（none 以外）
                if desired_mode == "none":
                    if current_mode != "none":
                        print(
                            "[usb_i2s_bridge] no capture and keep_silence=false; stopping"
                        )
                        _stop_all()
                        current_mode = "none"
                        _write_status(current_mode, current_rate, current_fmt)
                    next_poll = now + cfg.poll_interval_sec
                    time.sleep(cfg.poll_interval_sec)
                    continue

                need_restart_aplay = (
                    aplay_proc is None
                    or aplay_proc.poll() is not None
                    or current_rate != desired_rate
                    or current_fmt != desired_fmt
                )
                if need_restart_aplay:
                    _restart_aplay(rate=desired_rate, fmt=desired_fmt)

                # arecord は capture の時だけ起動
                if desired_mode == "capture":
                    need_restart_arecord = (
                        arecord_proc is None or arecord_proc.poll() is not None
                    )
                    if need_restart_arecord:
                        _restart_arecord(rate=desired_rate, fmt=desired_fmt)
                else:
                    _stop_arecord()

                current_mode = desired_mode
                _write_status(current_mode, desired_rate, desired_fmt)
                next_poll = now + cfg.poll_interval_sec

            # 出力先が無ければ待つ
            if aplay_in is None or aplay_proc is None or aplay_proc.poll() is not None:
                time.sleep(cfg.restart_backoff_sec)
                continue

            frame_bytes = int(cfg.channels) * _alsa_bytes_per_sample(current_fmt)
            frames_per_chunk = max(1, int(current_rate * chunk_ms / 1000))
            chunk_bytes = max(frame_bytes, frames_per_chunk * frame_bytes)

            try:
                if (
                    current_mode == "capture"
                    and arecord_out is not None
                    and arecord_proc is not None
                ):
                    data = arecord_out.read(chunk_bytes)
                    if not data:
                        rc = arecord_proc.poll()
                        print(
                            f"[usb_i2s_bridge] arecord EOF/exit rc={rc}; switching to silence and restarting"
                        )
                        _record_error(f"arecord EOF/exit rc={rc}", is_xrun=False)
                        _stop_arecord()
                        current_mode = (
                            "silence" if cfg.keep_silence_when_no_capture else "none"
                        )
                        time.sleep(cfg.restart_backoff_sec)
                        continue
                    aplay_in.write(data)
                else:
                    aplay_in.write(b"\x00" * chunk_bytes)
            except BrokenPipeError:
                print("[usb_i2s_bridge] aplay broken pipe; restarting")
                _record_error("aplay broken pipe", is_xrun=False)
                _terminate_process(aplay_proc)
                aplay_proc = None
                aplay_in = None
                time.sleep(cfg.restart_backoff_sec)
            except OSError as e:
                print(f"[usb_i2s_bridge] I/O error: {e}; restarting aplay/arecord")
                _record_error(f"I/O error: {e}", is_xrun=False)
                _stop_all()
                time.sleep(cfg.restart_backoff_sec)
    finally:
        if reporter:
            reporter.stop()


def _parse_args(argv: list[str] | None = None) -> UsbI2sBridgeConfig:
    parser = argparse.ArgumentParser(description="USB(PC) -> I2S bridge (Pi5)")
    parser.add_argument(
        "--capture-device",
        default=_env_str("USB_I2S_CAPTURE_DEVICE", _DEFAULT_CAPTURE_DEVICE),
    )
    parser.add_argument(
        "--playback-device",
        default=_env_str("USB_I2S_PLAYBACK_DEVICE", _DEFAULT_PLAYBACK_DEVICE),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=_env_int("USB_I2S_CHANNELS", _DEFAULT_CHANNELS),
    )
    parser.add_argument(
        "--fallback-rate",
        type=int,
        default=_env_int("USB_I2S_FALLBACK_RATE", _DEFAULT_FALLBACK_RATE),
    )
    parser.add_argument(
        "--preferred-format",
        default=_env_str("USB_I2S_PREFERRED_FORMAT", _DEFAULT_PREFERRED_FORMAT),
        choices=_SUPPORTED_ALSA_FORMATS,
    )
    parser.add_argument(
        "--alsa-buffer-time-us",
        type=int,
        default=_env_int("USB_I2S_ALSA_BUFFER_TIME_US", _DEFAULT_ALSA_BUFFER_TIME_US),
    )
    parser.add_argument(
        "--alsa-latency-time-us",
        type=int,
        default=_env_int("USB_I2S_ALSA_LATENCY_TIME_US", _DEFAULT_ALSA_LATENCY_TIME_US),
    )
    parser.add_argument(
        "--queue-time-ns",
        type=int,
        default=_env_int("USB_I2S_QUEUE_TIME_NS", _DEFAULT_QUEUE_TIME_NS),
    )
    parser.add_argument(
        "--fade-ms",
        type=int,
        default=_env_int("USB_I2S_FADE_MS", _DEFAULT_FADE_MS),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=_env_float("USB_I2S_POLL_INTERVAL_SEC", _DEFAULT_POLL_INTERVAL_SEC),
    )
    parser.add_argument(
        "--restart-backoff",
        type=float,
        default=_env_float("USB_I2S_RESTART_BACKOFF_SEC", _DEFAULT_RESTART_BACKOFF_SEC),
    )
    parser.add_argument(
        "--keep-silence",
        dest="keep_silence_when_no_capture",
        action="store_true",
        default=_env_bool("USB_I2S_KEEP_SILENCE", True),
        help="USB入力が無い時でもサイレンスをI2Sへ送ってリンク維持する (default: true)",
    )
    parser.add_argument(
        "--no-keep-silence",
        dest="keep_silence_when_no_capture",
        action="store_false",
        help="USB入力が無い時は出力を停止する",
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=_env_path("USB_I2S_STATUS_PATH", _DEFAULT_STATUS_PATH),
        help="現在の rate/format/ch をJSONで書き出すパス（Issue #824 連携用）。空なら無効。",
    )
    parser.add_argument(
        "--status-report-url",
        default=_env_str("USB_I2S_STATUS_REPORT_URL", _DEFAULT_STATUS_REPORT_URL),
        help="Jetson Web(:80)へ状態をPOSTするURL（例: http://192.168.55.1/i2s/peer-status）。空なら無効。",
    )
    parser.add_argument(
        "--status-report-timeout-ms",
        type=int,
        default=_env_int(
            "USB_I2S_STATUS_REPORT_TIMEOUT_MS", _DEFAULT_STATUS_REPORT_TIMEOUT_MS
        ),
        help="状態POSTのタイムアウト (ms)",
    )
    parser.add_argument(
        "--status-report-min-interval-sec",
        type=float,
        default=_env_float(
            "USB_I2S_STATUS_REPORT_MIN_INTERVAL_SEC",
            _DEFAULT_STATUS_REPORT_MIN_INTERVAL_SEC,
        ),
        help="状態POSTの最小送信間隔 (秒)",
    )
    parser.add_argument(
        "--control-endpoint",
        default=_env_str("USB_I2S_CONTROL_ENDPOINT", _DEFAULT_CONTROL_ENDPOINT),
        help="ZeroMQ REP エンドポイント (I2S 制御プレーン)。空文字で無効化。",
    )
    parser.add_argument(
        "--control-peer",
        default=_env_str("USB_I2S_CONTROL_PEER", _DEFAULT_CONTROL_PEER),
        help="制御プレーンの相手先 REQ エンドポイント（例: tcp://jetson:60101）。",
    )
    parser.add_argument(
        "--control-require-peer",
        action="store_true",
        default=_DEFAULT_CONTROL_REQUIRE_PEER,
        help="peer と同期できるまで capture を許可しない",
    )
    parser.add_argument(
        "--control-poll-interval",
        type=float,
        default=_DEFAULT_CONTROL_POLL_INTERVAL,
        help="制御プレーン同期のポーリング間隔 (秒)",
    )
    parser.add_argument(
        "--control-timeout-ms",
        type=int,
        default=_DEFAULT_CONTROL_TIMEOUT_MS,
        help="制御プレーン送受信タイムアウト (ms)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.getenv("USB_I2S_DRY_RUN", "false").lower() in {"1", "true", "yes"},
        help="起動コマンド/パイプラインを表示して終了",
    )
    args = parser.parse_args(argv)
    return UsbI2sBridgeConfig(
        capture_device=args.capture_device,
        playback_device=args.playback_device,
        channels=max(1, int(args.channels)),
        fallback_rate=max(1, int(args.fallback_rate)),
        preferred_format=str(args.preferred_format),
        alsa_buffer_time_us=max(1, int(args.alsa_buffer_time_us)),
        alsa_latency_time_us=max(1, int(args.alsa_latency_time_us)),
        queue_time_ns=max(1, int(args.queue_time_ns)),
        fade_ms=max(0, int(args.fade_ms)),
        poll_interval_sec=max(0.2, float(args.poll_interval)),
        restart_backoff_sec=max(0.0, float(args.restart_backoff)),
        keep_silence_when_no_capture=bool(args.keep_silence_when_no_capture),
        status_path=args.status_path,
        status_report_url=str(args.status_report_url or "").strip() or None,
        status_report_timeout_ms=max(1, int(args.status_report_timeout_ms)),
        status_report_min_interval_sec=max(
            0.0, float(args.status_report_min_interval_sec)
        ),
        control_endpoint=str(args.control_endpoint or "").strip() or None,
        control_peer=str(args.control_peer or "").strip() or None,
        control_require_peer=bool(args.control_require_peer),
        control_poll_interval_sec=max(0.2, float(args.control_poll_interval)),
        control_timeout_ms=max(1, int(args.control_timeout_ms)),
        dry_run=bool(args.dry_run),
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    cfg.validate()

    # NOTE: Issue #919: GStreamer を経由せず arecord|aplay パイプで動かす
    control = _build_control_sync(cfg)
    if control:
        control.start()
    try:
        _run_with_arecord_aplay(cfg, control)
    finally:
        if control:
            control.stop()


if __name__ == "__main__":
    main()
