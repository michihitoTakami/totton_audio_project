"""USB(PC) -> I2S(TX) bridge for Raspberry Pi.

要件:
- USB入力(PC)の rate/format を取得し I2S へ流す（取り急ぎは **パススルー優先**）
- 44.1k系/48k系の切替を検知し安全に再初期化（フェード/ミュート）
- USB切断(PC再起動/抜き差し)や XRUN で落ちても自動復帰
- Jetson側が再起動しても継続運用できるよう、入力が無い時はサイレンスを送出してI2Sを維持

実装方針:
- 可能な環境では Python GI (GStreamer) で volume を制御しフェードを実現
- GI が無い場合は gst-launch を再起動するフォールバック（この場合はミュート時間で代替）

将来拡張（Issue #824）:
- LAN制御プレーンで rate/format/ch を相互監視し、Jetson側の再初期化へ同期させる。
  そのため本実装は現在値を status ファイルに書き出せるようにしている（後で ZMQ/UDP/HTTP に置換可能）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

_DEFAULT_CAPTURE_DEVICE = "hw:2,0"  # USB Audio in (typical)
_DEFAULT_PLAYBACK_DEVICE = "hw:0,0"  # I2S out (typical)
_DEFAULT_CHANNELS = 2
_DEFAULT_FALLBACK_RATE = 48000
_DEFAULT_PREFERRED_FORMAT = "S32_LE"  # 24in32 を推奨

_DEFAULT_ALSA_BUFFER_TIME_US = 200_000
_DEFAULT_ALSA_LATENCY_TIME_US = 20_000
_DEFAULT_QUEUE_TIME_NS = 100_000_000

_DEFAULT_FADE_MS = 80
_DEFAULT_RESTART_BACKOFF_SEC = 0.5
_DEFAULT_POLL_INTERVAL_SEC = 1.0
_DEFAULT_STATUS_PATH = Path("/var/run/usb-i2s-bridge/status.json")
_DEFAULT_PASSTHROUGH = True

# ALSA hw_params (e.g. S24_3LE) -> GStreamer audio/x-raw format token
_ALSA_TO_GST_FORMAT: dict[str, str] = {
    "S16_LE": "S16LE",
    "S24_3LE": "S24LE",
    "S32_LE": "S32LE",
    "S16_BE": "S16BE",
    "S24_3BE": "S24BE",
    "S32_BE": "S32BE",
}


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
    passthrough: bool = _DEFAULT_PASSTHROUGH
    status_path: Path | None = _DEFAULT_STATUS_PATH
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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path | None) -> Path | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return None
    return Path(raw)


def _hw_params_path(device: str, stream: str) -> Optional[Path]:
    """ALSA device string -> /proc/asound/.../hw_params.

    stream: 'c' for capture, 'p' for playback
    """
    match = re.match(r"^(?:plughw:|hw:)(?P<card>\\d+),(?P<pcm>\\d+)", device)
    if not match:
        return None
    card = match.group("card")
    pcm = match.group("pcm")
    return Path(f"/proc/asound/card{card}/pcm{pcm}{stream}/sub0/hw_params")


def _parse_hw_params_rate(text: str) -> Optional[int]:
    if "closed" in text:
        return None
    for line in text.splitlines():
        if line.startswith("rate:"):
            for token in line.split():
                if token.isdigit():
                    return int(token)
    return None


def _parse_hw_params_format(text: str) -> Optional[str]:
    if "closed" in text:
        return None
    for line in text.splitlines():
        if line.startswith("format:"):
            # e.g. "format: S32_LE"
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].strip()
    return None


def _probe_capture_params(device: str) -> Tuple[Optional[int], Optional[str]]:
    path = _hw_params_path(device, "c")
    if path is None or not path.exists():
        return None, None
    try:
        text = path.read_text()
    except OSError:
        return None, None
    return _parse_hw_params_rate(text), _parse_hw_params_format(text)


def _gst_raw_format_from_alsa(alsa_format: Optional[str], preferred: str) -> str:
    if alsa_format and alsa_format in _ALSA_TO_GST_FORMAT:
        return _ALSA_TO_GST_FORMAT[alsa_format]
    if preferred in _ALSA_TO_GST_FORMAT:
        return _ALSA_TO_GST_FORMAT[preferred]
    return "S32LE"


def _persist_status(
    path: Path,
    *,
    running: bool,
    mode: str,
    sample_rate: int,
    alsa_format: str,
    channels: int,
) -> None:
    """Issue #824 の制御プレーン連携用に、現在値をファイルへ書き出す."""
    payload = {
        "running": running,
        "mode": mode,  # capture / silence / none
        "sample_rate": int(sample_rate),
        "format": str(alsa_format),
        "channels": int(channels),
        "updated_at_unix_ms": int(time.time() * 1000),
        "note": "For Issue #824 control-plane (rate/format/ch sync).",
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError:
        # 送出自体は継続する
        pass


def _choose_raw_caps(
    cfg: UsbI2sBridgeConfig,
    *,
    mode: str,
    sample_rate: int,
    capture_alsa_format: Optional[str],
    conversion: bool,
) -> tuple[str, str]:
    """caps 用の (raw_format, alsa_format_for_status) を決める.

    - passthrough + capture: ALSAで観測した format を優先
    - silence: preferred_format を使用
    - conversion=true の場合: 互換性のため preferred_format を使用
    """
    if mode == "silence":
        return _gst_raw_format_from_alsa(
            cfg.preferred_format, cfg.preferred_format
        ), cfg.preferred_format

    if conversion:
        return _gst_raw_format_from_alsa(
            cfg.preferred_format, cfg.preferred_format
        ), cfg.preferred_format

    if cfg.passthrough and capture_alsa_format:
        return _gst_raw_format_from_alsa(
            capture_alsa_format, cfg.preferred_format
        ), capture_alsa_format

    # passthrough しない（または不明）場合は preferred_format を使う
    return _gst_raw_format_from_alsa(
        cfg.preferred_format, cfg.preferred_format
    ), cfg.preferred_format


def build_gst_launch_command(
    cfg: UsbI2sBridgeConfig,
    *,
    mode: str,
    sample_rate: int,
    raw_format: str,
    conversion: bool,
) -> list[str]:
    """gst-launch-1.0 フォールバック用のコマンドを構築."""
    cfg.validate()
    if mode not in {"capture", "silence"}:
        raise ValueError("mode must be capture or silence")

    # NOTE: 安定性優先で period/avail ギリギリを避けるため buffer-time/latency-time を明示
    src: list[str]
    if mode == "capture":
        src = [
            "alsasrc",
            f"device={cfg.capture_device}",
            f"buffer-time={cfg.alsa_buffer_time_us}",
            f"latency-time={cfg.alsa_latency_time_us}",
            "do-timestamp=true",
        ]
    else:
        # I2S を維持するためのサイレンス源（USB未接続でも動かす）
        src = [
            "audiotestsrc",
            "is-live=true",
            "wave=silence",
        ]

    caps = f"audio/x-raw,rate={sample_rate},channels={cfg.channels},format={raw_format}"

    # 出力側は I2S の安定性優先。queue で揺らぎを吸収。
    pipeline = [
        "gst-launch-1.0",
        "-e",
        *src,
        "!",
        "queue",
        f"max-size-time={cfg.queue_time_ns}",
        "max-size-bytes=0",
        "max-size-buffers=0",
    ]

    # 取り急ぎは「受けた形式をそのまま投げる」(passthrough) を優先する。
    # ただし、I2S側が受け入れない format の場合はエラーになるため、
    # conversion=true で再起動するフォールバックを Supervisor 側で行う。
    if conversion:
        pipeline += [
            "!",
            "audioresample",
            "quality=10",
            "!",
            "audioconvert",
        ]

    pipeline += [
        "!",
        caps,
        "!",
        "volume",
        "volume=1.0",
        "!",
        "alsasink",
        f"device={cfg.playback_device}",
        # ライブ系で同期を取る（I2S維持）
        "sync=true",
        "async=false",
    ]
    return pipeline


def _try_import_gi_gst():
    """テスト環境で GI が無いことがあるため遅延import."""
    try:
        import gi  # type: ignore

        gi.require_version("Gst", "1.0")
        gi.require_version("GLib", "2.0")
        from gi.repository import GLib, Gst  # type: ignore

        return GLib, Gst
    except Exception:
        return None, None


def _run_with_gst_launch(cfg: UsbI2sBridgeConfig) -> None:
    """GI無し環境のフォールバック: gst-launch を再起動し続ける."""
    last_rate = cfg.fallback_rate
    last_observed_capture_format: Optional[str] = None
    current_mode = "silence" if cfg.keep_silence_when_no_capture else "capture"
    conversion = False

    while True:
        rate, fmt = _probe_capture_params(cfg.capture_device)
        if rate is None and not cfg.keep_silence_when_no_capture:
            # キャプチャが無いなら待機しつつリトライ（出力停止）
            time.sleep(cfg.poll_interval_sec)
            continue

        if rate is not None:
            last_rate = rate
        if fmt is not None:
            last_observed_capture_format = fmt

        desired_mode = "capture" if rate is not None else "silence"
        current_mode = desired_mode
        raw_format, status_alsa_format = _choose_raw_caps(
            cfg,
            mode=current_mode,
            sample_rate=last_rate,
            capture_alsa_format=fmt or last_observed_capture_format,
            conversion=conversion,
        )
        cmd = build_gst_launch_command(
            cfg,
            mode=current_mode,
            sample_rate=last_rate,
            raw_format=raw_format,
            conversion=conversion,
        )
        if cfg.dry_run:
            print(" ".join(cmd))
            return

        if cfg.status_path:
            _persist_status(
                cfg.status_path,
                running=True,
                mode=current_mode,
                sample_rate=last_rate,
                alsa_format=status_alsa_format,
                channels=cfg.channels,
            )

        print(
            f"[usb_i2s_bridge] (fallback) launch mode={current_mode} "
            f"rate={last_rate} fmt={raw_format} passthrough={cfg.passthrough} "
            f"conversion={conversion} cmd={' '.join(cmd)}"
        )
        proc = subprocess.Popen(cmd)
        # ポーリングでレート変化・切断を検知したら再起動
        try:
            while True:
                try:
                    rc = proc.wait(timeout=cfg.poll_interval_sec)
                    if rc != 0:
                        print(f"[usb_i2s_bridge] gst-launch exited rc={rc}; restarting")
                    break
                except subprocess.TimeoutExpired:
                    new_rate, new_fmt = _probe_capture_params(cfg.capture_device)
                    if (
                        new_rate != rate
                        or new_fmt != fmt
                        or (rate is None and new_rate is not None)
                        or (fmt is None and new_fmt is not None)
                    ):
                        print(
                            "[usb_i2s_bridge] detected capture change "
                            f"(rate {rate}->{new_rate}, fmt {fmt}->{new_fmt}); restarting"
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break
                    # USBが抜けた場合はサイレンスに切替（keep_silence=trueのみ）
                    if (
                        cfg.keep_silence_when_no_capture
                        and rate is not None
                        and new_rate is None
                    ):
                        print(
                            "[usb_i2s_bridge] capture disappeared; switching to silence"
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            return

        # パススルーで起動したが出力が受け付けず即死する場合に備え、
        # 次回は変換を有効化して再試行する。
        if cfg.passthrough and not conversion:
            # プロセスが短時間で落ちた場合だけ conversion に切り替える
            conversion = True

        time.sleep(cfg.restart_backoff_sec)


def _run_with_gi(cfg: UsbI2sBridgeConfig) -> None:
    """GI+GStreamer でフェード付きの再起動を行う本実装."""
    GLib, Gst = _try_import_gi_gst()
    if GLib is None or Gst is None:
        _run_with_gst_launch(cfg)
        return

    Gst.init(None)

    class Supervisor:
        def __init__(self, cfg: UsbI2sBridgeConfig) -> None:
            self.cfg = cfg
            self.loop = GLib.MainLoop()
            self.pipeline = None
            self.bus = None
            self.volume = None
            self.current_mode: str = "none"
            self.current_rate: int = cfg.fallback_rate
            self.current_fmt: str = cfg.preferred_format
            self.current_capture_fmt: Optional[str] = None
            self.conversion_enabled: bool = False
            self._restart_scheduled = False

        def _pipeline_str(
            self, mode: str, rate: int, capture_fmt: Optional[str]
        ) -> tuple[str, str]:
            raw_format, status_alsa_format = _choose_raw_caps(
                self.cfg,
                mode=mode,
                sample_rate=rate,
                capture_alsa_format=capture_fmt,
                conversion=self.conversion_enabled,
            )
            caps = f"audio/x-raw,rate={rate},channels={self.cfg.channels},format={raw_format}"
            if mode == "capture":
                src = (
                    f"alsasrc device={self.cfg.capture_device} "
                    f"buffer-time={self.cfg.alsa_buffer_time_us} "
                    f"latency-time={self.cfg.alsa_latency_time_us} "
                    f"do-timestamp=true"
                )
            else:
                src = "audiotestsrc is-live=true wave=silence"
            # name=vol で後から操作可能にする
            convert = ""
            if self.conversion_enabled:
                convert = "audioresample quality=10 ! audioconvert ! "
            return (
                f"{src} ! queue max-size-time={self.cfg.queue_time_ns} "
                f"max-size-bytes=0 max-size-buffers=0 ! {convert}"
                f"{caps} ! volume name=vol volume=0.0 ! "
                f"alsasink device={self.cfg.playback_device} sync=true async=false"
            ), status_alsa_format

        def _set_pipeline(self, pipeline_str: str) -> None:
            if self.pipeline is not None:
                self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect("message", self._on_message)
            self.volume = self.pipeline.get_by_name("vol")

        def _on_message(self, bus, message) -> None:  # noqa: ANN001
            t = message.type
            if t == Gst.MessageType.ERROR:
                err, dbg = message.parse_error()
                print(f"[usb_i2s_bridge] ERROR: {err} debug={dbg}")
                # パススルーで落ちる（I2S側非対応など）場合は次回変換を有効化して再試行する
                if self.cfg.passthrough and not self.conversion_enabled:
                    print(
                        "[usb_i2s_bridge] enabling conversion fallback after ERROR "
                        "(passthrough mode likely unsupported by sink)"
                    )
                    self.conversion_enabled = True
                self._schedule_restart()
            elif t == Gst.MessageType.EOS:
                print("[usb_i2s_bridge] EOS received; restarting")
                self._schedule_restart()

        def _fade(self, start: float, end: float, duration_ms: int, then) -> None:  # noqa: ANN001
            if self.volume is None:
                then()
                return
            if duration_ms <= 0:
                self.volume.set_property("volume", end)
                then()
                return
            steps = max(1, min(50, duration_ms // 10))
            interval_ms = max(5, duration_ms // steps)
            delta = (end - start) / float(steps)
            state = {"i": 0, "v": start}
            self.volume.set_property("volume", start)

            def _tick() -> bool:
                state["i"] += 1
                state["v"] += delta
                self.volume.set_property("volume", float(state["v"]))
                if state["i"] >= steps:
                    self.volume.set_property("volume", end)
                    then()
                    return False
                return True

            GLib.timeout_add(interval_ms, _tick)

        def _start(self, mode: str, rate: int, capture_fmt: Optional[str]) -> None:
            pipeline_str, status_alsa_format = self._pipeline_str(
                mode, rate, capture_fmt
            )
            print(
                f"[usb_i2s_bridge] start mode={mode} rate={rate} "
                f"passthrough={self.cfg.passthrough} conversion={self.conversion_enabled} "
                f"capture_fmt={capture_fmt}\n{pipeline_str}"
            )
            if self.cfg.dry_run:
                self.loop.quit()
                return
            self._set_pipeline(pipeline_str)
            self.pipeline.set_state(Gst.State.PLAYING)
            self.current_mode = mode
            self.current_rate = rate
            self.current_capture_fmt = capture_fmt
            self.current_fmt = status_alsa_format
            if self.cfg.status_path:
                _persist_status(
                    self.cfg.status_path,
                    running=True,
                    mode=mode,
                    sample_rate=rate,
                    alsa_format=status_alsa_format,
                    channels=self.cfg.channels,
                )
            self._fade(0.0, 1.0, self.cfg.fade_ms, lambda: None)

        def _stop_then(self, then) -> None:  # noqa: ANN001
            if self.pipeline is None:
                then()
                return

            def _do_stop() -> None:
                try:
                    self.pipeline.set_state(Gst.State.NULL)
                finally:
                    then()

            self._fade(1.0, 0.0, self.cfg.fade_ms, _do_stop)

        def _schedule_restart(self) -> None:
            if self._restart_scheduled:
                return
            self._restart_scheduled = True

            def _restart() -> bool:
                self._restart_scheduled = False
                self._reconcile(force=True)
                return False

            GLib.timeout_add(int(self.cfg.restart_backoff_sec * 1000), _restart)

        def _reconcile(self, force: bool = False) -> None:
            rate, fmt = _probe_capture_params(self.cfg.capture_device)
            capture_available = rate is not None

            if not capture_available and not self.cfg.keep_silence_when_no_capture:
                # 無入力時は停止（ポリシー）
                if self.current_mode != "none":
                    self._stop_then(lambda: setattr(self, "current_mode", "none"))
                return

            desired_mode = "capture" if capture_available else "silence"
            desired_rate = rate if capture_available else self.current_rate
            if desired_rate is None:
                desired_rate = self.cfg.fallback_rate
            desired_capture_fmt = fmt if capture_available else None

            need_switch = (
                force
                or self.current_mode != desired_mode
                or (desired_mode == "capture" and desired_rate != self.current_rate)
                or (
                    desired_mode == "capture"
                    and desired_capture_fmt != self.current_capture_fmt
                )
            )
            if not need_switch and self.pipeline is not None:
                return

            def _do_start() -> None:
                self._start(desired_mode, int(desired_rate), desired_capture_fmt)

            self._stop_then(_do_start)

        def run(self) -> None:
            # 初期状態: まずサイレンスで I2S を立ち上げ、USB が来たら capture にスイッチ
            self.current_rate = self.cfg.fallback_rate
            self.current_fmt = self.cfg.preferred_format
            initial_mode = (
                "silence" if self.cfg.keep_silence_when_no_capture else "capture"
            )
            self._start(initial_mode, self.current_rate, None)

            def _poll() -> bool:
                try:
                    self._reconcile(force=False)
                except Exception as e:
                    print(f"[usb_i2s_bridge] reconcile exception: {e}; restarting")
                    self._schedule_restart()
                return True

            GLib.timeout_add(int(self.cfg.poll_interval_sec * 1000), _poll)
            self.loop.run()

    Supervisor(cfg).run()


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
        choices=sorted(set(_ALSA_TO_GST_FORMAT.keys())),
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
        "--passthrough",
        dest="passthrough",
        action="store_true",
        default=_env_bool("USB_I2S_PASSTHROUGH", _DEFAULT_PASSTHROUGH),
        help="UAC2で受けた rate/format をそのまま I2S へ流す (default: true)",
    )
    parser.add_argument(
        "--no-passthrough",
        dest="passthrough",
        action="store_false",
        help="I2S側の安定性優先で preferred_format へ変換して出力する",
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
        passthrough=bool(args.passthrough),
        status_path=args.status_path,
        dry_run=bool(args.dry_run),
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    cfg.validate()

    # NOTE: GI が無い場合でも最低限動くようフォールバック実装を用意
    _run_with_gi(cfg)


if __name__ == "__main__":
    main()
