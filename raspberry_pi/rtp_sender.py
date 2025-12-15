"""GStreamer RTP sender for Raspberry Pi.

環境変数・CLI引数から gst-launch-1.0 の引数を組み立て、
RTCP付きのRTPパイプラインを起動するシンプルなランナーです。
TCPベースのC++実装を置き換える常用パスとして利用します。
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
from typing import Iterable, List, Tuple

_DEFAULT_DEVICE = "hw:2,0"
_DEFAULT_HOST = "192.168.55.1"
_DEFAULT_RTP_PORT = 46000
_DEFAULT_RTCP_PORT = 46001
_DEFAULT_RTCP_LISTEN_PORT = 46002
_DEFAULT_PAYLOAD_TYPE = 96
_DEFAULT_SAMPLE_RATE = 44100
_DEFAULT_CHANNELS = 2
_DEFAULT_FORMAT = "S24_3BE"
_DEFAULT_LATENCY_MS = 100
_DEFAULT_STATS_PATH = Path("/tmp/rtp_receiver_stats.json")
_DEFAULT_RATE_POLL_INTERVAL_SEC = 2.0

# 送信側パイプライン安定化のデフォルト値
# - alsasrc はデフォルト値依存だと period/avail がギリギリになりやすいため明示する
# - queue はオーディオ変換/リサンプルの遅延を吸収する
_DEFAULT_ALSA_BUFFER_TIME_US = 200_000  # 200ms
_DEFAULT_ALSA_LATENCY_TIME_US = 20_000  # 20ms
_DEFAULT_QUEUE_TIME_NS = 100_000_000  # 100ms
_DEFAULT_UDP_BUFFER_SIZE_BYTES = 1_048_576  # 1MiB (SO_SNDBUF)

_FORMAT_MAP: dict[str, Tuple[str, str, str]] = {
    # Little-endian variants (互換維持)
    "S16_LE": ("rtpL16pay", "L16", "S16LE"),
    "S24_3LE": ("rtpL24pay", "L24", "S24LE"),
    "S32_LE": ("rtpL32pay", "L32", "S32LE"),
    # Big-endian variants (GStreamer RTP の推奨パス)
    "S16_BE": ("rtpL16pay", "L16", "S16BE"),
    "S24_3BE": ("rtpL24pay", "L24", "S24BE"),
    "S32_BE": ("rtpL32pay", "L32", "S32BE"),
}


@dataclass
class RtpSenderConfig:
    """RTP送信パイプラインの設定."""

    device: str = _DEFAULT_DEVICE
    host: str = _DEFAULT_HOST
    rtp_port: int = _DEFAULT_RTP_PORT
    rtcp_port: int = _DEFAULT_RTCP_PORT
    rtcp_listen_port: int = _DEFAULT_RTCP_LISTEN_PORT
    payload_type: int = _DEFAULT_PAYLOAD_TYPE
    sample_rate: int = _DEFAULT_SAMPLE_RATE
    channels: int = _DEFAULT_CHANNELS
    audio_format: str = _DEFAULT_FORMAT
    latency_ms: int | None = _DEFAULT_LATENCY_MS
    alsa_buffer_time_us: int = _DEFAULT_ALSA_BUFFER_TIME_US
    alsa_latency_time_us: int = _DEFAULT_ALSA_LATENCY_TIME_US
    queue_time_ns: int = _DEFAULT_QUEUE_TIME_NS
    udp_buffer_size_bytes: int = _DEFAULT_UDP_BUFFER_SIZE_BYTES
    auto_sample_rate: bool = True
    stats_path: Path | None = _DEFAULT_STATS_PATH
    rate_poll_interval_sec: float = _DEFAULT_RATE_POLL_INTERVAL_SEC
    dry_run: bool = False

    def validate(self) -> None:
        if self.audio_format not in _FORMAT_MAP:
            raise ValueError(f"Unsupported audio format: {self.audio_format}")
        for name, port in (
            ("rtp_port", self.rtp_port),
            ("rtcp_port", self.rtcp_port),
            ("rtcp_listen_port", self.rtcp_listen_port),
        ):
            if port <= 0 or port > 65535:
                raise ValueError(f"Invalid {name}: {port}")
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"Invalid channels: {self.channels}")
        if self.payload_type <= 0 or self.payload_type > 127:
            raise ValueError(f"Invalid payload_type: {self.payload_type}")
        if self.alsa_buffer_time_us <= 0:
            raise ValueError(f"Invalid alsa_buffer_time_us: {self.alsa_buffer_time_us}")
        if self.alsa_latency_time_us <= 0:
            raise ValueError(
                f"Invalid alsa_latency_time_us: {self.alsa_latency_time_us}"
            )
        if self.queue_time_ns <= 0:
            raise ValueError(f"Invalid queue_time_ns: {self.queue_time_ns}")
        if self.udp_buffer_size_bytes <= 0:
            raise ValueError(
                f"Invalid udp_buffer_size_bytes: {self.udp_buffer_size_bytes}"
            )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path | None) -> Path | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return None
    return Path(raw)


def _hw_params_path(device: str) -> Path | None:
    """デバイス文字列 (hw:2,0 / plughw:2,0) から hw_params パスを組み立て."""
    match = re.match(r"^(?:plughw:|hw:)(?P<card>\\d+),(?P<pcm>\\d+)", device)
    if not match:
        return None
    card = match.group("card")
    pcm = match.group("pcm")
    return Path(f"/proc/asound/card{card}/pcm{pcm}c/sub0/hw_params")


def _parse_hw_params_rate(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("rate:"):
            for token in line.split():
                if token.isdigit():
                    return int(token)
    return None


def _probe_hw_params_rate(device: str) -> int | None:
    path = _hw_params_path(device)
    if path is None or not path.exists():
        return None
    try:
        text = path.read_text()
    except OSError:
        return None
    if "closed" in text:
        return None
    return _parse_hw_params_rate(text)


def _parse_arecord_rate(stdout: str) -> int | None:
    for line in stdout.splitlines():
        if line.strip().startswith("RATE"):
            numbers = [int(token) for token in line.split() if token.isdigit()]
            if numbers:
                return numbers[-1]
    return None


def _probe_arecord_rate(device: str, channels: int) -> int | None:
    """alsa-utils がある場合に arecord --dump-hw-params からレートを推測."""
    cmd = [
        "arecord",
        "-D",
        device,
        "-c",
        str(channels),
        "-f",
        "S16_LE",
        "-d",
        "0",
        "--dump-hw-params",
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None
    return _parse_arecord_rate(result.stdout)


def _detect_sample_rate(device: str, channels: int, fallback: int) -> int:
    for probe in (
        lambda: _probe_hw_params_rate(device),
        lambda: _probe_arecord_rate(device, channels),
    ):
        rate = probe()
        if rate:
            return rate
    return fallback


def _persist_stats(path: Path, sample_rate: int, latency_ms: int | None) -> None:
    """ZeroMQ ブリッジ用にサンプルレート等をJSONで書き出す."""
    payload = {
        "running": True,
        "sample_rate": sample_rate,
        "latency_ms": latency_ms if latency_ms is not None else 0,
        "packets_received": 0,
        "packets_lost": 0,
        "jitter_ms": 0.0,
        "clock_drift_ppm": 0.0,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError:
        # ファイル書き込みに失敗しても送出自体は継続する
        pass


def build_gst_command(cfg: RtpSenderConfig) -> List[str]:
    """設定から gst-launch-1.0 の引数配列を生成する."""
    cfg.validate()
    payloader, encoding, raw_format = _FORMAT_MAP[cfg.audio_format]

    rtp_caps = (
        f"application/x-rtp,media=audio,clock-rate={cfg.sample_rate},"
        f"encoding-name={encoding},payload={cfg.payload_type},channels={cfg.channels}"
    )
    raw_caps = f"audio/x-raw,rate={cfg.sample_rate},channels={cfg.channels},format={raw_format}"

    args: List[str] = [
        "gst-launch-1.0",
        "-e",
        "rtpbin",
        "name=rtpbin",
        "ntp-sync=true",
        # Align with README recommended pipeline (stabilize clock sync)
        "buffer-mode=sync",
    ]
    if cfg.latency_ms is not None:
        args.append(f"latency={cfg.latency_ms}")

    tail: List[str] = [
        "alsasrc",
        f"device={cfg.device}",
        # alsasrc のデフォルト依存だと period/avail がギリギリになりやすいので明示
        f"buffer-time={cfg.alsa_buffer_time_us}",
        f"latency-time={cfg.alsa_latency_time_us}",
        # ライブソースとしてタイムスタンプを付与（送出側の同期を安定化）
        "do-timestamp=true",
        "!",
        "audioresample",
        "quality=10",
        "!",
        "audioconvert",
        "!",
        "queue",
        f"max-size-time={cfg.queue_time_ns}",
        "max-size-bytes=0",
        "max-size-buffers=0",
        "!",
        raw_caps,
        "!",
        payloader,
        f"pt={cfg.payload_type}",
        "!",
        rtp_caps,
        "!",
        "rtpbin.send_rtp_sink_0",
        "rtpbin.send_rtp_src_0",
        "!",
        "udpsink",
        f"host={cfg.host}",
        f"port={cfg.rtp_port}",
        f"buffer-size={cfg.udp_buffer_size_bytes}",
        "sync=true",
        "async=false",
        "rtpbin.send_rtcp_src_0",
        "!",
        "udpsink",
        f"host={cfg.host}",
        f"port={cfg.rtcp_port}",
        f"buffer-size={cfg.udp_buffer_size_bytes}",
        "sync=false",
        "async=false",
        "udpsrc",
        f"port={cfg.rtcp_listen_port}",
        "!",
        "rtpbin.recv_rtcp_sink_0",
    ]
    args.extend(tail)
    return args


def command_to_string(args: Iterable[str]) -> str:
    """配列をスペース区切りの文字列に整形（テスト・ログ用）."""
    return " ".join(args)


def _parse_args(argv: list[str] | None = None) -> RtpSenderConfig:
    parser = argparse.ArgumentParser(description="Raspberry Pi GStreamer RTP sender")
    parser.add_argument(
        "--device", default=_env_str("RTP_SENDER_DEVICE", _DEFAULT_DEVICE)
    )
    parser.add_argument("--host", default=_env_str("RTP_SENDER_HOST", _DEFAULT_HOST))
    parser.add_argument(
        "--rtp-port",
        type=int,
        default=_env_int("RTP_SENDER_RTP_PORT", _DEFAULT_RTP_PORT),
    )
    parser.add_argument(
        "--rtcp-port",
        type=int,
        default=_env_int("RTP_SENDER_RTCP_PORT", _DEFAULT_RTCP_PORT),
    )
    parser.add_argument(
        "--rtcp-listen-port",
        type=int,
        default=_env_int("RTP_SENDER_RTCP_LISTEN_PORT", _DEFAULT_RTCP_LISTEN_PORT),
    )
    parser.add_argument(
        "--payload-type",
        type=int,
        default=_env_int("RTP_SENDER_PAYLOAD_TYPE", _DEFAULT_PAYLOAD_TYPE),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=_env_int("RTP_SENDER_SAMPLE_RATE", _DEFAULT_SAMPLE_RATE),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=_env_int("RTP_SENDER_CHANNELS", _DEFAULT_CHANNELS),
    )
    parser.add_argument(
        "--format",
        dest="audio_format",
        default=_env_str("RTP_SENDER_FORMAT", _DEFAULT_FORMAT),
        choices=sorted(_FORMAT_MAP.keys()),
    )
    parser.add_argument(
        "--latency-ms",
        type=int,
        default=_env_int("RTP_SENDER_LATENCY_MS", _DEFAULT_LATENCY_MS),
        help="RTP jitterbuffer latency (ms)",
    )
    parser.add_argument(
        "--alsa-buffer-time-us",
        type=int,
        default=_env_int(
            "RTP_SENDER_ALSA_BUFFER_TIME_US", _DEFAULT_ALSA_BUFFER_TIME_US
        ),
        help="alsasrc buffer-time (microseconds)",
    )
    parser.add_argument(
        "--alsa-latency-time-us",
        type=int,
        default=_env_int(
            "RTP_SENDER_ALSA_LATENCY_TIME_US", _DEFAULT_ALSA_LATENCY_TIME_US
        ),
        help="alsasrc latency-time (microseconds)",
    )
    parser.add_argument(
        "--queue-time-ns",
        type=int,
        default=_env_int("RTP_SENDER_QUEUE_TIME_NS", _DEFAULT_QUEUE_TIME_NS),
        help="queue max-size-time (nanoseconds)",
    )
    parser.add_argument(
        "--udp-buffer-size-bytes",
        type=int,
        default=_env_int(
            "RTP_SENDER_UDP_BUFFER_SIZE_BYTES", _DEFAULT_UDP_BUFFER_SIZE_BYTES
        ),
        help="udpsink SO_SNDBUF (bytes)",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=_env_path("RTP_BRIDGE_STATS_PATH", _DEFAULT_STATS_PATH),
        help="Write RTP stats JSON for ZeroMQ bridge (optional)",
    )
    parser.add_argument(
        "--auto-sample-rate",
        dest="auto_sample_rate",
        action="store_true",
        default=_env_bool("RTP_SENDER_AUTO_SAMPLE_RATE", True),
        help="Detect ALSA capture rate automatically (default: enabled)",
    )
    parser.add_argument(
        "--no-auto-sample-rate",
        dest="auto_sample_rate",
        action="store_false",
        help="Disable automatic sample rate detection",
    )
    parser.add_argument(
        "--rate-poll-interval",
        dest="rate_poll_interval_sec",
        type=float,
        default=_env_float(
            "RTP_SENDER_RATE_POLL_INTERVAL_SEC", _DEFAULT_RATE_POLL_INTERVAL_SEC
        ),
        help="Polling interval (seconds) to detect rate changes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.getenv("RTP_SENDER_DRY_RUN", "false").lower()
        in {"1", "true", "yes"},
        help="Print pipeline and exit",
    )

    args = parser.parse_args(argv)
    latency_ms: int | None = args.latency_ms
    if latency_ms is not None and latency_ms <= 0:
        latency_ms = None

    return RtpSenderConfig(
        device=args.device,
        host=args.host,
        rtp_port=args.rtp_port,
        rtcp_port=args.rtcp_port,
        rtcp_listen_port=args.rtcp_listen_port,
        payload_type=args.payload_type,
        sample_rate=args.sample_rate,
        channels=args.channels,
        audio_format=args.audio_format,
        latency_ms=latency_ms,
        alsa_buffer_time_us=max(1, int(args.alsa_buffer_time_us)),
        alsa_latency_time_us=max(1, int(args.alsa_latency_time_us)),
        queue_time_ns=max(1, int(args.queue_time_ns)),
        udp_buffer_size_bytes=max(1, int(args.udp_buffer_size_bytes)),
        auto_sample_rate=args.auto_sample_rate,
        stats_path=args.stats_path,
        rate_poll_interval_sec=max(0.5, args.rate_poll_interval_sec),
        dry_run=args.dry_run,
    )


def _launch_pipeline(cfg: RtpSenderConfig) -> subprocess.Popen:
    args = build_gst_command(cfg)
    cmd_str = command_to_string(args)
    print(f"[rtp_sender] launching pipeline (rate={cfg.sample_rate} Hz):\n{cmd_str}")
    return subprocess.Popen(args)


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    if cfg.auto_sample_rate:
        detected_rate = _detect_sample_rate(cfg.device, cfg.channels, cfg.sample_rate)
        if detected_rate != cfg.sample_rate:
            print(
                f"[rtp_sender] detected sample rate {detected_rate} Hz "
                f"(requested {cfg.sample_rate} Hz)"
            )
            cfg.sample_rate = detected_rate
        else:
            print(f"[rtp_sender] sample rate {cfg.sample_rate} Hz (auto-detected)")

    if cfg.stats_path:
        _persist_stats(cfg.stats_path, cfg.sample_rate, cfg.latency_ms)

    if cfg.dry_run:
        args = build_gst_command(cfg)
        print(command_to_string(args))
        return

    # 自動追従が無効なら従来通り1回だけ実行
    if not cfg.auto_sample_rate:
        proc = _launch_pipeline(cfg)
        rc = proc.wait()
        if rc != 0:
            raise SystemExit(rc)
        return

    # 自動追従: レート変化を検知したらパイプラインを再起動
    current_rate = cfg.sample_rate
    while True:
        proc = _launch_pipeline(cfg)
        try:
            while True:
                try:
                    rc = proc.wait(timeout=cfg.rate_poll_interval_sec)
                    if rc != 0:
                        raise SystemExit(rc)
                    return
                except subprocess.TimeoutExpired:
                    new_rate = _detect_sample_rate(
                        cfg.device, cfg.channels, current_rate
                    )
                    if new_rate != current_rate:
                        print(
                            "[rtp_sender] detected rate change: "
                            f"{current_rate} -> {new_rate} Hz (restarting)"
                        )
                        current_rate = new_rate
                        cfg.sample_rate = new_rate
                        if cfg.stats_path:
                            _persist_stats(
                                cfg.stats_path, cfg.sample_rate, cfg.latency_ms
                            )
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break  # restart with new rate
                    time.sleep(cfg.rate_poll_interval_sec)
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            return


if __name__ == "__main__":
    main()
