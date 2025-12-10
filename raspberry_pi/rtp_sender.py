"""GStreamer RTP sender for Raspberry Pi.

環境変数・CLI引数から gst-launch-1.0 の引数を組み立て、
RTCP付きのRTPパイプラインを起動するシンプルなランナーです。
TCPベースのC++実装を置き換える常用パスとして利用します。
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Tuple

_DEFAULT_DEVICE = "hw:0,0"
_DEFAULT_HOST = "jetson"
_DEFAULT_RTP_PORT = 46000
_DEFAULT_RTCP_PORT = 46001
_DEFAULT_RTCP_LISTEN_PORT = 46002
_DEFAULT_PAYLOAD_TYPE = 96
_DEFAULT_SAMPLE_RATE = 44100
_DEFAULT_CHANNELS = 2
_DEFAULT_FORMAT = "S24_3LE"
_DEFAULT_LATENCY_MS = 100

_FORMAT_MAP: dict[str, Tuple[str, str, str]] = {
    "S16_LE": ("rtpL16pay", "L16", "S16LE"),
    "S24_3LE": ("rtpL24pay", "L24", "S24LE"),
    "S32_LE": ("rtpL32pay", "L32", "S32LE"),
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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


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
        "buffer-mode=sync",
    ]
    if cfg.latency_ms is not None:
        args.append(f"latency={cfg.latency_ms}")

    tail: List[str] = [
        "alsasrc",
        f"device={cfg.device}",
        "!",
        "audioresample",
        "quality=10",
        "!",
        "audioconvert",
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
        "sync=true",
        "async=false",
        "rtpbin.send_rtcp_src_0",
        "!",
        "udpsink",
        f"host={cfg.host}",
        f"port={cfg.rtcp_port}",
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
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    args = build_gst_command(cfg)
    cmd_str = command_to_string(args)

    if cfg.dry_run:
        print(cmd_str)
        return

    print(f"[rtp_sender] launching pipeline:\n{cmd_str}")
    # Run gst-launch and forward exit code
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
