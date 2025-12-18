"""Jetson/Pi 双方で使える I2S 制御プレーンエージェント."""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from .control_plane import ControlPlaneSync, _is_supported_rate

logger = logging.getLogger(__name__)

#
# NOTE(#950):
# このエージェントは ZeroMQ 制御プレーン用の補助ツールだが、
# Jetson 側で常駐させる前提が崩れると「同期できない→Pi側が不安定」になり得るため、
# デフォルトは無効（空）に寄せる。必要な場合のみ明示的に指定する。
#
DEFAULT_ENDPOINT = os.getenv("I2S_CONTROL_ENDPOINT", "").strip()
DEFAULT_PEER = os.getenv("I2S_CONTROL_PEER", "").strip()
DEFAULT_DEVICE = os.getenv("I2S_CONTROL_DEVICE", "hw:Loopback,0,0")
DEFAULT_STREAM = os.getenv("I2S_CONTROL_STREAM", "c")  # c: capture, p: playback
DEFAULT_CHANNELS = int(os.getenv("I2S_CONTROL_CHANNELS", "2"))
DEFAULT_FORMAT = os.getenv("I2S_CONTROL_DEFAULT_FORMAT", "S32_LE")
DEFAULT_RATE = int(os.getenv("I2S_CONTROL_DEFAULT_RATE", "48000"))
DEFAULT_POLL_SEC = float(os.getenv("I2S_CONTROL_POLL_INTERVAL_SEC", "1.0"))
DEFAULT_TIMEOUT_MS = int(os.getenv("I2S_CONTROL_TIMEOUT_MS", "2000"))
DEFAULT_REQUIRE_PEER = os.getenv("I2S_CONTROL_REQUIRE_PEER", "0").lower() not in {
    "0",
    "false",
    "no",
}


def _hw_params_path(device: str, stream: str) -> Optional[Path]:
    match = re.match(r"^(?:plughw:|hw:)(?P<card>\\d+),(?P<pcm>\\d+)", device)
    if not match:
        return None
    card = match.group("card")
    pcm = match.group("pcm")
    suffix = "c" if stream == "c" else "p"
    return Path(f"/proc/asound/card{card}/pcm{pcm}{suffix}/sub0/hw_params")


def _parse_hw_params(path: Path) -> Tuple[Optional[int], Optional[str]]:
    if not path.exists():
        return None, None
    try:
        text = path.read_text()
    except OSError:
        return None, None
    if "closed" in text:
        return None, None
    rate = None
    fmt = None
    for line in text.splitlines():
        if line.startswith("rate:"):
            for token in line.split():
                if token.isdigit():
                    rate = int(token)
                    break
        elif line.startswith("format:"):
            parts = line.split()
            if len(parts) >= 2:
                fmt = parts[1].strip()
    return rate, fmt


def _device_present(device: str, stream: str) -> bool:
    match = re.match(r"^(?:plughw:|hw:)(?P<card>\\d+),(?P<pcm>\\d+)", device)
    if not match:
        return False
    card = match.group("card")
    pcm = match.group("pcm")
    suffix = "c" if stream == "c" else "p"
    node = Path(f"/dev/snd/pcmC{card}D{pcm}{suffix}")
    return node.exists()


def _env_or_default_rate(rate: int) -> int:
    return rate if _is_supported_rate(rate) else DEFAULT_RATE


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="I2S control-plane agent")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--peer", default=DEFAULT_PEER)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--stream", default=DEFAULT_STREAM, choices=["c", "p"])
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--default-rate", type=int, default=DEFAULT_RATE)
    parser.add_argument("--default-format", default=DEFAULT_FORMAT)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_SEC)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument(
        "--allow-without-peer",
        action="store_true",
        default=not DEFAULT_REQUIRE_PEER,
        help="peer 不達でも capture を許可する（デバッグ用）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return parser.parse_args(argv)


def _build_status(
    *,
    device: str,
    stream: str,
    channels: int,
    default_rate: int,
    default_format: str,
) -> tuple[bool, str, int, str, int]:
    present = _device_present(device, stream)
    rate, fmt = (None, None)
    if present:
        path = _hw_params_path(device, stream)
        if path:
            rate, fmt = _parse_hw_params(path)
    sample_rate = _env_or_default_rate(rate or default_rate)
    fmt = fmt or default_format
    mode = "capture" if present else "none"
    return present, mode, sample_rate, fmt, channels


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    endpoint = str(args.endpoint or "").strip()
    if not endpoint:
        logger.error(
            "I2S control agent: endpoint が未指定です。"
            " --endpoint もしくは I2S_CONTROL_ENDPOINT を指定してください。"
        )
        return 2

    sync = ControlPlaneSync(
        endpoint=endpoint,
        peer_endpoint=args.peer or None,
        require_peer=not args.allow_without_peer,
        poll_interval_sec=args.poll_interval,
        timeout_ms=args.timeout_ms,
    )
    sync.start()

    stop = False

    def _handle_signal(signum, frame):  # noqa: ANN001
        nonlocal stop
        _ = signum, frame
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "I2S control agent start endpoint=%s peer=%s device=%s stream=%s require_peer=%s",
        args.endpoint,
        args.peer,
        args.device,
        args.stream,
        not args.allow_without_peer,
    )

    try:
        while not stop:
            running, mode, sample_rate, fmt, channels = _build_status(
                device=args.device,
                stream=args.stream,
                channels=max(1, int(args.channels)),
                default_rate=args.default_rate,
                default_format=args.default_format,
            )
            if not _is_supported_rate(sample_rate):
                logger.warning("Unsupported rate detected: %s", sample_rate)
            sync.update_local(
                running=running,
                mode=mode,
                sample_rate=sample_rate,
                fmt=fmt,
                channels=max(1, int(channels)),
            )
            if args.verbose:
                peer = sync.peer_status()
                logger.debug(
                    "local mode=%s rate=%d fmt=%s ch=%d synced=%s peer=%s",
                    mode,
                    sample_rate,
                    fmt,
                    channels,
                    sync.is_synced(),
                    peer.to_dict() if peer else None,
                )
            time.sleep(args.poll_interval)
    finally:
        sync.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
