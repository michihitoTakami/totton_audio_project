"""ZeroMQ bridge launcher for RTP receiver (Raspberry Pi side).

主プロセスやGStreamerパイプラインと疎結合に連携できるよう、
以下の仕組みを提供する:
- 定期的に JSON ファイルから統計を読み出し、STATUS に反映
- SET_LATENCY を受け取った際にファイルへ書き出し、外部プロセスが適用できるようにする

既存の C++ 送出アプリに直接リンクせず、軽量なサイドカーとして動作することを想定。
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

from .zmq_bridge import (
    DEFAULT_ENDPOINT,
    DEFAULT_TIMEOUT_MS,
    RtpStatsStore,
    RtpZmqBridge,
)

DEFAULT_STATS_PATH = Path("/tmp/rtp_receiver_stats.json")
DEFAULT_LATENCY_PATH = Path("/tmp/rtp_receiver_latency_ms")
POLL_INTERVAL_SEC = 1.0


def _load_stats(stats_path: Path, store: RtpStatsStore) -> None:
    if not stats_path.exists():
        return
    try:
        data = json.loads(stats_path.read_text())
    except (ValueError, OSError):
        return

    store.update(
        packets_received=data.get("packets_received"),
        packets_lost=data.get("packets_lost"),
        jitter_ms=data.get("jitter_ms"),
        clock_drift_ppm=data.get("clock_drift_ppm"),
        sample_rate=data.get("sample_rate"),
        latency_ms=data.get("latency_ms"),
        running=data.get("running"),
    )


def _persist_latency(latency_ms: int, latency_path: Path) -> None:
    try:
        latency_path.write_text(str(latency_ms))
    except OSError:
        # 失敗してもブリッジ自体は動き続ける
        pass


def run_bridge(
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    stats_path: Path = DEFAULT_STATS_PATH,
    latency_path: Path = DEFAULT_LATENCY_PATH,
    poll_interval_sec: float = POLL_INTERVAL_SEC,
) -> None:
    """ZeroMQ ブリッジを起動し、ファイル経由で統計/レイテンシを連携する."""
    stats_store = RtpStatsStore()

    stop = threading.Event()

    def _poll_stats() -> None:
        while not stop.is_set():
            _load_stats(stats_path, stats_store)
            stop.wait(poll_interval_sec)

    poll_thread = threading.Thread(
        target=_poll_stats, name="rtp_stats_poll", daemon=True
    )
    poll_thread.start()

    def _on_latency(latency: int) -> None:
        _persist_latency(latency, latency_path)

    with RtpZmqBridge(
        stats_store,
        _on_latency,
        endpoint=endpoint,
        timeout_ms=timeout_ms,
    ):
        try:
            while not stop.wait(1.0):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            poll_thread.join(timeout=timeout_ms / 1000 + 1)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTP receiver ZeroMQ bridge (sidecar)")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("RTP_BRIDGE_ENDPOINT", DEFAULT_ENDPOINT),
        help=f"ZeroMQ REP endpoint (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=int(os.getenv("RTP_BRIDGE_TIMEOUT_MS", DEFAULT_TIMEOUT_MS)),
        help="ZeroMQ send/recv timeout in milliseconds",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path(os.getenv("RTP_BRIDGE_STATS_PATH", DEFAULT_STATS_PATH)),
        help="Path to JSON stats file to watch",
    )
    parser.add_argument(
        "--latency-path",
        type=Path,
        default=Path(os.getenv("RTP_BRIDGE_LATENCY_PATH", DEFAULT_LATENCY_PATH)),
        help="Path to write latency_ms when SET_LATENCY is received",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.getenv("RTP_BRIDGE_POLL_INTERVAL_SEC", POLL_INTERVAL_SEC)),
        help="Polling interval (seconds) for stats file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_bridge(
        endpoint=args.endpoint,
        timeout_ms=args.timeout_ms,
        stats_path=args.stats_path,
        latency_path=args.latency_path,
        poll_interval_sec=args.poll_interval,
    )


if __name__ == "__main__":
    main()
