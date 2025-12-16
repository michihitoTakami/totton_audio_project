#!/usr/bin/env python3
"""
Realtime monitor for /tmp/gpu_upsampler_stats.json.

Terminates with non-zero exit if:
 - clip_count increases compared to start
 - post-gain peak exceeds configured threshold

Intended for long duration stress tests while audio is playing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


def load_stats(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor daemon stats for clipping")
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("/tmp/gpu_upsampler_stats.json"),
        help="Path to stats JSON file",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1 second)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration in seconds. If omitted, runs until interrupted.",
    )
    parser.add_argument(
        "--headroom-threshold",
        type=float,
        default=0.995,
        help="Fail if post-gain linear peak exceeds this value (default: 0.995)",
    )
    parser.add_argument(
        "--allow-existing-clips",
        action="store_true",
        help="Ignore clip_count delta at startup (useful if daemon already accumulated clips)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    baseline_clip = None

    try:
        while True:
            if args.duration and (time.time() - start_time) >= args.duration:
                print("[monitor] duration reached without new clipping events")
                raise SystemExit(0)

            if not args.stats_path.exists():
                print(f"[monitor] waiting for {args.stats_path} ...")
                time.sleep(args.interval)
                continue

            try:
                stats = load_stats(args.stats_path)
            except json.JSONDecodeError:
                print("[monitor] stats file is being updated, retrying...")
                time.sleep(args.interval)
                continue

            clip_count = stats.get("clip_count", 0)
            clip_rate = stats.get("clip_rate", 0.0)
            peaks = stats.get("peaks", {})
            post_gain = peaks.get("post_gain", {})
            post_gain_linear = post_gain.get("linear", 0.0)
            post_gain_db = post_gain.get("dbfs", -200.0)

            if baseline_clip is None:
                baseline_clip = clip_count if args.allow_existing_clips else 0

            print(
                f"[monitor] clip_count={clip_count} clip_rate={clip_rate:.6%} "
                f"post_gain={post_gain_db:.2f} dBFS ({post_gain_linear:.6f})"
            )

            if clip_count > baseline_clip:
                print("[monitor] ERROR: clip count increased during observation window")
                raise SystemExit(1)

            if post_gain_linear >= args.headroom_threshold:
                print(
                    "[monitor] ERROR: post-gain peak exceeded threshold "
                    f"{args.headroom_threshold:.6f}"
                )
                raise SystemExit(1)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[monitor] interrupted by user")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
