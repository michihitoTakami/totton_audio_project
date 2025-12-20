#!/usr/bin/env python3
"""Output buffer monitoring tool."""

import json
import sys
import time
from pathlib import Path

STATS_FILE = Path("/tmp/gpu_upsampler_stats.json")


def read_buffer_stats():
    try:
        with STATS_FILE.open() as f:
            stats = json.load(f)
            return stats.get("output_buffer", {})
    except Exception as exc:
        return {"error": str(exc)}


def main():
    print("Output Buffer Monitor (Ctrl+C to exit)\n")

    try:
        while True:
            stats = read_buffer_stats()

            if "error" in stats:
                print(f"\rError: {stats['error']}", end="", flush=True)
            else:
                frames = stats.get("output_buffer_frames", 0)
                seconds = stats.get("output_buffer_seconds", 0)
                usage = stats.get("output_buffer_usage_percent", 0)
                drops = stats.get("buffer_drops_total", 0)
                ready = stats.get("output_ready", False)

                status = "READY" if ready else "DISCONNECTED"

                print(
                    f"\r[{status}] Buffer: {frames:>8} frames ({seconds:>6.2f}s) | "
                    f"Usage: {usage:>5.1f}% | Drops: {drops:>8}",
                    end="",
                    flush=True,
                )

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
