#!/bin/bash
set -e

BIN="/usr/local/bin/jetson-pcm-receiver"

log() {
    echo "[entrypoint] $*"
}

if [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    exec "$@"
fi

if [ ! -e "$BIN" ]; then
    echo "[entrypoint] binary not found: $BIN" >&2
    exit 1
fi

if [ ! -d /dev/snd ]; then
    log "/dev/snd がマウントされていません。--device /dev/snd を付与してください。"
fi

# Respect environment variable overrides (parsed inside the binary)
exec "$BIN" "$@"
