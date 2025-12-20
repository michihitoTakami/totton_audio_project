"""Raspberry Pi side control API for USB-I2S bridge."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from raspberry_pi.usb_i2s_bridge.bridge import UsbI2sBridgeConfig

DEFAULT_BIND_INTERFACE = os.getenv("RPI_CONTROL_BIND_INTERFACE", "usb0").strip()
DEFAULT_BIND_HOST = os.getenv("RPI_CONTROL_BIND_HOST", "").strip()
DEFAULT_BIND_SUBNET = os.getenv("RPI_CONTROL_BIND_SUBNET", "192.168.55.0/24").strip()
DEFAULT_BIND_WAIT_SECS = float(
    os.getenv("RPI_CONTROL_BIND_WAIT_SECS", "30").strip() or "30"
)
DEFAULT_BIND_RETRY_SECS = float(
    os.getenv("RPI_CONTROL_BIND_RETRY_SECS", "1").strip() or "1"
)
DEFAULT_PORT = int(os.getenv("RPI_CONTROL_PORT", "8081"))
DEFAULT_STATUS_PATH = Path(
    os.getenv("RPI_CONTROL_STATUS_PATH", "/var/run/usb-i2s-bridge/status.json")
)

DEFAULT_CONFIG_PATH = Path(
    os.getenv("RPI_CONTROL_CONFIG_PATH", "/var/lib/usb-i2s-bridge/config.env")
)
DEFAULT_RESTART_MODE = os.getenv("RPI_CONTROL_RESTART_MODE", "docker").strip()
DEFAULT_RESTART_CMD = os.getenv("RPI_CONTROL_RESTART_CMD", "").strip()
DEFAULT_DOCKER_CONTAINER = os.getenv(
    "RPI_CONTROL_DOCKER_CONTAINER", "rpi-usb-i2s-bridge"
).strip()


def _resolve_interface_ip(interface: str) -> Optional[str]:
    if not interface:
        return None
    try:
        import fcntl
        import socket
        import struct
    except Exception:
        return None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        request = struct.pack("256s", interface.encode("utf-8")[:15])
        res = fcntl.ioctl(sock.fileno(), 0x8915, request)  # SIOCGIFADDR
        return socket.inet_ntoa(res[20:24])
    except Exception:
        return None
    finally:
        sock.close()


def _iter_interface_names() -> list[str]:
    """Best-effort interface enumeration (Linux)."""
    try:
        import socket

        return [name for _, name in socket.if_nameindex()]
    except Exception:
        return []


def _resolve_any_interface_in_subnet(
    subnet_cidr: str,
    *,
    interface_names: Optional[list[str]] = None,
    resolver: Callable[[str], Optional[str]] = _resolve_interface_ip,
) -> Optional[str]:
    """Find first interface IP that belongs to the given subnet."""
    try:
        target_net = ip_network(subnet_cidr, strict=False)
    except ValueError:
        return None

    names = interface_names if interface_names is not None else _iter_interface_names()
    for name in names:
        if not name or name == "lo":
            continue
        ip = resolver(name)
        if not ip:
            continue
        try:
            if ip_address(ip) in target_net:
                return ip
        except ValueError:
            continue
    return None


def _is_bindable(host: str, port: int) -> bool:
    """Return True if (host, port) can be bound right now (best-effort)."""
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
        finally:
            sock.close()
    except OSError:
        return False


def _resolve_bind_host() -> str:
    """Resolve bind host from explicit host / interface / subnet."""
    if DEFAULT_BIND_HOST:
        return DEFAULT_BIND_HOST
    ip = _resolve_interface_ip(DEFAULT_BIND_INTERFACE)
    if ip:
        return ip
    ip = _resolve_any_interface_in_subnet(DEFAULT_BIND_SUBNET)
    if ip:
        return ip
    return ""


class UsbI2sConfig(BaseModel):
    capture_device: str
    playback_device: str
    channels: int = Field(ge=1)
    fallback_rate: int = Field(ge=1)
    preferred_format: str
    passthrough: bool
    alsa_buffer_time_us: int = Field(ge=1)
    alsa_latency_time_us: int = Field(ge=1)
    queue_time_ns: int = Field(ge=1)
    fade_ms: int = Field(ge=0)
    poll_interval_sec: float = Field(gt=0)
    restart_backoff_sec: float = Field(ge=0)
    keep_silence_when_no_capture: bool
    status_report_url: Optional[str] = None
    status_report_timeout_ms: int = Field(ge=1)
    status_report_min_interval_sec: float = Field(ge=0)
    control_endpoint: Optional[str] = None
    control_peer: Optional[str] = None
    control_require_peer: bool
    control_poll_interval_sec: float = Field(gt=0)
    control_timeout_ms: int = Field(ge=1)


class UsbI2sConfigUpdate(BaseModel):
    capture_device: Optional[str] = None
    playback_device: Optional[str] = None
    channels: Optional[int] = Field(default=None, ge=1)
    fallback_rate: Optional[int] = Field(default=None, ge=1)
    preferred_format: Optional[str] = None
    passthrough: Optional[bool] = None
    alsa_buffer_time_us: Optional[int] = Field(default=None, ge=1)
    alsa_latency_time_us: Optional[int] = Field(default=None, ge=1)
    queue_time_ns: Optional[int] = Field(default=None, ge=1)
    fade_ms: Optional[int] = Field(default=None, ge=0)
    poll_interval_sec: Optional[float] = Field(default=None, gt=0)
    restart_backoff_sec: Optional[float] = Field(default=None, ge=0)
    keep_silence_when_no_capture: Optional[bool] = None
    status_report_url: Optional[str] = None
    status_report_timeout_ms: Optional[int] = Field(default=None, ge=1)
    status_report_min_interval_sec: Optional[float] = Field(default=None, ge=0)
    control_endpoint: Optional[str] = None
    control_peer: Optional[str] = None
    control_require_peer: Optional[bool] = None
    control_poll_interval_sec: Optional[float] = Field(default=None, gt=0)
    control_timeout_ms: Optional[int] = Field(default=None, ge=1)


class StatusResponse(BaseModel):
    running: bool
    mode: str
    sample_rate: int
    format: str
    channels: int
    xruns: int
    last_error: Optional[str] = None
    last_error_at_unix_ms: Optional[int] = None
    uptime_sec: float
    updated_at_unix_ms: Optional[int] = None


_CONFIG_ENV_MAP = {
    "capture_device": "USB_I2S_CAPTURE_DEVICE",
    "playback_device": "USB_I2S_PLAYBACK_DEVICE",
    "channels": "USB_I2S_CHANNELS",
    "fallback_rate": "USB_I2S_FALLBACK_RATE",
    "preferred_format": "USB_I2S_PREFERRED_FORMAT",
    "passthrough": "USB_I2S_PASSTHROUGH",
    "alsa_buffer_time_us": "USB_I2S_ALSA_BUFFER_TIME_US",
    "alsa_latency_time_us": "USB_I2S_ALSA_LATENCY_TIME_US",
    "queue_time_ns": "USB_I2S_QUEUE_TIME_NS",
    "fade_ms": "USB_I2S_FADE_MS",
    "poll_interval_sec": "USB_I2S_POLL_INTERVAL_SEC",
    "restart_backoff_sec": "USB_I2S_RESTART_BACKOFF_SEC",
    "keep_silence_when_no_capture": "USB_I2S_KEEP_SILENCE",
    "status_report_url": "USB_I2S_STATUS_REPORT_URL",
    "status_report_timeout_ms": "USB_I2S_STATUS_REPORT_TIMEOUT_MS",
    "status_report_min_interval_sec": "USB_I2S_STATUS_REPORT_MIN_INTERVAL_SEC",
    "control_endpoint": "USB_I2S_CONTROL_ENDPOINT",
    "control_peer": "USB_I2S_CONTROL_PEER",
    "control_require_peer": "USB_I2S_CONTROL_REQUIRE_PEER",
    "control_poll_interval_sec": "USB_I2S_CONTROL_POLL_INTERVAL_SEC",
    "control_timeout_ms": "USB_I2S_CONTROL_TIMEOUT_MS",
}


def _default_config() -> UsbI2sConfig:
    defaults = UsbI2sBridgeConfig()
    return UsbI2sConfig(
        capture_device=defaults.capture_device,
        playback_device=defaults.playback_device,
        channels=defaults.channels,
        fallback_rate=defaults.fallback_rate,
        preferred_format=defaults.preferred_format,
        passthrough=defaults.passthrough,
        alsa_buffer_time_us=defaults.alsa_buffer_time_us,
        alsa_latency_time_us=defaults.alsa_latency_time_us,
        queue_time_ns=defaults.queue_time_ns,
        fade_ms=defaults.fade_ms,
        poll_interval_sec=defaults.poll_interval_sec,
        restart_backoff_sec=defaults.restart_backoff_sec,
        keep_silence_when_no_capture=defaults.keep_silence_when_no_capture,
        status_report_url=defaults.status_report_url,
        status_report_timeout_ms=defaults.status_report_timeout_ms,
        status_report_min_interval_sec=defaults.status_report_min_interval_sec,
        control_endpoint=defaults.control_endpoint,
        control_peer=defaults.control_peer,
        control_require_peer=defaults.control_require_peer,
        control_poll_interval_sec=defaults.control_poll_interval_sec,
        control_timeout_ms=defaults.control_timeout_ms,
    )


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        text = path.read_text()
    except OSError:
        return {}
    env: dict[str, str] = {}
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _coerce_value(value: str, target_type: type[Any]) -> Any:
    if target_type is bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def _load_config(path: Path) -> UsbI2sConfig:
    cfg = _default_config()
    env = _parse_env_file(path)
    data: dict[str, Any] = cfg.model_dump()
    for field, env_key in _CONFIG_ENV_MAP.items():
        if env_key not in env:
            continue
        value = env[env_key]
        if value == "" and data.get(field) is None:
            data[field] = None
            continue
        field_type = type(data[field])
        data[field] = _coerce_value(value, field_type)
    return UsbI2sConfig(**data)


def _format_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _write_config(path: Path, cfg: UsbI2sConfig) -> None:
    lines = [
        "# Auto-generated by raspberry_pi.control_api",
        f"# Updated at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    data = cfg.model_dump()
    for field, env_key in _CONFIG_ENV_MAP.items():
        value = data.get(field)
        lines.append(f"{env_key}={_format_env_value(value)}")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _load_status(path: Path) -> StatusResponse:
    if not path.exists():
        return StatusResponse(
            running=False,
            mode="none",
            sample_rate=0,
            format="",
            channels=0,
            xruns=0,
            uptime_sec=0.0,
        )
    try:
        payload = json.loads(path.read_text())
    except (ValueError, OSError):
        return StatusResponse(
            running=False,
            mode="unknown",
            sample_rate=0,
            format="",
            channels=0,
            xruns=0,
            uptime_sec=0.0,
        )
    return StatusResponse(
        running=bool(payload.get("running", False)),
        mode=str(payload.get("mode", "none")),
        sample_rate=int(payload.get("sample_rate", 0)),
        format=str(payload.get("format", "")),
        channels=int(payload.get("channels", 0)),
        xruns=int(payload.get("xruns", 0) or 0),
        last_error=payload.get("last_error"),
        last_error_at_unix_ms=payload.get("last_error_at_unix_ms"),
        uptime_sec=float(payload.get("uptime_sec", 0.0) or 0.0),
        updated_at_unix_ms=payload.get("updated_at_unix_ms"),
    )


def _restart_via_docker(container: str) -> dict[str, Any]:
    if not container:
        raise HTTPException(status_code=400, detail="docker container is not set")
    try:
        import docker
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail="docker SDK is not available"
        ) from exc
    try:
        client = docker.from_env()  # type: ignore[attr-defined]
        target = client.containers.get(container)
        target.restart(timeout=20)
        return {"mode": "docker", "container": container}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"docker restart failed: {exc}"
        ) from exc


def _restart_via_command(command: str) -> dict[str, Any]:
    if not command:
        raise HTTPException(status_code=400, detail="restart command is not configured")
    try:
        result = subprocess.run(
            shlex.split(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"restart failed: {exc.stderr.strip() or exc.stdout.strip()}",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="restart timed out") from exc
    return {
        "mode": "command",
        "command": command,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _run_restart(*, mode: str, command: str, container: str) -> dict[str, Any]:
    mode = mode.strip().lower()
    if mode == "docker":
        return _restart_via_docker(container)
    if mode == "command":
        return _restart_via_command(command)
    raise HTTPException(status_code=400, detail=f"unknown restart mode: {mode}")


def create_app(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    status_path: Path = DEFAULT_STATUS_PATH,
    restart_mode: str = DEFAULT_RESTART_MODE,
    restart_cmd: str = DEFAULT_RESTART_CMD,
    docker_container: str = DEFAULT_DOCKER_CONTAINER,
) -> FastAPI:
    app = FastAPI(title="Pi Control API", version="1.0")
    if restart_mode.strip().lower() == "docker":
        docker_sock = Path("/var/run/docker.sock")
        if not docker_sock.exists():
            print("[raspi-control-api] WARNING: /var/run/docker.sock not mounted")

    @app.get("/raspi/api/v1/status", response_model=StatusResponse)
    def get_status() -> StatusResponse:
        return _load_status(status_path)

    @app.get("/raspi/api/v1/config", response_model=UsbI2sConfig)
    def get_config() -> UsbI2sConfig:
        return _load_config(config_path)

    @app.put("/raspi/api/v1/config", response_model=UsbI2sConfig)
    def update_config(
        request: UsbI2sConfigUpdate,
        apply: bool = Query(
            default=True, description="Apply changes by restarting the bridge"
        ),
    ) -> UsbI2sConfig:
        current = _load_config(config_path)
        data = current.model_dump()
        updates = request.model_dump(exclude_unset=True)
        data.update(updates)
        merged = UsbI2sConfig(**data)
        _write_config(config_path, merged)
        if apply:
            _run_restart(
                mode=restart_mode,
                command=restart_cmd,
                container=docker_container,
            )
        return merged

    @app.post("/raspi/api/v1/actions/restart")
    def restart_bridge() -> dict[str, Any]:
        return {
            "status": "ok",
            "result": _run_restart(
                mode=restart_mode,
                command=restart_cmd,
                container=docker_container,
            ),
        }

    return app


app = create_app()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi control API")
    parser.add_argument(
        "--host",
        default=DEFAULT_BIND_HOST,
        help=(
            "bind host (default: auto-detect from interface/subnet; "
            "exits if auto-detect fails)"
        ),
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="path to USB-I2S config env file",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=DEFAULT_STATUS_PATH,
        help="path to USB-I2S status json file",
    )
    parser.add_argument(
        "--restart-mode",
        default=DEFAULT_RESTART_MODE,
        choices=["docker", "command"],
        help="restart method (docker or command)",
    )
    parser.add_argument(
        "--restart-cmd",
        default=DEFAULT_RESTART_CMD,
        help="command to restart the bridge (command mode only)",
    )
    parser.add_argument(
        "--docker-container",
        default=DEFAULT_DOCKER_CONTAINER,
        help="docker container name for restart",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    explicit_host = args.host.strip()

    # NOTE:
    # 起動直後は USB ネットワークのIPがまだ付いていないことがある。
    # bind できるまで少し待つ（成功したら即起動、タイムアウトしたら exit 非0 → restart）。
    wait_secs = max(0.0, DEFAULT_BIND_WAIT_SECS)
    retry_secs = max(0.2, DEFAULT_BIND_RETRY_SECS)
    deadline = time.monotonic() + wait_secs
    last_state = ""

    while True:
        host = explicit_host or _resolve_bind_host()
        if host and _is_bindable(host, args.port):
            break

        state = (
            f"host={host or '(none)'} port={args.port} "
            f"(interface={DEFAULT_BIND_INTERFACE}, subnet={DEFAULT_BIND_SUBNET}, "
            f"explicit_host={'yes' if explicit_host else 'no'})"
        )
        if state != last_state:
            print(f"[raspi-control-api] waiting for bindable address: {state}")
            last_state = state

        if time.monotonic() >= deadline:
            raise SystemExit(
                "[raspi-control-api] ERROR: failed to resolve/bind address within "
                f"{wait_secs:.0f}s. Last state: {state}. "
                "Set RPI_CONTROL_BIND_HOST (e.g. 192.168.55.100) or adjust "
                "RPI_CONTROL_BIND_INTERFACE / RPI_CONTROL_BIND_SUBNET."
            )

        time.sleep(retry_secs)

    print(
        f"[raspi-control-api] bind={host}:{args.port} "
        f"(interface={DEFAULT_BIND_INTERFACE}, subnet={DEFAULT_BIND_SUBNET}, wait={wait_secs:.0f}s)"
    )
    app_instance = create_app(
        config_path=args.config,
        status_path=args.status,
        restart_mode=args.restart_mode,
        restart_cmd=args.restart_cmd,
        docker_container=args.docker_container,
    )
    from uvicorn import run

    run(
        app_instance,
        host=host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
