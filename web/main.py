"""
GPU Upsampler Web Control API
FastAPI-based control interface for the GPU audio upsampler daemon.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

import shutil
import zmq

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add scripts directory to path for OPRA module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from opra import (
    apply_modern_target_correction,
    convert_opra_to_apo,
    get_database as get_opra_database,
)

# Configuration
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
EQ_PROFILES_DIR = Path(__file__).parent.parent / "data" / "EQ"
DAEMON_SERVICE = "gpu_upsampler_alsa"  # systemd service name (if using systemd)
DAEMON_BINARY = Path(__file__).parent.parent / "build" / "gpu_upsampler_alsa"
PID_FILE_PATH = Path("/tmp/gpu_upsampler_alsa.pid")
STATS_FILE_PATH = Path("/tmp/gpu_upsampler_stats.json")
ZEROMQ_IPC_PATH = "ipc:///tmp/gpu_os.sock"


# ============================================================================
# ZeroMQ Client for Daemon Control
# ============================================================================


class DaemonClient:
    """
    ZeroMQ client for communicating with the C++ audio daemon.
    Uses REQ/REP pattern over IPC socket.

    Thread Safety: Each DaemonClient instance should be used by a single
    thread/coroutine at a time. Use the context manager or create new
    instances per request for concurrent access.
    """

    def __init__(self, endpoint: str = ZEROMQ_IPC_PATH, timeout_ms: int = 3000):
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    def __enter__(self):
        """Context manager entry - creates connection."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes connection."""
        self.close()
        return False

    def _ensure_connected(self) -> zmq.Socket:
        """Ensure socket is connected, reconnect if needed."""
        if self._socket is None:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(self.endpoint)
        return self._socket

    def close(self):
        """Close connection."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    def send_command(self, command: str) -> tuple[bool, str]:
        """
        Send a command to the daemon and wait for response.

        Returns:
            (success, message) tuple
        """
        try:
            socket = self._ensure_connected()
            socket.send_string(command)
            response = socket.recv_string()

            # Parse response: "OK" or "OK:data" or "ERR:message"
            if response.startswith("OK"):
                if ":" in response:
                    return True, response.split(":", 1)[1]
                return True, "Command executed"
            elif response.startswith("ERR"):
                return False, response.split(":", 1)[
                    1
                ] if ":" in response else "Unknown error"
            else:
                return False, f"Unexpected response: {response}"

        except zmq.Again:
            # Timeout - daemon not responding
            self.close()  # Reset socket for next attempt
            return False, "Daemon not responding (timeout)"
        except zmq.ZMQError as e:
            self.close()
            return False, f"ZeroMQ error: {e}"

    def reload_config(self) -> tuple[bool, str]:
        """Send RELOAD command to daemon."""
        return self.send_command("RELOAD")

    def get_stats(self) -> tuple[bool, dict | str]:
        """Send STATS command and parse JSON response."""
        success, response = self.send_command("STATS")
        if success:
            try:
                return True, json.loads(response)
            except json.JSONDecodeError:
                return False, f"Invalid JSON: {response}"
        return False, response

    def ping(self) -> bool:
        """Check if daemon is responding."""
        success, _ = self.send_command("PING")
        return success


def get_daemon_client(timeout_ms: int = 3000) -> DaemonClient:
    """
    Factory function to create a new DaemonClient.

    Use as context manager for automatic cleanup:
        with get_daemon_client() as client:
            client.ping()
    """
    return DaemonClient(timeout_ms=timeout_ms)


app = FastAPI(
    title="GPU Upsampler Control",
    description="Web API for GPU Audio Upsampler daemon control",
    version="1.0.0",
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================================
# Data Models
# ============================================================================


class Settings(BaseModel):
    """Daemon configuration settings"""

    alsa_device: str = "hw:USB"
    input_sample_rate: int = 44100
    buffer_size: int = 262144
    period_size: int = 32768
    upsample_ratio: int = 16
    block_size: int = 4096
    gain: float = 16.0
    filter_path: str = "data/coefficients/filter_1m_min_phase.bin"
    eq_enabled: bool = False
    eq_profile_path: str = ""


class SettingsUpdate(BaseModel):
    """Partial settings update"""

    alsa_device: Optional[str] = None
    input_sample_rate: Optional[int] = None
    buffer_size: Optional[int] = None
    period_size: Optional[int] = None
    upsample_ratio: Optional[int] = None
    block_size: Optional[int] = None
    gain: Optional[float] = None
    filter_path: Optional[str] = None
    eq_enabled: Optional[bool] = None
    eq_profile_path: Optional[str] = None


class Status(BaseModel):
    """Current daemon status"""

    settings: Settings
    pipewire_connected: bool = False
    alsa_connected: bool = False
    clip_count: int = 0
    total_samples: int = 0
    clip_rate: float = 0.0
    daemon_running: bool = False
    eq_active: bool = False


class RewireRequest(BaseModel):
    """PipeWire rewire request"""

    app_name: str  # e.g., "spotify", "firefox"


class EqProfile(BaseModel):
    """Parametric EQ profile (future use)"""

    name: str
    bands: list[dict]  # [{freq, gain, q}, ...]


class ApiResponse(BaseModel):
    """Standard API response"""

    success: bool
    message: str
    data: Optional[dict] = None
    restart_required: bool = False


# ============================================================================
# Helper Functions
# ============================================================================


def load_config() -> Settings:
    """Load configuration from JSON file"""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            # Convert camelCase to snake_case
            return Settings(
                alsa_device=data.get("alsaDevice", "hw:USB"),
                input_sample_rate=data.get("inputSampleRate", 44100),
                buffer_size=data.get("bufferSize", 262144),
                period_size=data.get("periodSize", 32768),
                upsample_ratio=data.get("upsampleRatio", 16),
                block_size=data.get("blockSize", 4096),
                gain=data.get("gain", 16.0),
                filter_path=data.get(
                    "filterPath", "data/coefficients/filter_1m_min_phase.bin"
                ),
                eq_enabled=data.get("eqEnabled", False),
                eq_profile_path=data.get("eqProfilePath", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Config load error: {e}")
    return Settings()


def save_config(settings: Settings) -> bool:
    """Save configuration to JSON file"""
    try:
        data = {
            "alsaDevice": settings.alsa_device,
            "inputSampleRate": settings.input_sample_rate,
            "bufferSize": settings.buffer_size,
            "periodSize": settings.period_size,
            "upsampleRatio": settings.upsample_ratio,
            "blockSize": settings.block_size,
            "gain": settings.gain,
            "filterPath": settings.filter_path,
            "eqEnabled": settings.eq_enabled,
            "eqProfilePath": settings.eq_profile_path,
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except IOError as e:
        print(f"Config save error: {e}")
        return False


def check_daemon_running() -> bool:
    """Check if the daemon process is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "gpu_upsampler_alsa"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def get_daemon_pid() -> Optional[int]:
    """Get daemon PID from PID file, with process name verification"""
    if not PID_FILE_PATH.exists():
        return None
    try:
        pid = int(PID_FILE_PATH.read_text().strip())
        # Verify process exists
        os.kill(pid, 0)
        # Verify it's actually our daemon (guard against PID reuse)
        comm_path = Path(f"/proc/{pid}/comm")
        if comm_path.exists():
            comm = comm_path.read_text().strip()
            if comm != "gpu_upsampler_a":  # comm is truncated to 15 chars
                return None  # PID was reused by another process
        return pid
    except (ValueError, OSError):
        return None


def start_daemon() -> tuple[bool, str]:
    """Start the daemon process"""
    if check_daemon_running():
        return False, "Daemon is already running"

    if not DAEMON_BINARY.exists():
        return False, f"Daemon binary not found: {DAEMON_BINARY}"

    try:
        # Start daemon in background
        subprocess.Popen(
            [str(DAEMON_BINARY)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(DAEMON_BINARY.parent.parent),
        )
        return True, "Daemon started"
    except Exception as e:
        return False, f"Failed to start daemon: {e}"


def stop_daemon() -> tuple[bool, str]:
    """Stop the daemon process"""
    pid = get_daemon_pid()
    if pid is None:
        # Try pgrep as fallback
        try:
            result = subprocess.run(
                ["pgrep", "-f", "gpu_upsampler_alsa"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                pid = int(result.stdout.strip().split()[0])
            else:
                return False, "Daemon is not running"
        except Exception:
            return False, "Daemon is not running"

    try:
        os.kill(pid, signal.SIGTERM)
        return True, f"Sent SIGTERM to daemon (PID {pid})"
    except OSError as e:
        return False, f"Failed to stop daemon: {e}"


def check_pipewire_sink() -> bool:
    """Check if gpu_upsampler_sink exists in PipeWire"""
    try:
        result = subprocess.run(
            ["pw-cli", "list-objects"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "gpu_upsampler_sink" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def load_stats() -> dict:
    """Load statistics from daemon stats file"""
    if not STATS_FILE_PATH.exists():
        return {"clip_count": 0, "total_samples": 0, "clip_rate": 0.0}
    try:
        with open(STATS_FILE_PATH) as f:
            data = json.load(f)
        return {
            "clip_count": data.get("clip_count", 0),
            "total_samples": data.get("total_samples", 0),
            "clip_rate": data.get("clip_rate", 0.0),
        }
    except (json.JSONDecodeError, IOError):
        return {"clip_count": 0, "total_samples": 0, "clip_rate": 0.0}


def get_alsa_devices() -> list[dict]:
    """List available ALSA playback devices with friendly names"""
    devices = []
    try:
        # Use aplay -l to get device names
        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        import re

        # Parse: カード 3: AUDIO [SMSL USB AUDIO], デバイス 0: USB Audio [USB Audio]
        # Or:   card 3: AUDIO [SMSL USB AUDIO], device 0: USB Audio [USB Audio]
        pattern = r"(?:カード|card)\s+(\d+):\s+(\w+)\s+\[([^\]]+)\],\s+(?:デバイス|device)\s+(\d+):\s+([^\[]+)\[([^\]]+)\]"

        for line in result.stdout.splitlines():
            match = re.search(pattern, line)
            if match:
                match.group(1)
                card_id = match.group(2)
                card_name = match.group(3).strip()
                dev_num = match.group(4)
                dev_desc = match.group(6).strip()

                # Build hw:X,Y format

                # Build friendly name
                if "USB" in card_name or "SMSL" in card_name:
                    friendly = card_name
                elif "HDMI" in dev_desc:
                    friendly = f"{dev_desc}"
                else:
                    friendly = f"{card_name} - {dev_desc}"

                devices.append(
                    {
                        "id": f"hw:CARD={card_id},DEV={dev_num}",
                        "name": friendly,
                        "card": card_id,
                    }
                )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return devices


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI"""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    # Return embedded minimal HTML if file doesn't exist
    return get_embedded_html()


@app.get("/status", response_model=Status)
async def get_status():
    """
    Get current daemon status and settings.
    Returns configuration, connection state, and clipping statistics.
    """
    settings = load_config()
    daemon_running = check_daemon_running()
    pw_connected = check_pipewire_sink() if daemon_running else False
    stats = (
        load_stats()
        if daemon_running
        else {"clip_count": 0, "total_samples": 0, "clip_rate": 0.0}
    )

    return Status(
        settings=settings,
        pipewire_connected=pw_connected,
        alsa_connected=daemon_running,  # Simplified: assume ALSA connected if daemon running
        clip_count=stats["clip_count"],
        total_samples=stats["total_samples"],
        clip_rate=stats["clip_rate"],
        daemon_running=daemon_running,
        eq_active=settings.eq_enabled and bool(settings.eq_profile_path),
    )


@app.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """
    WebSocket endpoint for real-time stats streaming.
    Sends stats.json data every second while connected.
    When daemon is stopped, returns zero values (consistent with /status endpoint).
    """
    await websocket.accept()
    try:
        while True:
            daemon_running = check_daemon_running()
            if daemon_running:
                stats = load_stats()
            else:
                # Return zero values when daemon is stopped (consistent with /status)
                stats = {"clip_count": 0, "total_samples": 0, "clip_rate": 0.0}
            stats["daemon_running"] = daemon_running
            await websocket.send_json(stats)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception:
        # Connection closed unexpectedly
        pass


@app.post("/settings", response_model=ApiResponse)
async def update_settings(update: SettingsUpdate):
    """
    Update daemon settings.
    Some settings require a daemon restart to take effect.
    """
    current = load_config()
    restart_required = False

    # Apply updates
    if update.alsa_device is not None:
        if current.alsa_device != update.alsa_device:
            restart_required = True
        current.alsa_device = update.alsa_device

    if update.buffer_size is not None:
        if current.buffer_size != update.buffer_size:
            restart_required = True
        current.buffer_size = update.buffer_size

    if update.period_size is not None:
        if current.period_size != update.period_size:
            restart_required = True
        current.period_size = update.period_size

    if update.upsample_ratio is not None:
        if current.upsample_ratio != update.upsample_ratio:
            restart_required = True
        current.upsample_ratio = update.upsample_ratio

    if update.block_size is not None:
        if current.block_size != update.block_size:
            restart_required = True
        current.block_size = update.block_size

    if update.gain is not None:
        # Gain can potentially be changed without restart
        current.gain = update.gain

    if update.filter_path is not None:
        if current.filter_path != update.filter_path:
            restart_required = True
        current.filter_path = update.filter_path

    if update.eq_enabled is not None:
        if current.eq_enabled != update.eq_enabled:
            restart_required = True
        current.eq_enabled = update.eq_enabled

    if update.eq_profile_path is not None:
        if current.eq_profile_path != update.eq_profile_path:
            restart_required = True
        current.eq_profile_path = update.eq_profile_path

    # Save configuration
    if not save_config(current):
        raise HTTPException(status_code=500, detail="Failed to save configuration")

    return ApiResponse(
        success=True,
        message="Settings updated"
        + (" (restart required)" if restart_required else ""),
        restart_required=restart_required,
    )


# ============================================================================
# Daemon Control API Endpoints
# ============================================================================


@app.post("/daemon/start", response_model=ApiResponse)
async def daemon_start():
    """
    Start the daemon process.
    Returns error if daemon is already running.
    """
    success, message = start_daemon()
    if success:
        await asyncio.sleep(0.5)  # Wait for daemon to initialize
        running = check_daemon_running()
        return ApiResponse(
            success=running,
            message=message if running else "Daemon failed to start",
        )
    return ApiResponse(success=False, message=message)


@app.post("/daemon/stop", response_model=ApiResponse)
async def daemon_stop():
    """
    Stop the daemon process.
    Sends SIGTERM for graceful shutdown.
    """
    success, message = stop_daemon()
    if success:
        await asyncio.sleep(0.5)  # Wait for daemon to stop
        still_running = check_daemon_running()
        return ApiResponse(
            success=not still_running,
            message=message if not still_running else "Daemon did not stop in time",
        )
    return ApiResponse(success=False, message=message)


@app.post("/daemon/restart", response_model=ApiResponse)
async def daemon_restart():
    """
    Full restart: stop daemon, then start it again.
    Use /restart for hot reload via SIGHUP.
    """
    # Stop if running
    if check_daemon_running():
        stop_success, stop_msg = stop_daemon()
        if not stop_success:
            # Abort restart if stop failed
            return ApiResponse(
                success=False,
                message=f"Restart aborted: failed to stop daemon ({stop_msg})",
            )
        # Wait for graceful shutdown
        for _ in range(10):
            await asyncio.sleep(0.3)
            if not check_daemon_running():
                break
        else:
            return ApiResponse(
                success=False,
                message="Daemon did not stop in time for restart",
            )

    # Start daemon
    start_success, start_msg = start_daemon()
    if start_success:
        await asyncio.sleep(0.5)
        running = check_daemon_running()
        return ApiResponse(
            success=running,
            message="Daemon restarted successfully"
            if running
            else "Daemon failed to start after stop",
        )
    return ApiResponse(success=False, message=start_msg)


@app.get("/daemon/status")
async def daemon_status():
    """
    Get detailed daemon status including PID and uptime info.
    """
    pid = get_daemon_pid()
    running = check_daemon_running()
    pw_connected = check_pipewire_sink() if running else False

    # Check ZeroMQ connectivity (short timeout to avoid blocking)
    zmq_connected = False
    if running:
        with get_daemon_client(timeout_ms=500) as client:
            zmq_connected = client.ping()

    return {
        "running": running,
        "pid": pid,
        "pipewire_connected": pw_connected,
        "zmq_connected": zmq_connected,
        "pid_file": str(PID_FILE_PATH),
        "binary_path": str(DAEMON_BINARY),
        "binary_exists": DAEMON_BINARY.exists(),
    }


# ============================================================================
# ZeroMQ Control API Endpoints
# ============================================================================


@app.get("/daemon/zmq/ping")
async def zmq_ping():
    """
    Check if daemon is responding to ZeroMQ commands.
    """
    if not check_daemon_running():
        return {"connected": False, "message": "Daemon not running"}

    with get_daemon_client() as client:
        connected = client.ping()
        return {
            "connected": connected,
            "endpoint": ZEROMQ_IPC_PATH,
            "message": "Daemon responding"
            if connected
            else "Daemon not responding to ZeroMQ",
        }


@app.post("/daemon/zmq/command/{cmd}", response_model=ApiResponse)
async def zmq_command(cmd: str):
    """
    Send a raw command to daemon via ZeroMQ.

    Supported commands:
    - PING: Check connectivity
    - RELOAD: Reload configuration
    - STATS: Get statistics (JSON response)
    """
    if not check_daemon_running():
        return ApiResponse(
            success=False,
            message="Daemon not running",
        )

    # Validate command (whitelist for security)
    allowed_commands = {"PING", "RELOAD", "STATS"}
    cmd_upper = cmd.upper()
    if cmd_upper not in allowed_commands:
        return ApiResponse(
            success=False,
            message=f"Invalid command. Allowed: {', '.join(allowed_commands)}",
        )

    with get_daemon_client() as client:
        success, response = client.send_command(cmd_upper)
        return ApiResponse(
            success=success,
            message=response if isinstance(response, str) else "Command executed",
            data={"response": response} if success else None,
        )


@app.post("/restart", response_model=ApiResponse)
async def reload_daemon():
    """
    Reload daemon configuration.
    Uses ZeroMQ if daemon supports it, falls back to SIGHUP.
    """
    if not check_daemon_running():
        return ApiResponse(
            success=False,
            message="Daemon not running. Start it manually with ./build/gpu_upsampler_alsa",
        )

    # Try ZeroMQ first (preferred method)
    with get_daemon_client() as client:
        success, message = client.reload_config()
        if success:
            await asyncio.sleep(0.25)
            running = check_daemon_running()
            return ApiResponse(
                success=True,
                message="Config reloaded via ZeroMQ",
                restart_required=not running,
            )

    # Fallback to SIGHUP
    try:
        result = subprocess.run(
            ["pgrep", "-f", "gpu_upsampler_alsa"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return ApiResponse(
                success=False,
                message=f"ZeroMQ failed ({message}), SIGHUP fallback failed: daemon not found",
            )

        pid = result.stdout.strip().split()[0]
        hup_result = subprocess.run(["kill", "-HUP", pid], timeout=5)
        if hup_result.returncode != 0:
            return ApiResponse(
                success=False,
                message=f"ZeroMQ failed ({message}), SIGHUP also failed",
            )

        await asyncio.sleep(0.25)
        running = check_daemon_running()

        return ApiResponse(
            success=True,
            message=f"Reload via SIGHUP (PID {pid}) - ZeroMQ unavailable",
            restart_required=not running,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Restart command timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rewire", response_model=ApiResponse)
async def rewire_pipewire(request: RewireRequest):
    """
    Reconnect PipeWire audio routing for a specific application.
    Routes: app -> gpu_upsampler_sink
    """
    app_name = request.app_name.lower()

    try:
        # Find the application's output port
        result = subprocess.run(
            ["pw-link", "-o"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        output_ports = []
        for line in result.stdout.splitlines():
            if app_name in line.lower():
                output_ports.append(line.strip())

        if not output_ports:
            return ApiResponse(
                success=False,
                message=f"No output ports found for '{app_name}'",
            )

        # Find gpu_upsampler_sink input ports
        result = subprocess.run(
            ["pw-link", "-i"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        input_ports = []
        for line in result.stdout.splitlines():
            if "gpu_upsampler_sink" in line.lower():
                input_ports.append(line.strip())

        if not input_ports:
            return ApiResponse(
                success=False,
                message="gpu_upsampler_sink not found. Is the sink created?",
            )

        # Create links
        links_created = 0
        for out_port in output_ports[:2]:  # FL and FR
            for in_port in input_ports[:2]:
                # Match FL to FL, FR to FR
                if ("FL" in out_port and "FL" in in_port) or (
                    "FR" in out_port and "FR" in in_port
                ):
                    subprocess.run(
                        ["pw-link", out_port, in_port],
                        timeout=5,
                    )
                    links_created += 1

        return ApiResponse(
            success=True,
            message=f"Created {links_created} link(s) for '{app_name}'",
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="PipeWire command timed out")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, detail="pw-link not found. Is PipeWire installed?"
        )


# ============================================================================
# EQ API Endpoints
# ============================================================================


@app.get("/eq/profiles")
async def list_eq_profiles():
    """List available EQ profiles in data/EQ directory"""
    profiles = []
    if EQ_PROFILES_DIR.exists():
        for f in EQ_PROFILES_DIR.iterdir():
            if f.is_file() and f.suffix == ".txt":
                profiles.append(
                    {
                        "name": f.stem,
                        "filename": f.name,
                        "path": str(f),
                    }
                )
    return {"profiles": profiles}


@app.post("/eq/import", response_model=ApiResponse)
async def import_eq_profile(file: UploadFile):
    """Import an EQ profile file (AutoEq/Equalizer APO format)"""
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    # Ensure EQ profiles directory exists
    EQ_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    dest_path = EQ_PROFILES_DIR / file.filename
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return ApiResponse(
            success=True,
            message=f"Profile '{file.filename}' imported successfully",
            data={"path": str(dest_path)},
        )
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")


@app.post("/eq/activate/{name}", response_model=ApiResponse)
async def activate_eq_profile(name: str):
    """Activate an EQ profile by name"""
    # Find profile file
    profile_path = EQ_PROFILES_DIR / f"{name}.txt"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")

    # Update config
    settings = load_config()
    settings.eq_enabled = True
    settings.eq_profile_path = str(profile_path)

    if not save_config(settings):
        raise HTTPException(status_code=500, detail="Failed to save configuration")

    return ApiResponse(
        success=True,
        message=f"EQ profile '{name}' activated",
        data={"path": str(profile_path)},
        restart_required=True,
    )


@app.post("/eq/deactivate", response_model=ApiResponse)
async def deactivate_eq():
    """Deactivate EQ (disable without removing profile)"""
    settings = load_config()
    settings.eq_enabled = False

    if not save_config(settings):
        raise HTTPException(status_code=500, detail="Failed to save configuration")

    return ApiResponse(
        success=True,
        message="EQ deactivated",
        restart_required=True,
    )


@app.delete("/eq/profiles/{name}", response_model=ApiResponse)
async def delete_eq_profile(name: str):
    """Delete an EQ profile by name"""
    profile_path = EQ_PROFILES_DIR / f"{name}.txt"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")

    # Check if currently active
    settings = load_config()
    if settings.eq_profile_path == str(profile_path):
        # Deactivate first
        settings.eq_enabled = False
        settings.eq_profile_path = ""
        save_config(settings)

    try:
        profile_path.unlink()
        return ApiResponse(
            success=True,
            message=f"Profile '{name}' deleted",
        )
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")


@app.get("/eq/active")
async def get_active_eq():
    """Get currently active EQ profile info"""
    settings = load_config()
    if settings.eq_enabled and settings.eq_profile_path:
        path = Path(settings.eq_profile_path)
        return {
            "active": True,
            "name": path.stem if path.exists() else None,
            "path": settings.eq_profile_path,
        }
    return {"active": False, "name": None, "path": None}


@app.get("/devices")
async def list_devices():
    """List available ALSA playback devices"""
    devices = get_alsa_devices()
    return {"devices": devices}


# ============================================================================
# OPRA API Endpoints
# ============================================================================


@app.get("/opra/stats")
async def opra_stats():
    """Get OPRA database statistics"""
    try:
        db = get_opra_database()
        return {
            "vendors": db.vendor_count,
            "products": db.product_count,
            "eq_profiles": db.eq_profile_count,
            "license": "CC BY-SA 4.0",
            "attribution": "OPRA Project (https://github.com/opra-project/OPRA)",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/opra/vendors")
async def opra_vendors():
    """List all vendors in OPRA database"""
    try:
        db = get_opra_database()
        vendors = db.get_vendors()
        return {"vendors": vendors, "count": len(vendors)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/opra/search")
async def opra_search(q: str = "", limit: int = 50):
    """
    Search headphones by name.
    Returns products with embedded vendor info and EQ profiles.
    """
    try:
        db = get_opra_database()
        results = db.search(q, limit=limit)
        return {"results": results, "count": len(results), "query": q}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/opra/products/{product_id}")
async def opra_product(product_id: str):
    """Get a specific product with its EQ profiles"""
    try:
        db = get_opra_database()
        product = db.get_product(product_id)
        if product is None:
            raise HTTPException(
                status_code=404, detail=f"Product '{product_id}' not found"
            )
        return product
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/opra/eq/{eq_id}")
async def opra_eq_profile(eq_id: str, apply_correction: bool = False):
    """
    Get a specific EQ profile with APO format preview.

    Args:
        eq_id: EQ profile ID
        apply_correction: If True, apply Modern Target (KB5000_7) correction
    """
    try:
        db = get_opra_database()
        eq_data = db.get_eq_profile(eq_id)
        if eq_data is None:
            raise HTTPException(
                status_code=404, detail=f"EQ profile '{eq_id}' not found"
            )

        # Convert to APO format
        apo_profile = convert_opra_to_apo(eq_data)

        # Apply Modern Target correction if requested
        if apply_correction:
            apo_profile = apply_modern_target_correction(apo_profile)

        return {
            "id": eq_id,
            "name": eq_data.get("name", ""),
            "author": eq_data.get("author", ""),
            "details": apo_profile.details,  # Includes correction info if applied
            "parameters": eq_data.get("parameters", {}),
            "apo_format": apo_profile.to_apo_format(),
            "modern_target_applied": apply_correction,
            "attribution": {
                "license": "CC BY-SA 4.0",
                "source": "OPRA Project",
                "author": eq_data.get("author", "unknown"),
            },
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/opra/apply/{eq_id}", response_model=ApiResponse)
async def opra_apply_eq(eq_id: str, apply_correction: bool = False):
    """
    Apply an OPRA EQ profile.
    Converts to APO format, saves to data/EQ, and activates.

    Args:
        eq_id: EQ profile ID
        apply_correction: If True, apply Modern Target (KB5000_7) correction
    """
    try:
        db = get_opra_database()
        eq_data = db.get_eq_profile(eq_id)
        if eq_data is None:
            raise HTTPException(
                status_code=404, detail=f"EQ profile '{eq_id}' not found"
            )

        # Convert to APO format
        apo_profile = convert_opra_to_apo(eq_data)

        # Apply Modern Target correction if requested
        if apply_correction:
            apo_profile = apply_modern_target_correction(apo_profile)

        apo_content = apo_profile.to_apo_format()

        # Add attribution comment
        author = eq_data.get("author", "unknown")
        header = f"# OPRA: {eq_data.get('name', eq_id)}\n"
        header += f"# Author: {author}\n"
        header += (
            f"# Details: {apo_profile.details}\n"  # Includes correction info if applied
        )
        if apply_correction:
            header += "# Modern Target: KB5000_7 correction applied\n"
        header += "# License: CC BY-SA 4.0\n"
        header += "# Source: https://github.com/opra-project/OPRA\n\n"
        apo_content = header + apo_content

        # Save to EQ profiles directory
        EQ_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        # Use safe filename: replace problematic chars
        safe_name = eq_id.replace("/", "_").replace("\\", "_")
        suffix = "_kb5000_7" if apply_correction else ""
        profile_path = EQ_PROFILES_DIR / f"opra_{safe_name}{suffix}.txt"
        profile_path.write_text(apo_content)

        # Update config to activate
        settings = load_config()
        settings.eq_enabled = True
        settings.eq_profile_path = str(profile_path)

        if not save_config(settings):
            raise HTTPException(status_code=500, detail="Failed to save configuration")

        correction_msg = " (Modern Target)" if apply_correction else ""
        return ApiResponse(
            success=True,
            message=f"OPRA EQ '{eq_data.get('name', eq_id)}' by {author}{correction_msg} activated",
            data={
                "path": str(profile_path),
                "author": author,
                "modern_target_applied": apply_correction,
                "attribution": "CC BY-SA 4.0 - OPRA Project",
            },
            restart_required=True,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ============================================================================
# Embedded HTML UI
# ============================================================================


def get_embedded_html() -> str:
    """Return embedded HTML UI"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Upsampler</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            max-width: 500px;
            margin: 0 auto;
        }
        h1 { color: #00d4ff; margin-bottom: 20px; font-size: 1.4em; }
        h2 { color: #888; font-size: 11px; margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 1px; }
        .card {
            background: #16213e;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status-item {
            text-align: center;
            flex: 1;
        }
        .status-item .label { font-size: 10px; color: #666; text-transform: uppercase; }
        .status-item .value { font-size: 14px; font-weight: 600; margin-top: 2px; }
        .status-item.ok .value { color: #00ff88; }
        .status-item.error .value { color: #ff4444; }
        .form-group { margin-bottom: 12px; }
        .form-group label {
            display: block;
            margin-bottom: 6px;
            color: #aaa;
            font-size: 13px;
        }
        .form-group select {
            width: 100%;
            padding: 12px;
            border: 1px solid #0f3460;
            border-radius: 6px;
            background: #0f3460;
            color: #eee;
            font-size: 14px;
            cursor: pointer;
        }
        .form-group select:focus {
            outline: none;
            border-color: #00d4ff;
        }
        .btn-row { display: flex; gap: 10px; margin-top: 16px; }
        button {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-primary { background: #00d4ff; color: #000; }
        .btn-primary:hover { background: #00a8cc; }
        .btn-primary:disabled { background: #555; color: #888; cursor: not-allowed; }
        .btn-secondary { background: #0f3460; color: #eee; }
        .btn-secondary:hover { background: #1a4b7c; }
        .btn-secondary:disabled { background: #333; color: #666; cursor: not-allowed; }
        .message {
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            display: none;
            font-size: 13px;
            text-align: center;
        }
        .message.success { background: #00ff8840; display: block; }
        .message.error { background: #ff444440; display: block; }
    </style>
</head>
<body>
    <h1>GPU Upsampler</h1>

    <h2>Status</h2>
    <div class="card">
        <div class="status-row">
            <div class="status-item" id="daemonStatus">
                <div class="label">Daemon</div>
                <div class="value">-</div>
            </div>
            <div class="status-item" id="pwStatus">
                <div class="label">PipeWire</div>
                <div class="value">-</div>
            </div>
            <div class="status-item" id="eqStatus">
                <div class="label">EQ</div>
                <div class="value">-</div>
            </div>
        </div>
    </div>

    <h2>Output Device</h2>
    <div class="card">
        <form id="settingsForm">
            <div class="form-group">
                <select id="alsaDevice">
                    <option value="">Loading...</option>
                </select>
            </div>
            <div class="btn-row">
                <button type="submit" class="btn-primary" id="saveBtn">Save & Restart</button>
            </div>
        </form>
        <div id="settingsMessage" class="message"></div>
    </div>

    <h2>Headphone EQ (OPRA)</h2>
    <div class="card">
        <div class="form-group">
            <label>Search Headphones</label>
            <input type="text" id="opraSearch" placeholder="e.g. HD650, DT770, AirPods..."
                   style="width:100%; padding:12px; border:1px solid #0f3460; border-radius:6px; background:#0f3460; color:#eee; font-size:14px;">
        </div>
        <div id="opraResults" style="max-height:200px; overflow-y:auto; margin-top:8px;"></div>
        <div id="opraSelected" style="display:none; margin-top:12px; padding:12px; background:#0f3460; border-radius:6px;">
            <div style="font-weight:600;" id="selectedName">-</div>
            <div style="font-size:12px; color:#888;" id="selectedVendor">-</div>
            <select id="opraEqSelect" style="width:100%; padding:8px; margin-top:8px; border:1px solid #16213e; border-radius:4px; background:#16213e; color:#eee;">
            </select>
            <label style="display:flex; align-items:center; gap:8px; margin-top:12px; cursor:pointer;">
                <input type="checkbox" id="modernTargetCheckbox" checked style="width:16px; height:16px; accent-color:#00d4ff;">
                <span style="font-size:13px;">Modern Target (KB5000_7)</span>
            </label>
            <div style="font-size:10px; color:#666; margin-left:24px;">最新のターゲットカーブに補正</div>
        </div>
        <div class="btn-row">
            <button type="button" class="btn-primary" id="applyOpraBtn" disabled>Apply EQ</button>
            <button type="button" class="btn-secondary" id="deactivateEqBtn">EQ Off</button>
        </div>
        <div id="opraMessage" class="message"></div>
        <div style="font-size:10px; color:#555; margin-top:12px; text-align:center;">
            EQ data: <a href="https://github.com/opra-project/OPRA" target="_blank" style="color:#00d4ff;">OPRA Project</a> (CC BY-SA 4.0)
        </div>
    </div>

    <script>
        const API = '';
        let currentAlsaDevice = '';
        let deviceList = [];

        async function fetchDevices() {
            try {
                const res = await fetch(API + '/devices');
                const data = await res.json();
                deviceList = data.devices;
                updateDeviceSelect();
            } catch (e) {
                console.error('Failed to fetch devices:', e);
            }
        }

        function updateDeviceSelect() {
            const select = document.getElementById('alsaDevice');
            select.innerHTML = '';

            deviceList.forEach(device => {
                const opt = document.createElement('option');
                opt.value = device.id;
                opt.textContent = device.name;
                if (device.id === currentAlsaDevice ||
                    currentAlsaDevice.includes(device.card)) {
                    opt.selected = true;
                }
                select.appendChild(opt);
            });

            // If current device not in list, add it
            const ids = deviceList.map(d => d.id);
            if (currentAlsaDevice && !ids.some(id => currentAlsaDevice.includes(id.split(',')[0].split('=')[1]))) {
                const opt = document.createElement('option');
                opt.value = currentAlsaDevice;
                opt.textContent = currentAlsaDevice;
                opt.selected = true;
                select.insertBefore(opt, select.firstChild);
            }
        }

        async function fetchStatus() {
            try {
                const res = await fetch(API + '/status');
                const data = await res.json();

                setStatus('daemonStatus', data.daemon_running ? 'Running' : 'Stopped', data.daemon_running);
                setStatus('pwStatus', data.pipewire_connected ? 'OK' : 'N/A', data.pipewire_connected);
                setStatus('eqStatus', data.eq_active ? 'ON' : 'OFF', data.eq_active);

                if (currentAlsaDevice !== data.settings.alsa_device) {
                    currentAlsaDevice = data.settings.alsa_device;
                    updateDeviceSelect();
                }
            } catch (e) {
                console.error('Failed to fetch status:', e);
            }
        }

        function setStatus(id, text, ok) {
            const el = document.getElementById(id);
            el.querySelector('.value').textContent = text;
            el.classList.remove('ok', 'error');
            el.classList.add(ok ? 'ok' : 'error');
        }

        function showMessage(text, success) {
            const el = document.getElementById('settingsMessage');
            el.textContent = text;
            el.classList.remove('success', 'error');
            el.classList.add(success ? 'success' : 'error');
            setTimeout(() => el.classList.remove('success', 'error'), 4000);
        }

        document.getElementById('settingsForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const newDevice = document.getElementById('alsaDevice').value;
            const btn = document.getElementById('saveBtn');

            btn.disabled = true;
            btn.textContent = 'Saving...';

            try {
                const res = await fetch(API + '/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ alsa_device: newDevice }),
                });
                const data = await res.json();

                if (data.success && data.restart_required) {
                    btn.textContent = 'Restarting...';
                    await fetch(API + '/restart', { method: 'POST' });
                    showMessage('Daemon restarting...', true);
                    setTimeout(fetchStatus, 2000);
                } else {
                    showMessage(data.message, data.success);
                }
            } catch (e) {
                showMessage('Error: ' + e.message, false);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Save & Restart';
            }
        });

        // OPRA Functions
        let selectedProduct = null;
        let searchTimeout = null;

        function showOpraMessage(text, success) {
            const el = document.getElementById('opraMessage');
            el.textContent = text;
            el.classList.remove('success', 'error');
            el.classList.add(success ? 'success' : 'error');
            setTimeout(() => el.classList.remove('success', 'error'), 4000);
        }

        async function searchOpra(query) {
            if (!query || query.length < 2) {
                document.getElementById('opraResults').innerHTML = '';
                return;
            }
            try {
                const res = await fetch(API + '/opra/search?q=' + encodeURIComponent(query) + '&limit=20');
                const data = await res.json();
                renderOpraResults(data.results);
            } catch (e) {
                console.error('OPRA search failed:', e);
            }
        }

        function renderOpraResults(results) {
            const container = document.getElementById('opraResults');
            if (!results.length) {
                container.innerHTML = '<div style="color:#666; font-size:12px; padding:8px;">No results</div>';
                return;
            }
            container.innerHTML = results.map(r => `
                <div class="opra-item" data-id="${r.id}" style="padding:8px; cursor:pointer; border-bottom:1px solid #0f3460;">
                    <div style="font-weight:500;">${r.vendor.name} ${r.name}</div>
                    <div style="font-size:11px; color:#666;">${r.eq_profiles.length} EQ profile(s)</div>
                </div>
            `).join('');

            // Add click handlers
            container.querySelectorAll('.opra-item').forEach(el => {
                el.addEventListener('click', () => selectProduct(results.find(r => r.id === el.dataset.id)));
                el.addEventListener('mouseenter', () => el.style.background = '#0f3460');
                el.addEventListener('mouseleave', () => el.style.background = '');
            });
        }

        function selectProduct(product) {
            selectedProduct = product;
            document.getElementById('opraResults').innerHTML = '';
            document.getElementById('opraSearch').value = '';
            document.getElementById('opraSelected').style.display = 'block';
            document.getElementById('selectedName').textContent = product.name;
            document.getElementById('selectedVendor').textContent = product.vendor.name;

            // Populate EQ profiles
            const select = document.getElementById('opraEqSelect');
            select.innerHTML = product.eq_profiles.map(eq =>
                `<option value="${eq.id}">${eq.author || 'unknown'} - ${eq.details || 'EQ'}</option>`
            ).join('');

            document.getElementById('applyOpraBtn').disabled = false;
        }

        document.getElementById('opraSearch').addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => searchOpra(e.target.value), 300);
        });

        document.getElementById('applyOpraBtn').addEventListener('click', async () => {
            const eqId = document.getElementById('opraEqSelect').value;
            if (!eqId) {
                showOpraMessage('Select an EQ profile', false);
                return;
            }
            const btn = document.getElementById('applyOpraBtn');
            btn.disabled = true;
            btn.textContent = 'Applying...';
            try {
                const applyCorrection = document.getElementById('modernTargetCheckbox').checked;
                const res = await fetch(API + '/opra/apply/' + encodeURIComponent(eqId) + '?apply_correction=' + applyCorrection, { method: 'POST' });
                const data = await res.json();
                showOpraMessage(data.message, data.success);
                if (data.success && data.restart_required) {
                    btn.textContent = 'Restarting...';
                    await fetch(API + '/restart', { method: 'POST' });
                    setTimeout(fetchStatus, 2000);
                }
            } catch (e) {
                showOpraMessage('Error: ' + e.message, false);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Apply EQ';
            }
        });

        document.getElementById('deactivateEqBtn').addEventListener('click', async () => {
            try {
                const res = await fetch(API + '/eq/deactivate', { method: 'POST' });
                const data = await res.json();
                showOpraMessage(data.message, data.success);
                if (data.success && data.restart_required) {
                    await fetch(API + '/restart', { method: 'POST' });
                    setTimeout(fetchStatus, 2000);
                }
            } catch (e) {
                showOpraMessage('Error: ' + e.message, false);
            }
        });

        // Initial load
        fetchDevices();
        fetchStatus();
        setInterval(fetchStatus, 5000);
    </script>
</body>
</html>
"""


def get_admin_html() -> str:
    """Return admin dashboard HTML"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Upsampler - Admin</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            max-width: 600px;
            margin: 0 auto;
        }
        h1 { color: #ff6b6b; margin-bottom: 8px; font-size: 1.4em; }
        .subtitle { color: #666; font-size: 12px; margin-bottom: 20px; }
        h2 { color: #888; font-size: 11px; margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 1px; }
        .card {
            background: #16213e;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        .stat-item {
            background: #0f3460;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-item .label { font-size: 10px; color: #666; text-transform: uppercase; }
        .stat-item .value { font-size: 18px; font-weight: 600; margin-top: 4px; color: #00d4ff; }
        .stat-item.warning .value { color: #ffaa00; }
        .stat-item.error .value { color: #ff4444; }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .status-item { text-align: center; flex: 1; }
        .status-item .label { font-size: 10px; color: #666; text-transform: uppercase; }
        .status-item .value { font-size: 14px; font-weight: 600; margin-top: 2px; }
        .status-item.ok .value { color: #00ff88; }
        .status-item.error .value { color: #ff4444; }
        .btn-row { display: flex; gap: 10px; margin-top: 12px; }
        button {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-success { background: #00ff88; color: #000; }
        .btn-success:hover { background: #00cc6a; }
        .btn-danger { background: #ff4444; color: #fff; }
        .btn-danger:hover { background: #cc3333; }
        .btn-warning { background: #ffaa00; color: #000; }
        .btn-warning:hover { background: #cc8800; }
        button:disabled { background: #555; color: #888; cursor: not-allowed; }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #0f3460;
            font-size: 13px;
        }
        .info-row:last-child { border-bottom: none; }
        .info-row .label { color: #888; }
        .info-row .value { color: #eee; font-family: monospace; }
        .message {
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            display: none;
            font-size: 13px;
            text-align: center;
        }
        .message.success { background: #00ff8840; display: block; }
        .message.error { background: #ff444440; display: block; }
        .back-link { color: #00d4ff; text-decoration: none; font-size: 13px; }
        .back-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>GPU Upsampler Admin</h1>
    <div class="subtitle"><a href="/" class="back-link">&larr; Back to User Page</a></div>

    <h2>Daemon Control</h2>
    <div class="card">
        <div class="status-row">
            <div class="status-item" id="daemonStatus">
                <div class="label">Status</div>
                <div class="value">-</div>
            </div>
            <div class="status-item" id="pidStatus">
                <div class="label">PID</div>
                <div class="value">-</div>
            </div>
            <div class="status-item" id="pwStatus">
                <div class="label">PipeWire</div>
                <div class="value">-</div>
            </div>
        </div>
        <div class="btn-row">
            <button class="btn-success" id="startBtn">Start</button>
            <button class="btn-danger" id="stopBtn">Stop</button>
            <button class="btn-warning" id="restartBtn">Restart</button>
        </div>
        <div id="controlMessage" class="message"></div>
    </div>

    <h2>Statistics</h2>
    <div class="card">
        <div class="stat-grid">
            <div class="stat-item" id="clipRate">
                <div class="label">Clip Rate</div>
                <div class="value">-</div>
            </div>
            <div class="stat-item" id="clipCount">
                <div class="label">Clipped Samples</div>
                <div class="value">-</div>
            </div>
            <div class="stat-item" id="totalSamples">
                <div class="label">Total Samples</div>
                <div class="value">-</div>
            </div>
            <div class="stat-item" id="eqStatus">
                <div class="label">EQ</div>
                <div class="value">-</div>
            </div>
        </div>
    </div>

    <h2>System Info</h2>
    <div class="card">
        <div class="info-row">
            <span class="label">PID File</span>
            <span class="value" id="pidFile">-</span>
        </div>
        <div class="info-row">
            <span class="label">Binary</span>
            <span class="value" id="binaryPath">-</span>
        </div>
        <div class="info-row">
            <span class="label">ALSA Device</span>
            <span class="value" id="alsaDevice">-</span>
        </div>
        <div class="info-row">
            <span class="label">Upsample Ratio</span>
            <span class="value" id="upsampleRatio">-</span>
        </div>
    </div>

    <script>
        const API = '';
        let statsWebSocket = null;
        let wsReconnectAttempts = 0;
        const MAX_RECONNECT_ATTEMPTS = 5;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + window.location.host + '/ws/stats';

            try {
                statsWebSocket = new WebSocket(wsUrl);

                statsWebSocket.onopen = () => {
                    console.log('WebSocket connected');
                    wsReconnectAttempts = 0;
                };

                statsWebSocket.onmessage = (event) => {
                    try {
                        const stats = JSON.parse(event.data);
                        updateStatsFromWebSocket(stats);
                    } catch (e) {
                        console.error('Failed to parse WebSocket message:', e);
                    }
                };

                statsWebSocket.onclose = () => {
                    console.log('WebSocket disconnected');
                    statsWebSocket = null;
                    // Attempt to reconnect
                    if (wsReconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                        wsReconnectAttempts++;
                        setTimeout(connectWebSocket, 2000 * wsReconnectAttempts);
                    }
                };

                statsWebSocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            } catch (e) {
                console.error('Failed to create WebSocket:', e);
            }
        }

        function updateStatsFromWebSocket(stats) {
            // Stats - clip_rate is now a ratio (0-1), multiply by 100 for percentage
            const clipPct = (stats.clip_rate * 100).toFixed(4);
            setStat('clipRate', clipPct + '%', stats.clip_rate < 0.001 ? '' : (stats.clip_rate < 0.01 ? 'warning' : 'error'));
            setStat('clipCount', formatNumber(stats.clip_count), '');
            setStat('totalSamples', formatNumber(stats.total_samples), '');

            // Daemon running status from WebSocket
            const daemonRunning = stats.daemon_running;
            setStatus('daemonStatus', daemonRunning ? 'Running' : 'Stopped', daemonRunning);
            document.getElementById('startBtn').disabled = daemonRunning;
            document.getElementById('stopBtn').disabled = !daemonRunning;
        }

        async function fetchStatus() {
            try {
                const [statusRes, daemonRes] = await Promise.all([
                    fetch(API + '/status'),
                    fetch(API + '/daemon/status')
                ]);
                const status = await statusRes.json();
                const daemon = await daemonRes.json();

                // Daemon status
                setStatus('daemonStatus', daemon.running ? 'Running' : 'Stopped', daemon.running);
                document.getElementById('pidStatus').querySelector('.value').textContent = daemon.pid || '-';
                document.getElementById('pidStatus').classList.remove('ok', 'error');
                document.getElementById('pidStatus').classList.add(daemon.pid ? 'ok' : 'error');
                setStatus('pwStatus', daemon.pipewire_connected ? 'OK' : 'N/A', daemon.pipewire_connected);

                // Stats - clip_rate is now a ratio (0-1), multiply by 100 for percentage
                const clipPct = (status.clip_rate * 100).toFixed(4);
                setStat('clipRate', clipPct + '%', status.clip_rate < 0.001 ? '' : (status.clip_rate < 0.01 ? 'warning' : 'error'));
                setStat('clipCount', formatNumber(status.clip_count), '');
                setStat('totalSamples', formatNumber(status.total_samples), '');
                setStat('eqStatus', status.eq_active ? 'ON' : 'OFF', status.eq_active ? '' : 'error');

                // System info
                document.getElementById('pidFile').textContent = daemon.pid_file || '-';
                document.getElementById('binaryPath').textContent = daemon.binary_path ? daemon.binary_path.split('/').pop() : '-';
                document.getElementById('alsaDevice').textContent = status.settings.alsa_device || '-';
                document.getElementById('upsampleRatio').textContent = status.settings.upsample_ratio + 'x';

                // Enable/disable buttons based on state
                document.getElementById('startBtn').disabled = daemon.running;
                document.getElementById('stopBtn').disabled = !daemon.running;
            } catch (e) {
                console.error('Failed to fetch status:', e);
            }
        }

        function setStatus(id, text, ok) {
            const el = document.getElementById(id);
            el.querySelector('.value').textContent = text;
            el.classList.remove('ok', 'error');
            el.classList.add(ok ? 'ok' : 'error');
        }

        function setStat(id, value, level) {
            const el = document.getElementById(id);
            el.querySelector('.value').textContent = value;
            el.classList.remove('warning', 'error');
            if (level) el.classList.add(level);
        }

        function formatNumber(n) {
            if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
            if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
            if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
            return String(n);
        }

        function showMessage(text, success) {
            const el = document.getElementById('controlMessage');
            el.textContent = text;
            el.classList.remove('success', 'error');
            el.classList.add(success ? 'success' : 'error');
            setTimeout(() => el.classList.remove('success', 'error'), 4000);
        }

        document.getElementById('startBtn').addEventListener('click', async () => {
            const btn = document.getElementById('startBtn');
            btn.disabled = true;
            try {
                const res = await fetch(API + '/daemon/start', { method: 'POST' });
                const data = await res.json();
                showMessage(data.message, data.success);
                setTimeout(fetchStatus, 1000);
            } catch (e) {
                showMessage('Error: ' + e.message, false);
            }
            btn.disabled = false;
        });

        document.getElementById('stopBtn').addEventListener('click', async () => {
            const btn = document.getElementById('stopBtn');
            btn.disabled = true;
            try {
                const res = await fetch(API + '/daemon/stop', { method: 'POST' });
                const data = await res.json();
                showMessage(data.message, data.success);
                setTimeout(fetchStatus, 1000);
            } catch (e) {
                showMessage('Error: ' + e.message, false);
            }
            btn.disabled = false;
        });

        document.getElementById('restartBtn').addEventListener('click', async () => {
            const btn = document.getElementById('restartBtn');
            btn.disabled = true;
            btn.textContent = 'Restarting...';
            try {
                const res = await fetch(API + '/daemon/restart', { method: 'POST' });
                const data = await res.json();
                showMessage(data.message, data.success);
                setTimeout(fetchStatus, 2000);
            } catch (e) {
                showMessage('Error: ' + e.message, false);
            }
            btn.disabled = false;
            btn.textContent = 'Restart';
        });

        // Initial load and auto-refresh
        fetchStatus();
        // Full status refresh every 5 seconds (for daemon info, settings, etc.)
        setInterval(fetchStatus, 5000);
        // Connect WebSocket for real-time stats (1 second updates)
        connectWebSocket();
    </script>
</body>
</html>
"""


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve the admin dashboard"""
    return get_admin_html()


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=11881)
