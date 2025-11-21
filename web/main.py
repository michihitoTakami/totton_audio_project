"""
GPU Upsampler Web Control API
FastAPI-based control interface for the GPU audio upsampler daemon.
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configuration
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
DAEMON_SERVICE = "gpu_upsampler_alsa"  # systemd service name (if using systemd)

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
    buffer_size: int = 262144
    period_size: int = 32768
    upsample_ratio: int = 16
    block_size: int = 4096
    gain: float = 16.0
    filter_path: str = "data/coefficients/filter_1m_min_phase.bin"


class SettingsUpdate(BaseModel):
    """Partial settings update"""
    alsa_device: Optional[str] = None
    buffer_size: Optional[int] = None
    period_size: Optional[int] = None
    upsample_ratio: Optional[int] = None
    block_size: Optional[int] = None
    gain: Optional[float] = None
    filter_path: Optional[str] = None


class Status(BaseModel):
    """Current daemon status"""
    settings: Settings
    pipewire_connected: bool = False
    alsa_connected: bool = False
    clip_count: int = 0
    total_samples: int = 0
    clip_rate: float = 0.0
    daemon_running: bool = False


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
                buffer_size=data.get("bufferSize", 262144),
                period_size=data.get("periodSize", 32768),
                upsample_ratio=data.get("upsampleRatio", 16),
                block_size=data.get("blockSize", 4096),
                gain=data.get("gain", 16.0),
                filter_path=data.get("filterPath", "data/coefficients/filter_1m_min_phase.bin"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Config load error: {e}")
    return Settings()


def save_config(settings: Settings) -> bool:
    """Save configuration to JSON file"""
    try:
        data = {
            "alsaDevice": settings.alsa_device,
            "bufferSize": settings.buffer_size,
            "periodSize": settings.period_size,
            "upsampleRatio": settings.upsample_ratio,
            "blockSize": settings.block_size,
            "gain": settings.gain,
            "filterPath": settings.filter_path,
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
        pattern = r'(?:カード|card)\s+(\d+):\s+(\w+)\s+\[([^\]]+)\],\s+(?:デバイス|device)\s+(\d+):\s+([^\[]+)\[([^\]]+)\]'

        for line in result.stdout.splitlines():
            match = re.search(pattern, line)
            if match:
                card_num = match.group(1)
                card_id = match.group(2)
                card_name = match.group(3).strip()
                dev_num = match.group(4)
                dev_desc = match.group(6).strip()

                # Build hw:X,Y format
                hw_id = f"hw:{card_id},DEV={dev_num}" if dev_num != "0" else f"hw:{card_id}"

                # Build friendly name
                if "USB" in card_name or "SMSL" in card_name:
                    friendly = card_name
                elif "HDMI" in dev_desc:
                    friendly = f"{dev_desc}"
                else:
                    friendly = f"{card_name} - {dev_desc}"

                devices.append({
                    "id": f"hw:CARD={card_id},DEV={dev_num}",
                    "name": friendly,
                    "card": card_id,
                })
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

    return Status(
        settings=settings,
        pipewire_connected=pw_connected,
        alsa_connected=daemon_running,  # Simplified: assume ALSA connected if daemon running
        clip_count=0,  # TODO: Read from daemon stats file
        total_samples=0,
        clip_rate=0.0,
        daemon_running=daemon_running,
    )


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

    # Save configuration
    if not save_config(current):
        raise HTTPException(status_code=500, detail="Failed to save configuration")

    return ApiResponse(
        success=True,
        message="Settings updated" + (" (restart required)" if restart_required else ""),
        restart_required=restart_required,
    )


@app.post("/restart", response_model=ApiResponse)
async def restart_daemon():
    """
    Restart the daemon to apply new settings.
    Sends SIGHUP to trigger in-process config reload.
    """
    try:
        # Try to find daemon PID
        result = subprocess.run(
            ["pgrep", "-f", "gpu_upsampler_alsa"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return ApiResponse(
                success=False,
                message="Daemon not running. Start it manually with ./build/gpu_upsampler_alsa",
            )

        pid = result.stdout.strip().split()[0]
        hup_result = subprocess.run(["kill", "-HUP", pid], timeout=5)
        if hup_result.returncode != 0:
            raise HTTPException(status_code=500, detail="Failed to send SIGHUP to daemon")

        await asyncio.sleep(0.25)
        running = check_daemon_running()

        return ApiResponse(
            success=True,
            message=f"Reload signal sent to daemon (PID {pid})",
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
                if ("FL" in out_port and "FL" in in_port) or \
                   ("FR" in out_port and "FR" in in_port):
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
        raise HTTPException(status_code=500, detail="pw-link not found. Is PipeWire installed?")


@app.post("/eq-profile", response_model=ApiResponse)
async def set_eq_profile(profile: EqProfile):
    """
    [Future] Set parametric EQ profile.
    This endpoint is a placeholder for future EQ -> FIR coefficient generation.
    """
    # TODO: Implement EQ to FIR conversion
    return ApiResponse(
        success=False,
        message="EQ profile feature not yet implemented",
        data={"profile_name": profile.name, "bands_count": len(profile.bands)},
    )


@app.get("/devices")
async def list_devices():
    """List available ALSA playback devices"""
    devices = get_alsa_devices()
    return {"devices": devices}


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
            <div class="status-item" id="clipRate">
                <div class="label">Clipping</div>
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
                setStatus('clipRate', data.clip_rate.toFixed(2) + '%', data.clip_rate < 0.1);

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

        // Initial load
        fetchDevices();
        fetchStatus();
        setInterval(fetchStatus, 5000);
    </script>
</body>
</html>
"""


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=11881)
