"""PipeWire/PulseAudio sink management for GPU Upsampler.

This module provides functions equivalent to scripts/daemon.sh for:
- Creating the gpu_upsampler_sink (PipeWire null-sink)
- Managing default sink settings
- Setting up PipeWire monitor links to daemon input
"""

import subprocess
import time
from typing import Optional

from ..constants import (
    DEFAULT_SINK_FILE_PATH,
    GPU_SINK_NAME,
    GPU_UPSAMPLER_INPUT_NODE,
)


def get_default_sink() -> Optional[str]:
    """Get current default PulseAudio/PipeWire sink.

    Returns:
        Sink name or None if unable to determine.
    """
    try:
        result = subprocess.run(
            ["pactl", "info"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        for line in result.stdout.split("\n"):
            if "Default Sink:" in line:
                return line.split(":", 1)[1].strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def sink_exists(sink_name: str) -> bool:
    """Check if a PulseAudio/PipeWire sink exists.

    Args:
        sink_name: Name of the sink to check.

    Returns:
        True if sink exists, False otherwise.
    """
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        for line in result.stdout.split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1] == sink_name:
                return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return False


def select_fallback_sink() -> Optional[str]:
    """Select a fallback sink (first non-GPU sink).

    Returns:
        Sink name or None if no fallback available.
    """
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        for line in result.stdout.split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1] != GPU_SINK_NAME:
                return parts[1]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def remember_default_sink() -> None:
    """Remember current default sink before switching to GPU sink.

    Saves the current default sink to a file for later restoration.
    """
    current = get_default_sink()

    if current and current != GPU_SINK_NAME:
        try:
            DEFAULT_SINK_FILE_PATH.write_text(current)
            return
        except IOError:
            pass

    # If the default sink is already GPU, remember the first non-GPU sink
    if not DEFAULT_SINK_FILE_PATH.exists():
        fallback = select_fallback_sink()
        if fallback:
            try:
                DEFAULT_SINK_FILE_PATH.write_text(fallback)
            except IOError:
                pass


def move_sink_inputs(target_sink: str) -> None:
    """Move all existing sink inputs to target sink.

    Args:
        target_sink: Name of the sink to move inputs to.
    """
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sink-inputs"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        for line in result.stdout.split("\n"):
            if line.strip():
                input_id = line.split()[0]
                subprocess.run(
                    ["pactl", "move-sink-input", input_id, target_sink],
                    capture_output=True,
                    timeout=2,
                )
    except (subprocess.SubprocessError, FileNotFoundError):
        pass


def set_default_sink(sink_name: str) -> bool:
    """Set default PulseAudio/PipeWire sink and move existing inputs.

    Args:
        sink_name: Name of the sink to set as default.

    Returns:
        True if successful, False otherwise.
    """
    if not sink_exists(sink_name):
        return False

    try:
        # Set default sink
        current = get_default_sink()
        if current != sink_name:
            subprocess.run(
                ["pactl", "set-default-sink", sink_name],
                capture_output=True,
                timeout=2,
            )

        # Move existing sink inputs to new sink
        move_sink_inputs(sink_name)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def create_gpu_sink() -> bool:
    """Create gpu_upsampler_sink if it doesn't exist.

    Returns:
        True if sink exists or was created, False on failure.
    """
    if sink_exists(GPU_SINK_NAME):
        return True  # Already exists

    try:
        result = subprocess.run(
            [
                "pactl",
                "load-module",
                "module-null-sink",
                f"sink_name={GPU_SINK_NAME}",
                'sink_properties=device.description="GPU_Upsampler_Sink"',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            time.sleep(0.3)  # Wait for sink to be registered
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return False


def wait_for_daemon_node(timeout_sec: float = 5.0) -> bool:
    """Wait until GPU Upsampler Input node appears in PipeWire.

    Args:
        timeout_sec: Maximum time to wait in seconds.

    Returns:
        True if node appeared, False on timeout.
    """
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            result = subprocess.run(
                ["pw-link", "-i"],  # List input ports
                capture_output=True,
                text=True,
                timeout=2,
            )
            if f"{GPU_UPSAMPLER_INPUT_NODE}:input_FL" in result.stdout:
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        time.sleep(0.3)
    return False


def setup_pipewire_links() -> tuple[bool, str]:
    """Setup PipeWire links from gpu_upsampler_sink monitor to daemon input.

    Creates links:
    - gpu_upsampler_sink:monitor_FL -> GPU Upsampler Input:input_FL
    - gpu_upsampler_sink:monitor_FR -> GPU Upsampler Input:input_FR

    Returns:
        Tuple of (success, message).
    """
    links = [
        (f"{GPU_SINK_NAME}:monitor_FL", f"{GPU_UPSAMPLER_INPUT_NODE}:input_FL"),
        (f"{GPU_SINK_NAME}:monitor_FR", f"{GPU_UPSAMPLER_INPUT_NODE}:input_FR"),
    ]

    errors = []
    for source, target in links:
        try:
            result = subprocess.run(
                ["pw-link", source, target],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # pw-link returns non-zero if link already exists, which is OK
            if result.returncode != 0:
                stderr_lower = result.stderr.lower()
                # "already linked" or similar messages are not errors
                if "already" not in stderr_lower and "exists" not in stderr_lower:
                    errors.append(f"{source} -> {target}: {result.stderr.strip()}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            errors.append(f"{source} -> {target}: {e}")

    if errors:
        return False, "; ".join(errors)
    return True, "Links configured"


def restore_default_sink() -> None:
    """Restore previously remembered default sink.

    Called when stopping the daemon to restore original audio routing.
    """
    target = None

    if DEFAULT_SINK_FILE_PATH.exists():
        try:
            target = DEFAULT_SINK_FILE_PATH.read_text().strip()
        except IOError:
            pass

    # If no remembered sink or it's the GPU sink, use fallback
    if not target or target == GPU_SINK_NAME:
        target = select_fallback_sink()

    if target:
        set_default_sink(target)


def setup_audio_routing() -> tuple[bool, str]:
    """Complete audio routing setup for daemon startup.

    Equivalent to daemon.sh's startup sequence:
    1. Create gpu_upsampler_sink (if needed)
    2. Remember current default sink
    3. Set gpu_upsampler_sink as default

    Returns:
        Tuple of (success, message).
    """
    # Step 1: Create sink
    if not create_gpu_sink():
        return False, f"Failed to create {GPU_SINK_NAME}"

    # Step 2: Remember current default
    remember_default_sink()

    # Step 3: Set as default
    if not set_default_sink(GPU_SINK_NAME):
        return False, f"Failed to set {GPU_SINK_NAME} as default"

    current = get_default_sink()
    return True, f"Audio routed to {GPU_SINK_NAME} (default: {current})"
