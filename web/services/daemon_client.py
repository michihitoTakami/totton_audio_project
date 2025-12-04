"""ZeroMQ client for communicating with the C++ audio daemon.

This client prefers ``ZMQ_ENDPOINT`` if set; otherwise it falls back to ``web.constants.ZEROMQ_IPC_PATH``. Pass an explicit endpoint to override both.
"""

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any

import zmq

from ..constants import (
    PHASE_TYPE_LINEAR,
    PHASE_TYPE_MINIMUM,
    ZEROMQ_IPC_PATH,
)
from ..error_codes import ErrorCode, get_error_mapping


def _get_default_endpoint() -> str:
    """Resolve default ZeroMQ endpoint using env override."""
    return os.environ.get("ZMQ_ENDPOINT", ZEROMQ_IPC_PATH)


@dataclass
class DaemonError(Exception):
    """Exception raised when daemon returns an error.

    Captures structured error information from C++ Audio Engine,
    including error code and inner_error details for debugging.

    Attributes:
        error_code: Application error code (e.g., "DAC_RATE_NOT_SUPPORTED")
        message: Human-readable error message
        inner_error: Optional nested error details from lower layers
    """

    error_code: str
    message: str
    inner_error: dict[str, Any] | None = field(default=None)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    @property
    def http_status(self) -> int:
        """Get HTTP status code for this error."""
        return get_error_mapping(self.error_code).http_status

    @property
    def category(self) -> str:
        """Get error category."""
        return get_error_mapping(self.error_code).category.value

    @property
    def title(self) -> str:
        """Get error title."""
        return get_error_mapping(self.error_code).title


@dataclass
class DaemonResponse:
    """Structured response from daemon.

    Attributes:
        success: Whether the command succeeded
        data: Response data (for successful commands)
        error: DaemonError if command failed
    """

    success: bool
    data: Any = None
    error: DaemonError | None = None


class DaemonClient:
    """
    ZeroMQ client for communicating with the C++ audio daemon.
    Uses REQ/REP pattern over IPC socket.

    Thread Safety: Each DaemonClient instance should be used by a single
    thread/coroutine at a time. Use the context manager or create new
    instances per request for concurrent access.

    Response Format (JSON from C++):
        Success: {"status": "ok", "data": ...}
        Error: {
            "status": "error",
            "error_code": "DAC_RATE_NOT_SUPPORTED",
            "message": "Sample rate not supported",
            "inner_error": {"cpp_code": "0x2004", "alsa_errno": -22}
        }
    """

    def __init__(self, endpoint: str | None = None, timeout_ms: int = 3000):
        self.endpoint = endpoint if endpoint is not None else _get_default_endpoint()
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

    def _parse_json_response(self, response: str) -> DaemonResponse:
        """Parse JSON response from daemon.

        Expected format:
            Success: {"status": "ok", "data": ...}
            Error: {"status": "error", "error_code": "...", "message": "...", "inner_error": {...}}
        """
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: treat as legacy text response
            return self._parse_legacy_response(response)

        # Guard: JSON must be a dict with "status" field
        if not isinstance(data, dict):
            error = DaemonError(
                error_code=ErrorCode.IPC_PROTOCOL_ERROR.value,
                message=f"Expected JSON object, got {type(data).__name__}",
            )
            return DaemonResponse(success=False, error=error)

        status = data.get("status", "")

        if status == "ok":
            return DaemonResponse(success=True, data=data.get("data"))
        elif status == "error":
            error = DaemonError(
                error_code=data.get("error_code", "IPC_PROTOCOL_ERROR"),
                message=data.get("message", "Unknown error"),
                inner_error=data.get("inner_error"),
            )
            return DaemonResponse(success=False, error=error)
        else:
            # Unknown status - treat as protocol error
            error = DaemonError(
                error_code=ErrorCode.IPC_PROTOCOL_ERROR.value,
                message=f"Unknown response status: {status}",
            )
            return DaemonResponse(success=False, error=error)

    def _parse_legacy_response(self, response: str) -> DaemonResponse:
        """Parse legacy text response format.

        Legacy format: "OK" or "OK:data" or "ERR:message"
        """
        if response.startswith("OK"):
            if ":" in response:
                return DaemonResponse(success=True, data=response.split(":", 1)[1])
            return DaemonResponse(success=True, data="Command executed")
        elif response.startswith("ERR"):
            msg = response.split(":", 1)[1] if ":" in response else "Unknown error"
            error = DaemonError(
                error_code=ErrorCode.IPC_PROTOCOL_ERROR.value,
                message=msg,
            )
            return DaemonResponse(success=False, error=error)
        else:
            error = DaemonError(
                error_code=ErrorCode.IPC_PROTOCOL_ERROR.value,
                message=f"Unexpected response: {response}",
            )
            return DaemonResponse(success=False, error=error)

    def send_command(self, command: str) -> tuple[bool, str]:
        """
        Send a command to the daemon and wait for response.

        Returns:
            (success, message) tuple for backward compatibility.
            Use send_command_v2() for structured error handling.
        """
        result = self.send_command_v2(command)
        if result.success:
            if isinstance(result.data, dict):
                return True, json.dumps(result.data)
            return True, str(result.data) if result.data else "Command executed"
        else:
            return False, str(result.error) if result.error else "Unknown error"

    def send_command_v2(self, command: str) -> DaemonResponse:
        """
        Send a command to the daemon and return structured response.

        Returns:
            DaemonResponse with success/data or error details.

        Raises:
            No exceptions - all errors are returned in DaemonResponse.
        """
        try:
            socket = self._ensure_connected()
            socket.send_string(command)
            response = socket.recv_string()
            return self._parse_json_response(response)

        except zmq.Again:
            # Timeout - daemon not responding
            self.close()
            error = DaemonError(
                error_code=ErrorCode.IPC_TIMEOUT.value,
                message="Daemon not responding (timeout)",
            )
            return DaemonResponse(success=False, error=error)

        except zmq.ZMQError as e:
            self.close()
            # Determine appropriate error code
            if e.errno == zmq.ECONNREFUSED:
                error_code = ErrorCode.IPC_DAEMON_NOT_RUNNING.value
            else:
                error_code = ErrorCode.IPC_CONNECTION_FAILED.value
            error = DaemonError(
                error_code=error_code,
                message=f"ZeroMQ error: {e}",
                inner_error={"zmq_errno": e.errno},
            )
            return DaemonResponse(success=False, error=error)

    def reload_config(self) -> tuple[bool, str]:
        """Send RELOAD command to daemon."""
        return self.send_command("RELOAD")

    def get_stats(self) -> tuple[bool, dict | str]:
        """Send STATS command and parse JSON response."""
        result = self.send_command_v2("STATS")
        if result.success:
            if isinstance(result.data, dict):
                return True, result.data
            # Try to parse data as JSON if it's a string
            if isinstance(result.data, str):
                try:
                    return True, json.loads(result.data)
                except json.JSONDecodeError:
                    return False, f"Invalid JSON response: {result.data}"
            return True, {}
        return False, str(result.error) if result.error else "Unknown error"

    def soft_reset(self) -> tuple[bool, str]:
        """Send SOFT_RESET command to daemon."""
        return self.send_command("SOFT_RESET")

    def ping(self) -> bool:
        """
        Ping the daemon to check if it's responding.

        Returns:
            True if daemon responds, False otherwise.
        """
        result = self.send_command_v2("PING")
        return result.success

    def get_phase_type(self) -> tuple[bool, dict | str]:
        """
        Get current phase type from daemon.

        Returns:
            (success, {"phase_type": "minimum"|"linear"}) on success
            (False, error_message) on failure
        """
        result = self.send_command_v2("PHASE_TYPE_GET")
        if result.success:
            if isinstance(result.data, dict):
                phase = result.data.get("phase_type")
                if isinstance(phase, str):
                    result.data["phase_type"] = (
                        phase if phase in ["minimum", "linear"] else PHASE_TYPE_MINIMUM
                    )
                return True, result.data
            if isinstance(result.data, str):
                try:
                    data = json.loads(result.data)
                    if isinstance(data, dict):
                        phase = data.get("phase_type")
                        if isinstance(phase, str):
                            data["phase_type"] = (
                                phase
                                if phase in ["minimum", "linear"]
                                else PHASE_TYPE_MINIMUM
                            )
                    return True, data
                except json.JSONDecodeError:
                    return False, f"Invalid JSON response: {result.data}"
            return True, {}
        return False, str(result.error) if result.error else "Unknown error"

    def set_phase_type(self, phase_type: str) -> tuple[bool, str]:
        """
        Set phase type on daemon.

        Args:
            phase_type: "minimum" or "linear"

        Returns:
            (success, message) tuple
        """
        normalized = str(phase_type).lower()
        allowed = {PHASE_TYPE_MINIMUM, PHASE_TYPE_LINEAR}
        if normalized not in allowed:
            return False, f"Invalid phase type: {phase_type}"
        return self.send_command(f"PHASE_TYPE_SET:{normalized}")

    # ========== JSON Command Methods (#150) ==========

    def send_json_command_v2(
        self, cmd: str, params: dict[str, Any] | None = None
    ) -> DaemonResponse:
        """
        Send a JSON-formatted command to the daemon and return structured response.

        Args:
            cmd: Command type (e.g., "CROSSFEED_ENABLE")
            params: Optional parameters dict

        Returns:
            DaemonResponse with success/data or error details.
        """
        try:
            socket = self._ensure_connected()
            request: dict[str, Any] = {"cmd": cmd}
            if params:
                request["params"] = params
            socket.send_string(json.dumps(request))
            response = socket.recv_string()
            return self._parse_json_response(response)

        except zmq.Again:
            self.close()
            error = DaemonError(
                error_code=ErrorCode.IPC_TIMEOUT.value,
                message="Daemon not responding (timeout)",
            )
            return DaemonResponse(success=False, error=error)

        except zmq.ZMQError as e:
            self.close()
            if e.errno == zmq.ECONNREFUSED:
                error_code = ErrorCode.IPC_DAEMON_NOT_RUNNING.value
            else:
                error_code = ErrorCode.IPC_CONNECTION_FAILED.value
            error = DaemonError(
                error_code=error_code,
                message=f"ZeroMQ error: {e}",
                inner_error={"zmq_errno": e.errno},
            )
            return DaemonResponse(success=False, error=error)

    # ========== DAC Commands ==========

    def dac_list_devices(self) -> DaemonResponse:
        """List runtime DAC devices from the daemon."""
        return self.send_json_command_v2("DAC_LIST")

    def dac_status(self) -> DaemonResponse:
        """Get current DAC state from the daemon."""
        return self.send_json_command_v2("DAC_STATUS")

    def dac_select(self, device: str) -> DaemonResponse:
        """Request that the daemon switch to the specified ALSA device."""
        return self.send_json_command_v2("DAC_SELECT", {"device": device})

    def dac_rescan(self) -> DaemonResponse:
        """Trigger a DAC rescan on the daemon."""
        return self.send_json_command_v2("DAC_RESCAN")

    # ========== Crossfeed Commands (#150) ==========

    def crossfeed_enable(self) -> DaemonResponse:
        """
        Enable crossfeed (HRTF) processing.

        Returns:
            DaemonResponse with success/data or error details.
        """
        return self.send_json_command_v2("CROSSFEED_ENABLE")

    def crossfeed_disable(self) -> DaemonResponse:
        """
        Disable crossfeed processing.

        Returns:
            DaemonResponse with success/data or error details.
        """
        return self.send_json_command_v2("CROSSFEED_DISABLE")

    def crossfeed_set_combined(
        self,
        rate_family: str,
        combined_ll: bytes,
        combined_lr: bytes,
        combined_rl: bytes,
        combined_rr: bytes,
    ) -> DaemonResponse:
        """
        Set combined crossfeed filter coefficients.

        Args:
            rate_family: "44k" or "48k"
            combined_ll: Left-to-Left filter (raw bytes, will be Base64 encoded)
            combined_lr: Left-to-Right filter
            combined_rl: Right-to-Left filter
            combined_rr: Right-to-Right filter

        Returns:
            DaemonResponse with success/data or error details.
        """
        if rate_family not in ("44k", "48k"):
            error = DaemonError(
                error_code=ErrorCode.IPC_INVALID_PARAMS.value,
                message=f"Invalid rate family: {rate_family}",
            )
            return DaemonResponse(success=False, error=error)

        params = {
            "rate_family": rate_family,
            "combined_ll": base64.b64encode(combined_ll).decode("ascii"),
            "combined_lr": base64.b64encode(combined_lr).decode("ascii"),
            "combined_rl": base64.b64encode(combined_rl).decode("ascii"),
            "combined_rr": base64.b64encode(combined_rr).decode("ascii"),
        }
        return self.send_json_command_v2("CROSSFEED_SET_COMBINED", params)

    def crossfeed_get_status(self) -> DaemonResponse:
        """
        Get crossfeed status.

        Returns:
            DaemonResponse with data containing:
                enabled: bool - Whether crossfeed is enabled
                initialized: bool - Whether HRTF processor is initialized
                head_size: str - Current head size setting
                headphone: str - Current headphone model
        """
        return self.send_json_command_v2("CROSSFEED_GET_STATUS")

    def crossfeed_set_size(self, head_size: str) -> DaemonResponse:
        """
        Set crossfeed head size.

        Args:
            head_size: Head size to set ('xs', 's', 'm', 'l', or 'xl')

        Returns:
            DaemonResponse with success/data or error details.
        """
        if head_size.lower() not in ("xs", "s", "m", "l", "xl"):
            error = DaemonError(
                error_code=ErrorCode.IPC_INVALID_PARAMS.value,
                message=f"Invalid head size: {head_size}",
            )
            return DaemonResponse(success=False, error=error)

        params = {"head_size": head_size.lower()}
        return self.send_json_command_v2("CROSSFEED_SET_SIZE", params)

    # ========== RTP Session Commands (#359) ==========

    def rtp_start_session(self, params: dict[str, Any]) -> DaemonResponse:
        """Start an RTP session with the provided SessionConfig parameters."""
        return self.send_json_command_v2("RTP_START_SESSION", params)

    def rtp_stop_session(self, session_id: str) -> DaemonResponse:
        """Stop a running RTP session."""
        return self.send_json_command_v2("RTP_STOP_SESSION", {"session_id": session_id})

    def rtp_list_sessions(self) -> DaemonResponse:
        """List current RTP sessions and telemetry."""
        return self.send_json_command_v2("RTP_LIST_SESSIONS")

    def rtp_get_session(self, session_id: str) -> DaemonResponse:
        """Retrieve metrics for a single RTP session."""
        return self.send_json_command_v2("RTP_GET_SESSION", {"session_id": session_id})

    def rtp_discover_streams(self) -> DaemonResponse:
        """Trigger a short-lived scan for available RTP senders."""
        return self.send_json_command_v2("RTP_DISCOVER_STREAMS")


def get_daemon_client(
    timeout_ms: int = 3000, endpoint: str | None = None
) -> DaemonClient:
    """
    Factory function to get a DaemonClient instance.

    Usage:
        with get_daemon_client() as client:
            success, msg = client.reload_config()

    Args:
        timeout_ms: Socket timeout in milliseconds.
        endpoint: Optional explicit ZeroMQ endpoint (overrides `ZMQ_ENDPOINT`).
    """
    return DaemonClient(endpoint=endpoint, timeout_ms=timeout_ms)
