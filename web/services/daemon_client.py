"""ZeroMQ client for communicating with the C++ audio daemon."""

import zmq

from ..constants import ZEROMQ_IPC_PATH


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
                return False, (
                    response.split(":", 1)[1] if ":" in response else "Unknown error"
                )
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
        import json

        success, response = self.send_command("STATS")
        if success:
            try:
                return True, json.loads(response)
            except json.JSONDecodeError:
                return False, f"Invalid JSON response: {response}"
        return False, response

    def soft_reset(self) -> tuple[bool, str]:
        """Send SOFT_RESET command to daemon."""
        return self.send_command("SOFT_RESET")

    def ping(self) -> bool:
        """
        Ping the daemon to check if it's responding.

        Returns:
            True if daemon responds, False otherwise.
        """
        success, _ = self.send_command("PING")
        return success

    def get_phase_type(self) -> tuple[bool, dict | str]:
        """
        Get current phase type from daemon.

        Returns:
            (success, {"phase_type": "minimum"|"linear"}) on success
            (False, error_message) on failure
        """
        import json

        success, response = self.send_command("PHASE_TYPE_GET")
        if success:
            try:
                return True, json.loads(response)
            except json.JSONDecodeError:
                return False, f"Invalid JSON response: {response}"
        return False, response

    def set_phase_type(self, phase_type: str) -> tuple[bool, str]:
        """
        Set phase type on daemon.

        Args:
            phase_type: "minimum" or "linear"

        Returns:
            (success, message) tuple
        """
        if phase_type not in ("minimum", "linear"):
            return False, f"Invalid phase type: {phase_type}"
        return self.send_command(f"PHASE_TYPE_SET:{phase_type}")


def get_daemon_client(timeout_ms: int = 3000) -> DaemonClient:
    """
    Factory function to get a DaemonClient instance.

    Usage:
        with get_daemon_client() as client:
            success, msg = client.reload_config()
    """
    return DaemonClient(timeout_ms=timeout_ms)
