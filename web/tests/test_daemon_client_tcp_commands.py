"""DaemonClient TCP input command tests (#685)."""

from unittest.mock import patch

from web.error_codes import ErrorCode
from web.models import (
    TcpInputConfigUpdate,
    TcpInputSettings,
    TcpInputStatusResponse,
    TcpInputTelemetry,
)
from web.services.daemon_client import DaemonClient, DaemonResponse


class TestTcpInputCommands:
    """TCP入力用ZeroMQコマンドの挙動を検証するテスト群."""

    def test_tcp_input_start_and_stop(self):
        """START/STOPコマンドを正しく送信する."""
        client = DaemonClient()
        with patch.object(client, "send_json_command_v2") as mock_send:
            mock_send.return_value = DaemonResponse(success=True, data={"status": "ok"})

            start_resp = client.tcp_input_start()
            stop_resp = client.tcp_input_stop()

            assert mock_send.call_count == 2
            mock_send.assert_any_call("TCP_INPUT_START")
            mock_send.assert_any_call("TCP_INPUT_STOP")
            assert start_resp.success is True
            assert stop_resp.success is True

    def test_tcp_input_status_parses_telemetry(self):
        """テレメトリペイロードをモデルへ正規化する."""
        client = DaemonClient()
        payload = {
            "listening": True,
            "bound_port": 46001,
            "client_connected": True,
            "streaming": True,
            "xrun_count": 1,
            "ring_buffer_frames": 8192,
            "watermark_frames": 4096,
            "buffered_frames": 2048,
            "max_buffered_frames": 8192,
            "dropped_frames": 0,
            "connection_mode": "priority",
            "priority_clients": ["10.0.0.1"],
            "last_header": {
                "sample_rate": 48000,
                "channels": 2,
                "format": 1,
                "version": 1,
            },
        }

        with patch.object(client, "send_json_command_v2") as mock_send:
            mock_send.return_value = DaemonResponse(success=True, data=payload)

            response = client.tcp_input_status()

        mock_send.assert_called_once_with("TCP_INPUT_STATUS")
        assert response.success is True
        assert isinstance(response.data, TcpInputTelemetry)
        assert response.data.listening is True
        assert response.data.bound_port == 46001
        assert response.data.connection_mode == "priority"
        assert response.data.last_header is not None
        assert response.data.last_header.format == "S16_LE"

    def test_tcp_input_status_with_settings_returns_status_model(self):
        """settings+telemetryをまとめてTCP Input statusモデルにする."""
        client = DaemonClient()
        payload = {
            "settings": {
                "enabled": True,
                "bind_address": "0.0.0.0",
                "port": 47000,
                "buffer_size_bytes": 262144,
                "connection_mode": "single",
                "priority_clients": [],
            },
            "telemetry": {
                "listening": True,
                "bound_port": 47000,
                "priority_clients": [],
                "rep_endpoint": "ipc:///tmp/tcp.rep",
            },
        }

        with patch.object(client, "send_json_command_v2") as mock_send:
            mock_send.return_value = DaemonResponse(success=True, data=payload)

            response = client.tcp_input_status()

        mock_send.assert_called_once_with("TCP_INPUT_STATUS")
        assert response.success is True
        assert isinstance(response.data, TcpInputStatusResponse)
        status: TcpInputStatusResponse = response.data
        assert isinstance(status.settings, TcpInputSettings)
        assert status.settings.port == 47000
        assert status.telemetry.bound_port == 47000
        assert status.telemetry.rep_endpoint == "ipc:///tmp/tcp.rep"

    def test_tcp_input_config_update_validates_and_serializes(self):
        """設定更新のバリデーションとエイリアス変換を確認する."""
        client = DaemonClient()
        update = TcpInputConfigUpdate(
            bind_address="127.0.0.1",
            port=5002,
            buffer_size_bytes=131072,
            connection_mode="priority",
            priority_clients=["10.0.0.1"],
        )

        with patch.object(client, "send_json_command_v2") as mock_send:
            mock_send.return_value = DaemonResponse(success=True, data={})

            response = client.tcp_input_config_update(update)

            mock_send.assert_called_once_with(
                "TCP_INPUT_CONFIG_UPDATE",
                {
                    "bind_address": "127.0.0.1",
                    "port": 5002,
                    "buffer_size_bytes": 131072,
                    "connection_mode": "priority",
                    "priority_clients": ["10.0.0.1"],
                },
            )
            assert response.success is True

        # dict入力のバリデーションと空入力エラー
        with patch.object(client, "send_json_command_v2") as mock_send:
            mock_send.return_value = DaemonResponse(success=True, data={})

            response_dict = client.tcp_input_config_update(
                {"bindAddress": "0.0.0.0", "bufferSizeBytes": 65536}
            )

            mock_send.assert_called_once_with(
                "TCP_INPUT_CONFIG_UPDATE",
                {"bind_address": "0.0.0.0", "buffer_size_bytes": 65536},
            )
            assert response_dict.success is True

            mock_send.reset_mock()
            error_response = client.tcp_input_config_update({})
            mock_send.assert_not_called()
            assert error_response.success is False
            assert error_response.error is not None
            assert error_response.error.error_code == ErrorCode.IPC_INVALID_PARAMS.value
