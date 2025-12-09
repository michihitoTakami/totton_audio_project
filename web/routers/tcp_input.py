"""TCP入力用RESTエンドポイント (#686)."""

from fastapi import APIRouter, HTTPException

from ..error_codes import ErrorCode
from ..models import (
    ApiResponse,
    TcpInputConfigUpdate,
    TcpInputSettings,
    TcpInputStatusResponse,
    TcpInputTelemetry,
)
from ..services import get_daemon_client
from ..services.daemon_client import DaemonError, DaemonResponse
from ..services.tcp_input import parse_tcp_telemetry

router = APIRouter(prefix="/api/tcp-input", tags=["tcp-input"])

_CONNECTION_ERRORS = {
    ErrorCode.IPC_CONNECTION_FAILED.value,
    ErrorCode.IPC_DAEMON_NOT_RUNNING.value,
    ErrorCode.IPC_TIMEOUT.value,
}


def _raise_for_daemon_error(error: DaemonError) -> None:
    """Mapデーモン接続系エラーを502、それ以外は標準ハンドラへ."""
    if error.error_code in _CONNECTION_ERRORS:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": error.error_code,
                "message": error.message,
            },
        )
    raise error


def _unwrap_response(response: DaemonResponse):
    """成功ならdataを返し、失敗なら適切なHTTPエラーに変換する."""
    if response.success:
        return response.data
    if response.error:
        _raise_for_daemon_error(response.error)
    raise HTTPException(status_code=502, detail="Failed to communicate with daemon")


def _coerce_status_payload(payload: object) -> TcpInputStatusResponse:
    """様々な戻り値を統一したステータスモデルに変換する."""
    if isinstance(payload, TcpInputStatusResponse):
        return payload
    if isinstance(payload, TcpInputTelemetry):
        return TcpInputStatusResponse(settings=TcpInputSettings(), telemetry=payload)
    if isinstance(payload, dict):
        telemetry_payload = payload.get("telemetry", payload)
        telemetry = parse_tcp_telemetry(
            telemetry_payload if isinstance(telemetry_payload, dict) else None
        )
        settings_payload = payload.get("settings")
        try:
            settings = (
                TcpInputSettings.model_validate(settings_payload)
                if isinstance(settings_payload, dict)
                else TcpInputSettings()
            )
        except Exception:
            settings = TcpInputSettings()
        return TcpInputStatusResponse(settings=settings, telemetry=telemetry)
    return TcpInputStatusResponse(
        settings=TcpInputSettings(), telemetry=TcpInputTelemetry()
    )


@router.get(
    "/status",
    response_model=TcpInputStatusResponse,
    summary="TCP入力ステータス・テレメトリ取得",
)
async def tcp_input_status():
    """ZeroMQ経由でTCP入力のステータスを取得する."""
    with get_daemon_client(timeout_ms=1500) as client:
        response = client.tcp_input_status()
    payload = _unwrap_response(response)
    return _coerce_status_payload(payload)


@router.post(
    "/start",
    response_model=ApiResponse,
    summary="TCP入力を開始",
)
async def tcp_input_start():
    """デーモンにTCP入力開始を要求する."""
    with get_daemon_client(timeout_ms=2000) as client:
        response = client.tcp_input_start()
    _unwrap_response(response)
    return ApiResponse(success=True, message="TCP input started")


@router.post(
    "/stop",
    response_model=ApiResponse,
    summary="TCP入力を停止",
)
async def tcp_input_stop():
    """デーモンにTCP入力停止を要求する."""
    with get_daemon_client(timeout_ms=2000) as client:
        response = client.tcp_input_stop()
    _unwrap_response(response)
    return ApiResponse(success=True, message="TCP input stopped")


@router.put(
    "/config",
    response_model=ApiResponse,
    summary="TCP入力設定を更新",
)
async def tcp_input_config_update(request: TcpInputConfigUpdate):
    """TCP入力設定をバリデートしてデーモンに反映する."""
    with get_daemon_client(timeout_ms=3000) as client:
        response = client.tcp_input_config_update(request)
    data = _unwrap_response(response)
    return ApiResponse(
        success=True,
        message="TCP input config updated",
        data=data if data is not None else None,
    )
