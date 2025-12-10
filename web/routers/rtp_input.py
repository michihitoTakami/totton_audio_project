"""RTP入力制御用エンドポイント."""

from fastapi import APIRouter, Depends, HTTPException

from ..models import ApiResponse, RtpInputConfigUpdate, RtpInputSettings, RtpInputStatus
from ..services.rtp_input import RtpReceiverManager, get_rtp_receiver_manager

router = APIRouter(prefix="/api/rtp-input", tags=["rtp-input"])


@router.get(
    "/status",
    response_model=RtpInputStatus,
    summary="RTP入力ステータス取得",
)
async def rtp_input_status(
    manager: RtpReceiverManager = Depends(get_rtp_receiver_manager),
):
    """RTP入力の稼働状況を返す."""
    return await manager.status()


@router.post(
    "/start",
    response_model=ApiResponse,
    summary="RTP入力を開始",
)
async def rtp_input_start(
    manager: RtpReceiverManager = Depends(get_rtp_receiver_manager),
):
    """GStreamerベースのRTP受信を開始する."""
    try:
        await manager.start()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Failed to start RTP receiver: {exc}"
        ) from exc
    return ApiResponse(success=True, message="RTP input started")


@router.post(
    "/stop",
    response_model=ApiResponse,
    summary="RTP入力を停止",
)
async def rtp_input_stop(
    manager: RtpReceiverManager = Depends(get_rtp_receiver_manager),
):
    """RTP受信を停止する."""
    await manager.stop()
    return ApiResponse(success=True, message="RTP input stopped")


@router.put(
    "/config",
    response_model=RtpInputSettings,
    summary="RTP入力設定を更新",
)
async def rtp_input_config_update(
    request: RtpInputConfigUpdate,
    manager: RtpReceiverManager = Depends(get_rtp_receiver_manager),
):
    """RTP入力設定を更新する（再起動は自動では行わない）。"""
    try:
        return await manager.apply_config(request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid config: {exc}") from exc


__all__ = ["router"]
