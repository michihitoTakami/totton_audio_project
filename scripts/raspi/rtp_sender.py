#!/usr/bin/env python3
"""RTP Sender for RasPi simulation (Development/Testing)

このスクリプトはPCをRasPiに見立てて、PipeWireのRTPストリームを
受信し、Magic Boxに中継します。

設計方針:
- フォーマット（サンプルレート、チャンネル数、ビット深度）は環境変数で指定
- PipeWireの設定と同じ値を指定することで、確実に正しいSDPを送信
- RTPパケットからのフォーマット推定は行わない（原理的に不可能なため）

動作フロー:
1. 起動時にMagic BoxへSDP送信（セッション確立）
2. マルチキャストグループ (224.0.0.56:46000) でRTPパケットを受信
3. 受信したRTPパケットをMagic Boxに転送

SDP再送信:
- SIGHUPシグナルを送信すると、SDPを再送信します
- 環境変数を変更後、コンテナ再起動なしでレート変更をテストできます
- 使用方法: docker kill -s HUP raspi-rtp-sender
"""

import logging
import os
import signal
import socket
import struct
import sys

import requests  # type: ignore[import-untyped]

# =============================================================================
# 環境変数から設定を読み込み
# =============================================================================

# RTP受信設定（PipeWireから）
RTP_RECV_MULTICAST_GROUP = os.getenv("RTP_RECV_MULTICAST_GROUP", "224.0.0.56")
RTP_RECV_PORT = int(os.getenv("RTP_RECV_PORT", "46000"))

# Magic Box接続設定
MAGIC_BOX_HOST = os.getenv("MAGIC_BOX_HOST", "192.168.1.10")
MAGIC_BOX_API_PORT = int(os.getenv("MAGIC_BOX_API_PORT", "8000"))
RTP_SEND_PORT = int(os.getenv("RTP_SEND_PORT", "46001"))

# オーディオフォーマット設定（PipeWire設定と一致させること）
SAMPLE_RATE = int(os.getenv("RTP_SAMPLE_RATE", "44100"))
CHANNELS = int(os.getenv("RTP_CHANNELS", "2"))
BITS_PER_SAMPLE = int(os.getenv("RTP_BITS_PER_SAMPLE", "16"))
PAYLOAD_TYPE = int(os.getenv("RTP_PAYLOAD_TYPE", "127"))  # 動的ペイロードタイプ

# セッションID
SESSION_ID = os.getenv("RTP_SESSION_ID", "raspi-uac2")

# ログ設定
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "/var/log/rtp-sender/rtp_sender.log")

# =============================================================================
# ログ設定
# =============================================================================

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
if LOG_LEVEL not in VALID_LOG_LEVELS:
    print(f"Warning: Invalid LOG_LEVEL '{LOG_LEVEL}', using INFO", file=sys.stderr)
    LOG_LEVEL = "INFO"

log_dir = os.path.dirname(LOG_FILE)
if log_dir and not os.path.exists(log_dir):
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(
            f"Warning: Could not create log directory {log_dir}: {e}", file=sys.stderr
        )
        LOG_FILE = ""

handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
if LOG_FILE:
    try:
        handlers.append(logging.FileHandler(LOG_FILE))
    except OSError as e:
        print(f"Warning: Could not open log file {LOG_FILE}: {e}", file=sys.stderr)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)


# =============================================================================
# グローバル状態
# =============================================================================

# SIGHUP受信フラグ（SDP再送信トリガー）
_resend_sdp_requested = False


def _handle_sighup(signum: int, frame: object) -> None:
    """SIGHUPハンドラ: SDP再送信をリクエスト"""
    global _resend_sdp_requested
    _resend_sdp_requested = True
    logger.info("SIGHUP received, will resend SDP")


# =============================================================================
# SDP送信
# =============================================================================


def send_sdp_to_magicbox() -> bool:
    """Magic BoxにSDPを送信してRTPセッションを確立

    Returns:
        成功した場合True
    """
    # SDP生成（RFC 4566準拠）
    sdp = f"""v=0
o=- 0 0 IN IP4 {MAGIC_BOX_HOST}
s=RasPi UAC2 Stream
c=IN IP4 {MAGIC_BOX_HOST}
t=0 0
m=audio {RTP_SEND_PORT} RTP/AVP {PAYLOAD_TYPE}
a=rtpmap:{PAYLOAD_TYPE} L{BITS_PER_SAMPLE}/{SAMPLE_RATE}/{CHANNELS}
"""

    # REST API経由でセッション作成
    api_url = f"http://{MAGIC_BOX_HOST}:{MAGIC_BOX_API_PORT}/api/rtp/sessions"
    payload = {
        "session_id": SESSION_ID,
        "endpoint": {
            "bind_address": "0.0.0.0",
            "port": RTP_SEND_PORT,
        },
        "sdp": sdp.strip(),
    }

    try:
        response = requests.post(api_url, json=payload, timeout=5)
        if response.status_code in [200, 201]:
            logger.info(
                f"Successfully registered RTP session: "
                f"{SAMPLE_RATE}Hz, {CHANNELS}ch, {BITS_PER_SAMPLE}bit, PT{PAYLOAD_TYPE}"
            )
            return True
        else:
            logger.error(
                f"Failed to register RTP session: {response.status_code} - {response.text}"
            )
            return False
    except requests.RequestException as e:
        logger.error(f"Failed to connect to Magic Box API: {e}")
        return False


# =============================================================================
# メイン処理
# =============================================================================


def main() -> None:
    """メイン処理"""
    global _resend_sdp_requested

    logger.info("Starting RTP Sender (RasPi simulation)...")
    logger.info(f"Receiving from: {RTP_RECV_MULTICAST_GROUP}:{RTP_RECV_PORT}")
    logger.info(f"Sending to Magic Box: {MAGIC_BOX_HOST}:{RTP_SEND_PORT}")
    logger.info(
        f"Audio format: {SAMPLE_RATE}Hz, {CHANNELS}ch, {BITS_PER_SAMPLE}bit, PT{PAYLOAD_TYPE}"
    )

    # SIGHUPハンドラを設定（SDP再送信用）
    signal.signal(signal.SIGHUP, _handle_sighup)
    logger.info("SIGHUP handler registered (send SIGHUP to resend SDP)")

    # 起動時にSDPを送信
    if not send_sdp_to_magicbox():
        logger.error("Failed to register RTP session, exiting...")
        sys.exit(1)

    # 受信ソケット作成（マルチキャスト）
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    recv_sock.bind(("", RTP_RECV_PORT))

    # マルチキャストグループに参加
    mreq = struct.pack(
        "4sl", socket.inet_aton(RTP_RECV_MULTICAST_GROUP), socket.INADDR_ANY
    )
    recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    # 送信ソケット作成
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    logger.info("Forwarding RTP packets...")
    packet_count = 0

    try:
        while True:
            # SIGHUP再送信チェック
            if _resend_sdp_requested:
                _resend_sdp_requested = False
                logger.info("Resending SDP due to SIGHUP...")
                if send_sdp_to_magicbox():
                    logger.info("SDP resent successfully")
                else:
                    logger.warning("Failed to resend SDP, continuing...")

            # RTPパケット受信
            data, addr = recv_sock.recvfrom(2048)

            # RTPパケットをMagic Boxに転送
            send_sock.sendto(data, (MAGIC_BOX_HOST, RTP_SEND_PORT))

            packet_count += 1
            if packet_count % 10000 == 0:
                logger.debug(f"Forwarded {packet_count} packets")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        recv_sock.close()
        send_sock.close()
        logger.info(f"Total packets forwarded: {packet_count}")


if __name__ == "__main__":
    main()
