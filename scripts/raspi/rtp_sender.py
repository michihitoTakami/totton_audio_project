#!/usr/bin/env python3
"""RTP Sender for RasPi simulation (Development/Testing)

このスクリプトはPCをRasPiに見立てて、PipeWireのRTPストリームを
受信し、Magic Boxに中継します。

動作フロー:
1. マルチキャストグループ (224.0.0.56:46000) でRTPパケットを受信
2. 最初のパケットからSDPを生成
3. Magic BoxのREST APIにSDP送信してセッション確立
4. 受信したRTPパケットをMagic Boxに転送
"""

import logging
import os
import socket
import struct
import sys
from typing import Optional, Tuple

import requests  # type: ignore[import-untyped]

# 環境変数から設定を読み込み
RTP_RECV_MULTICAST_GROUP = os.getenv("RTP_RECV_MULTICAST_GROUP", "224.0.0.56")
RTP_RECV_PORT = int(os.getenv("RTP_RECV_PORT", "46000"))
MAGIC_BOX_HOST = os.getenv("MAGIC_BOX_HOST", "192.168.1.10")
MAGIC_BOX_API_PORT = int(os.getenv("MAGIC_BOX_API_PORT", "8000"))
RTP_SEND_PORT = int(os.getenv("RTP_SEND_PORT", "46001"))
AUTO_REGISTER = os.getenv("AUTO_REGISTER", "true").lower() == "true"
SEND_SDP_ON_RATE_CHANGE = os.getenv("SEND_SDP_ON_RATE_CHANGE", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "/var/log/rtp-sender/rtp_sender.log")

# ログディレクトリを自動作成
log_dir = os.path.dirname(LOG_FILE)
if log_dir and not os.path.exists(log_dir):
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(
            f"Warning: Could not create log directory {log_dir}: {e}", file=sys.stderr
        )
        print("Logging to stdout only", file=sys.stderr)
        LOG_FILE = None

# ログ設定
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


def parse_rtp_header(packet: bytes) -> Optional[dict]:
    """RTPヘッダーをパース

    Args:
        packet: RTPパケット（バイト列）

    Returns:
        パース結果の辞書、またはNone（失敗時）
    """
    if len(packet) < 12:
        return None

    # RTPヘッダー (RFC 3550)
    #  0                   1                   2                   3
    #  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |V=2|P|X|  CC   |M|     PT      |       sequence number         |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |                           timestamp                           |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |           synchronization source (SSRC) identifier            |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    byte0, byte1 = struct.unpack("!BB", packet[0:2])
    version = (byte0 & 0b11000000) >> 6
    padding = (byte0 & 0b00100000) != 0
    extension = (byte0 & 0b00010000) != 0
    csrc_count = byte0 & 0b00001111
    marker = (byte1 & 0b10000000) != 0
    payload_type = byte1 & 0b01111111

    if version != 2:
        return None

    seq_num, timestamp, ssrc = struct.unpack("!HII", packet[2:12])

    return {
        "version": version,
        "padding": padding,
        "extension": extension,
        "csrc_count": csrc_count,
        "marker": marker,
        "payload_type": payload_type,
        "sequence_number": seq_num,
        "timestamp": timestamp,
        "ssrc": ssrc,
        "header_size": 12 + csrc_count * 4,
    }


def detect_format_from_rtp(packets: list) -> Tuple[int, int, int]:
    """複数のRTPパケットからフォーマットを推定

    Args:
        packets: RTPパケットのリスト

    Returns:
        (sample_rate, channels, bits_per_sample)
    """
    if not packets:
        return 44100, 2, 16

    # 最初のパケットを解析
    header = parse_rtp_header(packets[0])
    if not header:
        return 44100, 2, 16

    payload_size = len(packets[0]) - header["header_size"]

    # L16フォーマット推定 (Big Endian, 16bit)
    bits_per_sample = 16

    # チャンネル数推定（仮定: ステレオ）
    channels = 2

    # サンプルレート推定
    # RTPタイムスタンプの差からサンプルレートを逆算
    # L16の場合、RTP clock rate = sample rate なので、
    # ts_diff = パケット間のサンプル数
    if len(packets) >= 2:
        header2 = parse_rtp_header(packets[1])
        if header2:
            ts_diff = header2["timestamp"] - header["timestamp"]
            # ペイロードから実際のサンプル数を計算
            samples_in_packet = payload_size // (channels * (bits_per_sample // 8))

            if samples_in_packet > 0 and ts_diff > 0:
                # タイムスタンプ差 ≈ パケット内サンプル数 であるべき
                # タイムスタンプ差をそのままサンプルレートの候補として扱う
                # 複数パケットから平均を取るのが理想だが、ここでは ts_diff を基準にする

                # RTPのタイムスタンプはclock rate単位なので、
                # 実際のサンプルレートはパケット送信間隔から推定
                # 一般的なレートとの最近傍マッチング
                common_rates = [44100, 48000, 88200, 96000, 176400, 192000]

                # ts_diff が非常に小さい場合（例: 480サンプル/10ms @ 48kHz）、
                # これは1パケットあたりのサンプル数を示している
                # 実際のサンプルレートは、パケット送信レートから推定する必要がある

                # ヒューリスティック: ts_diff が一般的なレートより小さい場合は
                # パケット間隔から推定（例: 480サンプル/パケット → 48000Hz）
                if ts_diff < 10000:
                    # パケット間隔を仮定（典型的には10ms）
                    # ts_diff サンプル/10ms → ts_diff * 100 サンプル/秒
                    estimated_rate = ts_diff * 100
                    sample_rate = min(
                        common_rates, key=lambda x: abs(x - estimated_rate)
                    )
                else:
                    # ts_diff が大きい場合は直接使用
                    sample_rate = min(common_rates, key=lambda x: abs(x - ts_diff))

                return sample_rate, channels, bits_per_sample

    # デフォルト
    return 44100, 2, 16


def send_sdp_to_magicbox(
    sample_rate: int, channels: int, bits_per_sample: int, payload_type: int
) -> bool:
    """Magic BoxにSDPを送信してRTPセッションを確立

    Args:
        sample_rate: サンプリングレート
        channels: チャンネル数
        bits_per_sample: ビット深度
        payload_type: ペイロードタイプ

    Returns:
        成功した場合True
    """
    # SDP生成
    # c=行の形式: c=IN IP4 <address> または c=IN IP4 <address>/<ttl> (マルチキャストの場合)
    # ポート番号はm=行で指定する
    sdp = f"""v=0
o=- 0 0 IN IP4 {MAGIC_BOX_HOST}
s=RasPi UAC2 Stream
c=IN IP4 {MAGIC_BOX_HOST}
t=0 0
m=audio {RTP_SEND_PORT} RTP/AVP {payload_type}
a=rtpmap:{payload_type} L{bits_per_sample}/{sample_rate}/{channels}
"""

    # REST API経由でセッション作成
    api_url = f"http://{MAGIC_BOX_HOST}:{MAGIC_BOX_API_PORT}/api/rtp/sessions"
    payload = {
        "session_id": "raspi-uac2",
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
                f"Successfully registered RTP session: {sample_rate}Hz, {channels}ch, {bits_per_sample}bit"
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


def main():
    """メイン処理"""
    logger.info("Starting RTP Sender (RasPi simulation)...")
    logger.info(f"Receiving from: {RTP_RECV_MULTICAST_GROUP}:{RTP_RECV_PORT}")
    logger.info(f"Sending to Magic Box: {MAGIC_BOX_HOST}:{RTP_SEND_PORT}")

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

    # フォーマット検出用のバッファ
    detection_packets = []
    detected = False
    current_sample_rate = None
    packet_count = 0
    rate_check_interval = 1000  # 1000パケットごとにレート変更をチェック

    logger.info("Waiting for RTP packets...")

    try:
        while True:
            # RTPパケット受信
            data, addr = recv_sock.recvfrom(2048)
            packet_count += 1

            # 初回フォーマット検出
            if not detected and len(detection_packets) < 10:
                detection_packets.append(data)
                if len(detection_packets) == 10:
                    sample_rate, channels, bits_per_sample = detect_format_from_rtp(
                        detection_packets
                    )
                    header = parse_rtp_header(data)
                    payload_type = header["payload_type"] if header else 127

                    logger.info(
                        f"Detected format: {sample_rate}Hz, {channels}ch, {bits_per_sample}bit, PT{payload_type}"
                    )

                    if AUTO_REGISTER:
                        if send_sdp_to_magicbox(
                            sample_rate, channels, bits_per_sample, payload_type
                        ):
                            detected = True
                            current_sample_rate = sample_rate
                        else:
                            logger.warning("Failed to register session, will retry...")
                            detection_packets = []  # リトライのためリセット

            # 動的レート変更検出（SEND_SDP_ON_RATE_CHANGE が有効な場合）
            elif (
                detected
                and SEND_SDP_ON_RATE_CHANGE
                and packet_count % rate_check_interval == 0
            ):
                # 定期的に最新のパケットバッファでレート再推定
                recent_packets = (
                    detection_packets[-10:]
                    if len(detection_packets) >= 10
                    else detection_packets
                )
                if len(recent_packets) >= 2:
                    new_rate, channels, bits_per_sample = detect_format_from_rtp(
                        recent_packets
                    )
                    header = parse_rtp_header(data)
                    payload_type = header["payload_type"] if header else 127

                    if new_rate != current_sample_rate:
                        logger.info(
                            f"Rate change detected: {current_sample_rate}Hz → {new_rate}Hz"
                        )
                        if send_sdp_to_magicbox(
                            new_rate, channels, bits_per_sample, payload_type
                        ):
                            current_sample_rate = new_rate
                            logger.info("SDP updated successfully")
                        else:
                            logger.warning("Failed to update SDP")

            # パケットバッファを更新（直近10パケットを保持）
            if detected:
                detection_packets.append(data)
                if len(detection_packets) > 10:
                    detection_packets.pop(0)

            # RTPパケットをMagic Boxに転送
            if detected:
                send_sock.sendto(data, (MAGIC_BOX_HOST, RTP_SEND_PORT))

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        recv_sock.close()
        send_sock.close()


if __name__ == "__main__":
    main()
