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
        packets: RTPパケットのリスト（最低2パケット推奨）

    Returns:
        (sample_rate, channels, bits_per_sample)
    """
    if len(packets) < 2:
        return 44100, 2, 16

    # 最初のパケットを解析
    header = parse_rtp_header(packets[0])
    if not header:
        return 44100, 2, 16

    payload_size = len(packets[0]) - header["header_size"]

    # チャンネル数とビット深度の候補
    # 一般的な組み合わせを試行
    format_candidates = [
        (2, 16),  # ステレオ 16bit (最も一般的)
        (2, 24),  # ステレオ 24bit
        (2, 32),  # ステレオ 32bit
        (1, 16),  # モノラル 16bit
        (1, 24),  # モノラル 24bit
    ]

    # サンプルレート候補
    common_rates = [44100, 48000, 88200, 96000, 176400, 192000]

    # 複数パケットからタイムスタンプ差を収集
    ts_diffs = []
    for i in range(1, min(len(packets), 5)):  # 最大4ペアを解析
        h1 = parse_rtp_header(packets[i - 1])
        h2 = parse_rtp_header(packets[i])
        if h1 and h2:
            # タイムスタンプのラップアラウンドに対応
            ts1, ts2 = h1["timestamp"], h2["timestamp"]
            if ts2 >= ts1:
                ts_diff = ts2 - ts1
            else:
                # ラップアラウンド発生
                ts_diff = (0x100000000 - ts1) + ts2

            if ts_diff > 0 and ts_diff < 100000:  # 異常値を除外
                ts_diffs.append(ts_diff)

    if not ts_diffs:
        return 44100, 2, 16

    # タイムスタンプ差の平均（= パケットあたりのサンプル数）
    avg_ts_diff = sum(ts_diffs) // len(ts_diffs)

    # 各フォーマット候補について、一貫性をチェック
    best_match = None
    best_score = float("inf")

    for channels, bits_per_sample in format_candidates:
        bytes_per_sample = bits_per_sample // 8
        # このフォーマットでのパケットあたりのサンプル数
        samples_in_packet = payload_size // (channels * bytes_per_sample)

        if samples_in_packet <= 0:
            continue

        # タイムスタンプ差がサンプル数と一致するか確認
        # L16/L24の場合、RTP clock rate = sample rateなので、
        # ts_diff ≈ samples_in_packet のはず
        if abs(avg_ts_diff - samples_in_packet) > samples_in_packet * 0.1:
            # 10%以上のずれは不整合
            continue

        # このフォーマットが妥当なので、サンプルレートを推定
        # 一般的なパケット間隔（5ms, 10ms, 20ms）を想定
        packet_intervals_ms = [5, 10, 20, 2.5]  # ms

        for interval_ms in packet_intervals_ms:
            # interval_ms ミリ秒でavg_ts_diffサンプル
            # → 1秒あたり: avg_ts_diff * (1000 / interval_ms)
            estimated_rate = int(avg_ts_diff * (1000 / interval_ms))

            # 最も近い一般的なレートを探す
            closest_rate = min(common_rates, key=lambda x: abs(x - estimated_rate))

            # 誤差を計算
            error = abs(closest_rate - estimated_rate)
            relative_error = error / closest_rate

            # 相対誤差が5%以下なら候補とする
            if relative_error < 0.05:
                score = relative_error
                if score < best_score:
                    best_score = score
                    best_match = (closest_rate, channels, bits_per_sample)

    if best_match:
        return best_match

    # フォールバック: タイムスタンプ差が小さい場合、10ms間隔と仮定
    if avg_ts_diff < 10000:
        estimated_rate = avg_ts_diff * 100
        sample_rate = min(common_rates, key=lambda x: abs(x - estimated_rate))
        return sample_rate, 2, 16

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
    current_channels = None
    current_bits_per_sample = None
    current_ssrc = None
    last_seq_num = None
    packet_count = 0

    logger.info("Waiting for RTP packets...")

    try:
        while True:
            # RTPパケット受信
            data, addr = recv_sock.recvfrom(2048)
            packet_count += 1

            # 現在のパケットのヘッダーを解析
            header = parse_rtp_header(data)
            if not header:
                continue

            # 初回フォーマット検出
            if not detected and len(detection_packets) < 10:
                detection_packets.append(data)
                if len(detection_packets) == 10:
                    sample_rate, channels, bits_per_sample = detect_format_from_rtp(
                        detection_packets
                    )
                    payload_type = header["payload_type"]

                    logger.info(
                        f"Detected format: {sample_rate}Hz, {channels}ch, {bits_per_sample}bit, PT{payload_type}"
                    )

                    if AUTO_REGISTER:
                        if send_sdp_to_magicbox(
                            sample_rate, channels, bits_per_sample, payload_type
                        ):
                            detected = True
                            current_sample_rate = sample_rate
                            current_channels = channels
                            current_bits_per_sample = bits_per_sample
                            current_ssrc = header["ssrc"]
                            last_seq_num = header["sequence_number"]
                        else:
                            logger.warning("Failed to register session, will retry...")
                            detection_packets = []  # リトライのためリセット

            # 動的レート変更検出（SEND_SDP_ON_RATE_CHANGE が有効な場合）
            elif detected and SEND_SDP_ON_RATE_CHANGE:
                # SSRC変更を検出（ストリーム切り替えの兆候）
                if header["ssrc"] != current_ssrc:
                    logger.info(
                        f"SSRC change detected: {current_ssrc} → {header['ssrc']}"
                    )
                    # 新しいストリームとして再検出
                    detection_packets = [data]
                    detected = False
                    current_ssrc = header["ssrc"]
                    continue

                # シーケンス番号の大幅な不連続を検出
                if last_seq_num is not None:
                    expected_seq = (last_seq_num + 1) & 0xFFFF
                    actual_seq = header["sequence_number"]
                    seq_diff = abs(actual_seq - expected_seq)

                    # 10パケット以上の飛びはストリーム切り替えの可能性
                    if seq_diff > 10 and seq_diff < 60000:  # ラップアラウンドを考慮
                        logger.warning(
                            f"Sequence discontinuity detected: expected {expected_seq}, got {actual_seq}"
                        )
                        # 再検出モードに移行
                        detection_packets = [data]
                        detected = False
                        last_seq_num = actual_seq
                        continue

                last_seq_num = header["sequence_number"]

                # パケットバッファを更新（直近10パケットを保持）
                detection_packets.append(data)
                if len(detection_packets) > 10:
                    detection_packets.pop(0)

                # 定期的にフォーマット再確認（100パケットごと）
                if packet_count % 100 == 0 and len(detection_packets) >= 10:
                    new_rate, new_channels, new_bits = detect_format_from_rtp(
                        detection_packets
                    )

                    # フォーマット変更を検出
                    if (
                        new_rate != current_sample_rate
                        or new_channels != current_channels
                        or new_bits != current_bits_per_sample
                    ):
                        logger.info(
                            f"Format change detected: {current_sample_rate}Hz/{current_channels}ch/{current_bits_per_sample}bit "
                            f"→ {new_rate}Hz/{new_channels}ch/{new_bits}bit"
                        )
                        payload_type = header["payload_type"]
                        if send_sdp_to_magicbox(
                            new_rate, new_channels, new_bits, payload_type
                        ):
                            current_sample_rate = new_rate
                            current_channels = new_channels
                            current_bits_per_sample = new_bits
                            logger.info("SDP updated successfully")
                        else:
                            logger.warning("Failed to update SDP")

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
