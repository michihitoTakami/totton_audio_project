# RTP 移行リリースノート (2025-12)

## 概要
- Raspberry Pi 側の TCP/C++ 送出と Jetson 側の TCP 受信を完全に撤廃し、GStreamer RTP (RTCP 同期付き) を正式ルートに統一。
- 送信は `raspberry_pi/rtp_sender.py`（gst-launch ラッパー）、受信は Jetson の `rtp_input` サービス (`web/services/rtp_input.py`) が担当。
- ZeroMQ ブリッジ (`raspberry_pi/rtp_receiver`) を使うことで Web UI からレイテンシ調整や統計確認が可能。

## 移行理由
- **クロックドリフト耐性**: `audioresample quality=10` + RTCP で送信側クロックに追従し、PCM 生 TCP よりドロップ/スリップを抑制。
- **安定性**: rtpbin の jitterbuffer によりパケットロス時のリカバリが容易。TCP 再接続ループによる突発 XRUN を解消。
- **運用一貫性**: FastAPI の `rtp_input` API で開始/停止・設定変更を統一管理。Docker も GStreamer 依存に一本化。

## 性能比較（目安）
- **レイテンシ**: RTP パイプライン + 100ms jitterbuffer でエンドツーエンド ~120–150ms。TCP 実装の再接続時スパイクを解消。
- **安定性**: 24時間連続再生で XRUN < 10 回を想定（RTCP 同期時）。TCP 実装で散発したドロップ/再接続は非再現。
- **スケーラビリティ**: 44.1k/48k 系全レートを L16/L24/L32 で送受信可能。payload type 96 デフォルト。

## 移行手順（ユーザー向け）
1. Raspberry Pi: `raspberry_pi/rtp_sender.py` を使用し、`RTP_SENDER_*` 環境変数でデバイス/ポート/フォーマットを設定。
2. Jetson: Web API で `POST /api/rtp-input/start` を実行。必要に応じて `PUT /api/rtp-input/config` でレート・フォーマット・レイテンシを調整。
3. 旧 `jetson_pcm_receiver` / `rpi_pcm_bridge` バイナリや CMake プロジェクトは削除済み。Docker は `raspberry_pi/Dockerfile` と `raspberry_pi/docker-compose.yml` を使用。
4. レイテンシ調整を UI/ZeroMQ で行う場合は `raspberry_pi/rtp_receiver` をサイドカーとして起動し、`RTP_BRIDGE_*` パスを共有。

## 既知の互換性変更
- `raspberry_pi/src`, `raspberry_pi/include`, `raspberry_pi/tests`, `jetson_pcm_receiver/` の C++ コードはリポジトリから削除。
- `docker/jetson_pcm_receiver` ビルドは非推奨。Jetson 側は本体コンテナ内の `rtp_input` サービスを利用してください。
