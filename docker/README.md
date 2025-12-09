# Docker 概要

- `jetson_pcm_receiver/` : Jetson 向け PCM 受信ブリッジ専用の Dockerfile / Compose
- `jetson/` : 既存 Magic Box (Web + Daemon) 用の構成（必要に応じて使用）

ローカル検証用 (`docker/local`) と Raspberry Pi 用 (`docker/raspi`) は不要になったため削除しました。

## jetson-pcm-receiver を試す
Jetson 上で PCM over TCP → ALSA Loopback を動かす最小構成です。詳細は `jetson_pcm_receiver/README.md` を参照してください。

```bash
cd docker
docker compose -f jetson_pcm_receiver/docker-compose.jetson.yml up -d --build
docker compose -f jetson_pcm_receiver/docker-compose.jetson.yml logs -f
# 停止
docker compose -f jetson_pcm_receiver/docker-compose.jetson.yml down
```

ポイント:
- JetPack 6.1 以降 + NVIDIA Container Runtime 必須
- `--device /dev/snd` を必ず付与（Loopback/実デバイスをコンテナへ渡す）
- 環境変数 `JPR_*` で CLI 相当の設定を上書き可能（ポート、デバイス、接続モード、ZeroMQ など）

### 入力レート/フォーマットの扱い（TCPのみ）
- Raspberry Pi 送信側が `PCMA` ヘッダで `sample_rate` / `channels` / `format` を通知し、Jetson 側 `jetson_pcm_receiver` がヘッダを検証して ALSA を開き直します。JSON で固定値を持たせるとミスマッチ時に無音になるため **config.docker.json では入力レート/フォーマットを設定しません**。
- 受理するヘッダ: 44.1k / 48k 系の {×1,×2,×4,×8,×16}、チャンネル=2、フォーマット=`S16_LE(1)` / `S24_3LE(2)` / `S32_LE(4)`（`jetson_pcm_receiver` / `PcmFormatSet` に準拠）。
- GPU パイプライン側の自動ネゴシエーションは受信した実レートを使って出力レート・アップサンプル比を決めます（固定値依存なし）。
- `docker/jetson/config.docker.json` の `rtp` は削除予定のため `enabled: false` にしています。今後は TCP 経由のみを想定してください（PipeWire/RTP はデプリケート予定）。

## 既存 Magic Box Jetson コンテナ
`docker/jetson/` は従来の Magic Box (Web/UI + Audio Daemon) 用です。今回の Issue では新規 `jetson_pcm_receiver/` を優先してください。

## 設定の永続化と初期化（Jetson magicbox）
- `magicbox-config` ボリューム(`/opt/magicbox/config`)に `config.json` を保存し、コンテナ再ビルドでも設定が維持されます。
- 初回またはリセット時は `docker/jetson/config.docker.json` の安全な初期値をコピーします（本体イメージ内の `config-default` からシード）。
- 設定を工場出荷状態に戻したい場合は `MAGICBOX_RESET_CONFIG=true docker compose -f jetson/docker-compose.jetson.yml up -d` を実行してください。
- JSONが壊れている場合は自動的にデフォルトへ復旧し、壊れたファイルは `config.json.bak` にバックアップします。
- ボリュームを削除して完全初期化する場合: `docker volume rm $(docker volume ls -q | grep magicbox-config)`（再作成時にデフォルトが再配置されます）。
