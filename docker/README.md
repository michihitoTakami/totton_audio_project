# Docker 概要

- `jetson/` : Magic Box (Web + Daemon) コンテナ。RTP受信（GStreamer）は同一コンテナ内で実行する。
- `jetson_pcm_receiver/` : 旧TCPブリッジのDockerfile置き場（Composeからは除去済み、ビルド非推奨）

ローカル検証用 (`docker/local`) と Raspberry Pi 用 (`docker/raspi`) は不要になったため削除しました。

## Jetson Compose（magicboxのみ）
Jetson 上で Magic Box を起動する構成です。RTP受信は Magic Box コンテナ内の FastAPI エンドポイント (`/api/rtp-input/*`) から開始・停止・設定変更できます。

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.yml up -d --build
docker compose -f jetson/docker-compose.jetson.yml logs -f
# 停止
docker compose -f jetson/docker-compose.jetson.yml down
```

ポイント:
- JetPack 6.1 以降 + NVIDIA Container Runtime 必須
- `--device /dev/snd` を必ず付与（Loopback/実デバイスをコンテナへ渡す）
- 環境変数 `JPR_*` で CLI 相当の設定を上書き可能（ポート、デバイス、接続モード、ZeroMQ など）
- TCP入力は `jetson-pcm-receiver` が受け、Magic Box には ALSA Loopback で渡す（Magic Box 側のTCP穴あけ不要）
- サービスを個別に起動したい場合: `docker compose -f jetson/docker-compose.jetson.yml up -d --build jetson-pcm-receiver` のようにサービス名を指定

### 入力レート/フォーマットの扱い（TCPのみ）
- Raspberry Pi 送信側が `PCMA` ヘッダで `sample_rate` / `channels` / `format` を通知し、Jetson 側 `jetson_pcm_receiver` がヘッダを検証して ALSA を開き直します。JSON で固定値を持たせるとミスマッチ時に無音になるため **config.docker.json では入力レート/フォーマットを設定しません**。
- 受理するヘッダ: 44.1k / 48k 系の {×1,×2,×4,×8,×16}、チャンネル=2、フォーマット=`S16_LE(1)` / `S24_3LE(2)` / `S32_LE(4)`（`jetson_pcm_receiver` / `PcmFormatSet` に準拠）。
- GPU パイプライン側の自動ネゴシエーションは受信した実レートを使って出力レート・アップサンプル比を決めます（固定値依存なし）。

## Magic Box コンテナの事前ビルド（Jetson 本体で実行）
`docker/jetson/Dockerfile.jetson` はホストでビルド済みのバイナリをコピーする前提です。コンテナを立ち上げる前に Jetson 上で以下を実行し、`build/gpu_upsampler_alsa` と `build/gpu_upsampler_daemon` を用意してください。

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc)
ls -l build/gpu_upsampler_alsa build/gpu_upsampler_daemon
```

ビルド手順の詳細は `../docs/setup/build.md` を参照してください。

## 設定の永続化と初期化（Jetson magicbox）
- `magicbox-config` ボリューム(`/opt/magicbox/config`)に `config.json` を保存し、コンテナ再ビルドでも設定が維持されます。
- 初回またはリセット時は `docker/jetson/config.docker.json` の安全な初期値をコピーします（本体イメージ内の `config-default` からシード）。
- 設定を工場出荷状態に戻したい場合は `MAGICBOX_RESET_CONFIG=true docker compose -f jetson/docker-compose.jetson.yml up -d` を実行してください。
- JSONが壊れている場合は自動的にデフォルトへ復旧し、壊れたファイルは `config.json.bak` にバックアップします。
- ボリュームを削除して完全初期化する場合: `docker volume rm $(docker volume ls -q | grep magicbox-config)`（再作成時にデフォルトが再配置されます）。
