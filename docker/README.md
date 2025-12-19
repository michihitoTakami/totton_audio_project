# Docker 概要

- `jetson/` : Magic Box (Web + Daemon) コンテナ。RTP受信（GStreamer）は同一コンテナ内で実行する。
- `jetson_pcm_receiver/` : 旧TCPブリッジのDockerfile置き場（Composeからは除去済み、ビルド非推奨）
- `raspberry_pi/rtp_receiver/` : ラズパイ側で ZeroMQ ブリッジをサイドカーとして起動できる Python 実装（`python -m raspberry_pi.rtp_receiver`）。

ローカル検証用 (`docker/local`) と Raspberry Pi 用 (`docker/raspi`) は不要になったため削除しました。

## Jetson Compose（magicboxのみ）
Jetson 上で Magic Box を起動する構成です。コンテナ起動直後から RTP 受信が自動で開始されるため、API を叩かなくても音が入ります（環境変数で無効化可能）。RTP/RTCP のデフォルトポートは以下です:

- RTP: 46000/udp
- RTCP (Jetson受信): 46001/udp
- RTCP (Jetson送信): 46002/udp

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
- RTP 受信は Magic Box コンテナ内の `rtp_input` サービスが担当し、デフォルトで自動起動します。無効化したい場合は `MAGICBOX_RTP_AUTOSTART=false` を指定してください。
- 受信設定（ポート/レート/デバイス/品質）は環境変数で上書き可能です（例: `MAGICBOX_RTP_PORT`, `MAGICBOX_RTP_SAMPLE_RATE`, `MAGICBOX_RTP_DEVICE`, `MAGICBOX_RTP_QUALITY`）。詳細は Web API `/api/rtp-input/config` のスキーマに準拠します。
- 起動後に gst-launch が立ち上がっているかを確認するには `docker compose -f jetson/docker-compose.jetson.yml exec magicbox pgrep -f rtpbin` を利用してください。RTPプロセスが異常終了した場合も自動でリトライします（ALSAデバイス未接続時はリトライし続けるのでデバイスのマウントを確認してください）。連続失敗時のログは一定間隔(デフォルト30s)で抑制されます。
- サービスを個別に起動したい場合: `docker compose -f jetson/docker-compose.jetson.yml up -d --build magicbox` のようにサービス名を指定
- `restart: always` を指定済み。systemd で単体起動する場合も `Restart=always` を付け、片側クラッシュ時に自動復帰させてください。

## Magic Box コンテナの事前ビルド（Jetson 本体で実行）
`docker/jetson/Dockerfile.jetson` はホストでビルド済みのバイナリをコピーする前提です。コンテナを立ち上げる前に Jetson 上で以下を実行し、`build/gpu_upsampler_alsa` を用意してください。

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc)
ls -l build/gpu_upsampler_alsa
```

ビルド手順の詳細は `../docs/setup/build.md` を参照してください。

## 設定の永続化と初期化（Jetson magicbox）
- `magicbox-config` ボリューム(`/opt/magicbox/config`)に `config.json` を保存し、コンテナ再ビルドでも設定が維持されます。
- 初回またはリセット時は `config.json` の安全な初期値をコピーします（本体イメージ内の `config-default` からシード）。
- Jetson 固有の設定（I2S 有効化など）は `MAGICBOX_PROFILE=jetson` で上書きされます。
- 設定を工場出荷状態に戻したい場合は `MAGICBOX_RESET_CONFIG=true docker compose -f jetson/docker-compose.jetson.yml up -d` を実行してください。
- JSONが壊れている場合は自動的にデフォルトへ復旧し、壊れたファイルは `config.json.bak` にバックアップします。
- ボリュームを削除して完全初期化する場合: `docker volume rm $(docker volume ls -q | grep magicbox-config)`（再作成時にデフォルトが再配置されます）。
