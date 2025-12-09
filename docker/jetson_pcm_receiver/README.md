## jetson-pcm-receiver Docker (Jetson向け)

Jetson上で PCM over TCP を受信して ALSA Loopback に書き込む `jetson-pcm-receiver` を Docker で起動する構成です。ビルドもコンテナ内で完結します。

### 前提
- JetPack 6.1 (L4T r36.4) 以降
- NVIDIA Container Runtime が有効 (`sudo docker run --rm --runtime=nvidia nvidia-smi` で確認)
- ループバックデバイスまたは実デバイスをコンテナに渡せること（`--device /dev/snd` が必須）

### ビルドと起動（Compose）
```bash
cd docker
docker compose -f jetson_pcm_receiver/docker-compose.jetson.yml up -d --build
docker compose -f jetson_pcm_receiver/docker-compose.jetson.yml logs -f
```

### 単体起動（docker run）
```bash
docker run --rm \
  --runtime=nvidia \
  --device /dev/snd \
  -p 46001:46001/tcp \
  jetson-pcm-receiver:latest \
  jetson-pcm-receiver
```

### 環境変数での設定上書き
バイナリ内で直接パースされるため、そのまま環境変数を渡すだけで CLI と同等の設定が可能です。

- `JPR_PORT` (default: 46001)
- `JPR_DEVICE` (`loopback` | `null` | `alsa:<pcm>` | `<raw pcm>`)
- `JPR_LOG_LEVEL` (`error|warn|info|debug`, default: `warn`)
- `JPR_RING_FRAMES`, `JPR_WATERMARK_FRAMES`, `JPR_DISABLE_RING_BUFFER`
- `JPR_RECV_TIMEOUT_MS`, `JPR_RECV_TIMEOUT_SLEEP_MS`, `JPR_ACCEPT_COOLDOWN_MS`, `JPR_MAX_CONSEC_TIMEOUTS`
- `JPR_CONNECTION_MODE` (`single|takeover|priority`)
- `JPR_PRIORITY_CLIENTS` (カンマ区切り)
- `JPR_DISABLE_ZMQ`, `JPR_ENABLE_ZMQ_RATE_NOTIFY`, `JPR_ZMQ_ENDPOINT`, `JPR_ZMQ_PUB_INTERVAL_MS`, `JPR_ZMQ_TOKEN`

### 起動時の注意点
- `/dev/snd` がない場合、コンテナは起動しますが再生できません。必ず `--device /dev/snd` を付与してください。
- Jetson ホストのカーネルで `snd-aloop` をロードしておくと Loopback への出力が行えます。
  例: `sudo modprobe snd-aloop`
- ZeroMQ エンドポイントは `ipc:///tmp/jetson_pcm_receiver.sock` をデフォルトとし、`.pub` サフィックスで PUB をバインドします。必要に応じて `JPR_ZMQ_ENDPOINT` を変更してください。

### クリーンアップ
```bash
docker compose -f jetson_pcm_receiver/docker-compose.jetson.yml down
```
