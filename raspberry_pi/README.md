# Raspberry Pi PCM Bridge (prototype)

Raspberry Pi向けのPCMブリッジ雛形です。ALSAキャプチャとTCPクライアントのスタブを含み、リンクが通る最小構成を提供します。

## 必要パッケージ

Raspberry Pi OS / Debian系:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libasound2-dev
```

## ビルド手順

```bash
cmake -S raspberry_pi -B raspberry_pi/build -DCMAKE_BUILD_TYPE=Release
cmake --build raspberry_pi/build
```

Debugビルド例:

```bash
cmake -S raspberry_pi -B raspberry_pi/build -DCMAKE_BUILD_TYPE=Debug
cmake --build raspberry_pi/build
```

生成物:
- バイナリ: `raspberry_pi/build/rpi_pcm_bridge`

## 動作概要

- 起動時にPCMヘッダ（`PCMA` / version=1 / rate / ch / format）をTCP接続確立後に送信し、その後はALSAから読み取ったPCMを順次送出します。
- TCP切断時は無制限リトライし、再接続後にヘッダを再送してPCM送出を再開します。
- SIGINT/SIGTERM受信でALSAとソケットをクリーンにクローズして終了します。
- `--log-level` で `debug|info|warn|error` を選択（デフォルト: `info`）。主要イベントを標準出力/標準エラーへ出力します。
- `--iterations` はテスト用。0以下で無限ループ、正の値を指定するとその回数で自動終了します。
- ALSA実測サンプルレート/チャネル数/フォーマットの変化やデバイス再列挙を検知すると、ALSAを再オープンし、TCP再接続とヘッダ再送で追従します。
- ALSAデバイスが一時的に消失しても、復活するまでバックオフしながら再オープンを繰り返し、復旧後にヘッダを再送して送出を再開します。
- `--device auto` を指定すると、起動時に入力可能な最初のALSAデバイスを自動選択します。安定運用する場合は `hw:CARD=<名前>,DEV=0` などカードIDを用いた安定名指定を推奨します。

## 実行例

ヘルプ・バージョン表示:

```bash
./raspberry_pi/build/rpi_pcm_bridge --help
./raspberry_pi/build/rpi_pcm_bridge --version
```

TCP送信先とキャプチャ設定を指定する例:

```bash
./raspberry_pi/build/rpi_pcm_bridge \
  --device hw:0,0 \
  --host 192.168.1.50 \
  --port 46001 \
  --rate 96000 \
  --format S24_3LE \
  --frames 4096
```

- 対応フォーマット: `S16_LE`, `S24_3LE`, `S32_LE`
- 対応レート: `44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000, 705600, 768000`
- ポート範囲外や未対応フォーマット/レート指定時は起動時にエラー終了します。
- XRUN発生時は `snd_pcm_prepare()` でリカバリし、ログへ出力します。
<<<<<<< HEAD
- `PCM_BRIDGE_*` 環境変数でデフォルト値を上書き可能です（Dockerで利用）。

## Docker 実行（Raspberry Pi上）

### 単体コンテナ（/dev/sndを直接渡す）

```bash
# プロジェクトルートでビルド
docker build -f raspberry_pi/Dockerfile -t rpi-pcm-bridge .

# ヘルプ表示だけ行う例
docker run --rm --device /dev/snd -e PCM_BRIDGE_MODE=help rpi-pcm-bridge

# 実際に送信待機を行う例
docker run --rm --device /dev/snd \\
  --group-add audio \\
  --restart unless-stopped \\
  -e PCM_BRIDGE_HOST=192.168.55.1 \\
  -e PCM_BRIDGE_PORT=46001 \\
  -e PCM_BRIDGE_DEVICE=hw:0,0 \\
  -e PCM_BRIDGE_RATE=48000 \\
  -e PCM_BRIDGE_FORMAT=S16_LE \\
  rpi-pcm-bridge
```

- 基本は `/dev/snd` と `audio` グループ付与で動作します。権限問題がある場合のみ `--privileged` を検討してください。
- `PCM_BRIDGE_MODE=version` でバージョン表示、`PCM_BRIDGE_MODE=help` でヘルプを表示します。

### docker compose + nginxリバースプロキシ

Jetsonの80番ポートにNginxでリバースプロキシを挟み、同一LAN内からWeb UIを見られるようにする例です。

```bash
# Raspberry Pi上で実行
docker compose -f raspberry_pi/docker-compose.yml up -d --build

# ログ確認
docker compose -f raspberry_pi/docker-compose.yml logs -f pcm-bridge
docker compose -f raspberry_pi/docker-compose.yml logs -f jetson-proxy

# 停止
docker compose -f raspberry_pi/docker-compose.yml down
```

- 環境変数 `JETSON_HOST` と `JETSON_PORT` でプロキシ先を変更できます（デフォルト: `jetson:80`）。
- 80番ポートで待ち受けるため、ホスト側で別のサービスが使用していないことを確認してください。必要なら `ports` を `8080:80` に変更して利用してください。
- `restart: always` で電源ロスト・Dockerデーモン再起動後も自動再立ち上げします（composeデフォルト設定）。

### 環境変数一覧（コンテナ）

- `PCM_BRIDGE_DEVICE` (既定: `hw:0,0`)
- `PCM_BRIDGE_HOST` (既定: `127.0.0.1`)
- `PCM_BRIDGE_PORT` (既定: `46001`)
- `PCM_BRIDGE_RATE` (既定: `48000`)
- `PCM_BRIDGE_FORMAT` (`S16_LE` | `S24_3LE` | `S32_LE`)
- `PCM_BRIDGE_FRAMES` (既定: `4096`)
- `PCM_BRIDGE_LOG_LEVEL` (`debug` | `info` | `warn` | `error`)
- `PCM_BRIDGE_ITERATIONS` (`-1` で無限送信)
=======

## 手動テスト（null sink/loopback + nc）

Jetson側TCPサーバが無くても、ローカルのALSA loopback + `nc` でヘッダとPCM転送を検証できます。

1. ALSAループバックを有効化
   `sudo modprobe snd-aloop pcm_substreams=2`
2. TCP受信を起動（別シェル）
   `nc -l -p 46001 > /tmp/pcm_dump.raw`
3. ブリッジを起動（Loopbackキャプチャを指定）
   ```bash
   ./raspberry_pi/build/rpi_pcm_bridge \
     --device hw:Loopback,1 \
     --host 127.0.0.1 \
     --port 46001 \
     --rate 48000 \
     --format S16_LE \
     --frames 1024 \
     --log-level debug
   ```
4. 任意の音声をLoopback再生側へ流す（別シェル）
   `speaker-test -D hw:Loopback,0 -c 2 -r 48000 -F S16_LE`
   または `aplay -D hw:Loopback,0 /path/to/test.wav`
5. 受信を確認
   - `hexdump -C /tmp/pcm_dump.raw | head` で先頭16バイトが `50 43 4d 41` (`PCMA`) になっていることを確認。
   - ファイルサイズが再生に合わせて増えていくことを確認。
   - ブリッジは `Ctrl+C` で終了（SIGINTでクリーンに停止）。
>>>>>>> origin/main
