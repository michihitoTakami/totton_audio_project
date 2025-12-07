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
- SIGINT/SIGTERM受信でALSAとソケットをクリーンにクローズして終了します。
- `--log-level` で `debug|info|warn|error` を選択（デフォルト: `info`）。主要イベントを標準出力/標準エラーへ出力します。
- `--iterations` はテスト用。0以下で無限ループ、正の値を指定するとその回数で自動終了します。

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
  --frames 4096 \
  --log-level info
```

- 対応フォーマット: `S16_LE`, `S24_3LE`, `S32_LE`
- 対応レート: `44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000, 705600, 768000`
- ポート範囲外や未対応フォーマット/レート指定時は起動時にエラー終了します。
- XRUN発生時は `snd_pcm_prepare()` でリカバリし、ログへ出力します。

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
