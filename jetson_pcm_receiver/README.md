# jetson-pcm-receiver

Jetson 向けの PCM over TCP 受信ブリッジです。Raspberry Pi 側の送信アプリから PCM を受信し、ALSA Loopback の playback 側へ書き込みます（初期実装は S16_LE / 2ch / 48kHz を想定）。

## 前提パッケージ (Jetson / Ubuntu 系)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libasound2-dev
```

## ビルド方法

```bash
cmake -S jetson_pcm_receiver -B jetson_pcm_receiver/build -DCMAKE_BUILD_TYPE=Release
cmake --build jetson_pcm_receiver/build -j$(nproc)
```

> `CMAKE_BUILD_TYPE` を `Debug` に変えることでデバッグ向けビルドに切り替えられます。

## 実行方法

```bash
./jetson_pcm_receiver/build/jetson_pcm_receiver \
  --port 46001 \
  --device hw:Loopback,0,0 \
  --log-level info      # error / warn / info / debug
```

- 受信ヘッダが `PCMA` / version 1 かつ 44.1kHz or 48kHz の {1,2,4,8,16} 倍、2ch、フォーマットが `S16_LE(1)` / `S24_3LE(2)` / `S32_LE(4)` の場合に再生します。
- フォーマットやレートが未対応の場合はエラーログを出して接続を閉じます。
- XRUN (`-EPIPE`) が発生した場合は `snd_pcm_prepare()` で復旧を試み、結果をログします。
- SIGINT/SIGTERM で停止要求を検出し、接続待受ループを抜けて終了します。

## ディレクトリ構成

- `src/` : エントリポイントとクラス実装の雛形
- `include/` : `TcpServer` / `AlsaPlayback` / `PcmStreamHandler` のヘッダ
- `CMakeLists.txt` : ALSA・pthread・BSD ソケット検出を行う単体プロジェクト

## ALSA Loopback への疎通確認（簡易）

1. ループバックをロード（必要な場合）
   `sudo modprobe snd-aloop`
2. 受信側を起動
   `./jetson_pcm_receiver/build/jetson_pcm_receiver --port 46001 --device hw:Loopback,0,0`
3. 別ターミナルから 1 秒分の無音を送信
   ```bash
   python - <<'PY'
   import socket, struct
   hdr = struct.pack("<4sIIHH", b"PCMA", 1, 48000, 2, 1)  # S16_LE / 48k / 2ch
   pcm = b"\x00\x00" * 2 * 48000  # 1秒分のステレオ無音 (4バイト/フレーム)
   s = socket.create_connection(("127.0.0.1", 46001))
   s.sendall(hdr + pcm)
   s.close()
   PY
   ```
4. Loopback capture 側で再生データを確認（例）
   `arecord -D hw:Loopback,1,0 -f S16_LE -c 2 -r 48000 -d 2 /tmp/captured.wav`

### nc を使った簡易疎通（ヘッダのみ）

```bash
{ printf "PCMA"; printf "\x01\x00\x00\x00"; printf "\x80\xBB\x00\x00"; printf "\x02\x00"; printf "\x01\x00"; } | nc 127.0.0.1 46001
# 上記は version=1, rate=48000, channels=2, format=1(S16_LE)
```

### パラメータ調整メモ
- ALSA period / buffer サイズの初期値は `DEFAULT_PERIOD_FRAMES=512`、`DEFAULT_BUFFER_FRAMES=2048`（`src/alsa_playback.cpp`）です。XRUN が多い場合はここを拡大してください。
- フォーマットやチャンネル数を増やす際は `toPcmFormat()` と `bytesPerSample()` に対応を追加してください。
