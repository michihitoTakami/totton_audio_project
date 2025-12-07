# jetson-pcm-receiver

Jetson 向けの PCM over TCP 受信ブリッジです。Raspberry Pi 側の送信アプリから PCM を受信し、ALSA Loopback の playback 側へ書き込みます（初期実装は S16_LE / 2ch / 48kHz を想定）。

## 前提パッケージ (Jetson / Ubuntu 系)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libasound2-dev libzmq3-dev
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
  --log-level info \        # error / warn / info / debug
  --ring-buffer-frames 8192 # 0で無効（デフォルト 8192）
  --ring-buffer-watermark 0 # 0で自動(75%)
  --zmq-endpoint ipc:///tmp/jetson_pcm_receiver.sock \
  --zmq-token "" \
  --zmq-pub-interval 1000
```

- 受信ヘッダが `PCMA` / version 1 かつ 44.1kHz or 48kHz の {1,2,4,8,16} 倍、2ch、フォーマットが `S16_LE(1)` / `S24_3LE(2)` / `S32_LE(4)` の場合に再生します。
- フォーマットやレートが未対応の場合はエラーログを出して接続を閉じます。
- XRUN (`-EPIPE`) が発生した場合は `snd_pcm_prepare()` で復旧を試み、結果をログします。
- ジッタ吸収リングバッファ（デフォルト有効）。溢れた場合は古いフレームをドロップし、ウォーターマーク到達・ドロップ数をログします。
- SIGINT/SIGTERM で停止要求を検出し、接続待受ループを抜けて終了します。

## ZeroMQ ステータス/制御 API

- デフォルト: REP `ipc:///tmp/jetson_pcm_receiver.sock` / PUB `ipc:///tmp/jetson_pcm_receiver.sock.pub`
- 無効化: `--disable-zmq`（デフォルトは有効）。ローカル IPC 以外を使う場合は `--zmq-token` を必ず設定してください。
- PUB 間隔: `--zmq-pub-interval <ms>`（0 で無効）
- リクエスト例（REQ/REP、token は任意・設定時は必須）
  - ステータス取得:
    - 送信: `{"cmd":"STATUS","token":"<token>"}`
    - 応答: `{"status":"ok","data":{"listening":true,"bound_port":46001,"client_connected":false,"streaming":false,"ring_buffer_frames":8192,"watermark_frames":6144,"buffered_frames":0,"max_buffered_frames":0,"dropped_frames":0,"xrun_count":0,"last_header":null,"rep_endpoint":"...","pub_endpoint":"..."}}`
  - キャッシュ/レイテンシ変更: `{"cmd":"SET_CACHE","token":"<token>","params":{"ring_buffer_frames":16384,"watermark_frames":0}}`
    - `ring_buffer_frames=0` でリングバッファ無効、`watermark_frames=0` で自動 75%
  - 再起動要求: `{"cmd":"RESTART","token":"<token>"}`
- PUB でのステータス通知: `{"event":"status", ...上記 data と同等のフィールド...}` を周期送信
- 取得できる主なフィールド:
  - `listening` / `bound_port` / `client_connected` / `streaming`
  - `last_header` (sample_rate, channels, format, version) 最後に受理したヘッダ
  - `ring_buffer_frames`, `watermark_frames`, `buffered_frames`, `max_buffered_frames`, `dropped_frames`
  - `xrun_count` (XRUN 検出回数)

## ディレクトリ構成

- `src/` : エントリポイントとクラス実装の雛形
- `include/` : `TcpServer` / `AlsaPlayback` / `PcmStreamHandler` のヘッダ
- `CMakeLists.txt` : ALSA・pthread・BSD ソケット検出を行う単体プロジェクト

## ALSA Loopback への疎通確認（簡易）

### ヘッダのみ送って受理/拒否を確認
PCM ペイロードなしでヘッダ検証だけ確認できます。

```bash
# 正常ヘッダ（48000Hz, 2ch, S16_LE=1）を送る例
python - <<'PY'
import socket, struct
hdr = struct.pack("<4sIIHH", b"PCMA", 1, 48000, 2, 1)
s = socket.create_connection(("127.0.0.1", 46001))
s.sendall(hdr)
s.close()
PY

# 不正ヘッダ（magic違い）で拒否を確認する例
python - <<'PY'
import socket, struct
hdr = struct.pack("<4sIIHH", b"XXXX", 1, 48000, 2, 1)
s = socket.create_connection(("127.0.0.1", 46001))
s.sendall(hdr)
s.close()
PY
```

### ループバックで無音1秒を流す
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
