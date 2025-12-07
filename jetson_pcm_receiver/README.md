# jetson-pcm-receiver

Jetson 向けの PCM over TCP 受信ブリッジの足場プロジェクトです。Raspberry Pi 側の送信アプリから PCM を受信し、ALSA Loopback の playback 側へ書き込む常駐プロセスを想定しています。本コミットではビルドできる最小限の骨組みのみを用意しており、実装は今後追加します。

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

## 実行方法 (雛形段階)

```bash
./jetson_pcm_receiver/build/jetson_pcm_receiver --port 46001 --device hw:Loopback,0,0
```

現時点では TCP 待受や ALSA 書き込みは未実装で、起動時に設定値を表示して終了します。`--help` で簡易ヘルプを確認できます。

## ディレクトリ構成

- `src/` : エントリポイントとクラス実装の雛形
- `include/` : `TcpServer` / `AlsaPlayback` / `PcmStreamHandler` のヘッダ
- `CMakeLists.txt` : ALSA・pthread・BSD ソケット検出を行う単体プロジェクト

