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
