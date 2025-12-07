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

## 実行

```bash
./raspberry_pi/build/rpi_pcm_bridge --help
```

現状はスタブ実装のため、ヘルプとプレースホルダーのログのみを出力します。
対応サンプリングレートは 44.1/48kHz の 2/4/8/16倍（最大 705.6kHz/768kHz）。

## ALSA キャプチャ簡易テスト

実機のALSAデバイスを指定して数周期分を読み取ります（TCP送信は未実装）。

```bash
./raspberry_pi/build/rpi_pcm_bridge \
  --device hw:0,0 \
  --rate 48000 \
  --format S16_LE \
  --frames 4096 \
  --iterations 3
```

- 対応フォーマット: `S16_LE`, `S24_3LE`, `S32_LE`
- XRUN発生時は `snd_pcm_prepare()` でリカバリし、ログへ出力します。
- 未対応フォーマットを指定すると即エラー終了します。
