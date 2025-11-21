# コードベース総点検メモ（Jetson Orin DDC 想定）

## 調査概要
- 対象: `src/`, `include/`, `CMakeLists.txt`, `config.json`, `docs/setup_guide.md`。
- 観点: 1) 環境依存の強い記述・前提、2) 未使用/未実装コード、3) Jetson Orin で独立DDCとして動かす際の注意。

## 環境依存が強い箇所
- LV2プラグインは現状非メンテにつきビルド対象から除去済み（`src/lv2_plugin/`を削除）。過去のドキュメントに残る参照は無効。
- `CMakeLists.txt:42-55`  
  - CUDA アーキテクチャを `"75"` 固定（RTX 2070S 前提）。Orin (SM87) では `CUDA_ARCHITECTURES "87"` などへ変更が必要。  
  - `CUDA::nvml` を必須リンクにしており、NVML が無い/権限不足の環境（Jetson の一部イメージ）ではビルドが失敗する。
- `include/config_loader.h:7-16` および `config.json`  
  - 既定の ALSA デバイスが `hw:USB` / `hw:CARD=AUDIO,DEV=0` と SMSL DAC 前提。Orin の I2S/内蔵デバイスでは適宜書き換えが必要。
- `src/alsa_daemon.cpp:19-26, 111-160, 536-599`  
  - 入力は 44.1kHz 固定、PipeWire ノード `gpu_upsampler_sink.monitor` からのキャプチャ前提。出力メッセージも SMSL DAC 705.6kHz を想定。Jetson で PipeWire を使わない構成ではキャプチャ経路・サンプリングレートを可変化する必要あり。
- `src/pipewire_daemon.cpp:12-18, 289-376`  
  - 44.1k→705.6k (16x) 固定。出力ターゲット `alsa_output.usb-SMSL_SMSL_USB_AUDIO-00.iec958-stereo` にハードコードされており、別デバイスでは動作しない。
- `docs/setup_guide.md:19, 236-237`  
  - `WorkingDirectory=/home/michihito/Working/gpu_os` などローカルパス前提の systemd 例。Orin 用にはパスとユーザーを差し替える必要がある。

## 未使用/未実装コード
- `include/convolution_engine.h:172-175`  
  - `Utils::zeroPad` を宣言しているが実装・参照ともに無し。不要なら削除、必要なら定義を追加する。
- `src/convolution_engine.cu:279-286`  
  - `d_inputBlock_` / `d_outputBlock_` を確保しているが未使用。GPU メモリを浪費するので削除か実際の処理に統合すべき。

## Jetson Orin での移行上の注意（提案）
- CUDA: `CUDA_ARCHITECTURES` を SM87 に合わせる。NVML を optional にし、未インストール時は GPU 使用率取得をスキップするガードを入れる。
- I/O 経路: PipeWire 依存を外し、ALSA 直キャプチャ（I2S/USB）と 44.1k/48k の両対応が必要。ノード名やデバイス名は `config.json` で必ず上書きできるようにする。
- サンプリング設定: Daemon 側は 44.1k 固定のままなので、48k→768k を扱う場合は `DEFAULT_INPUT_SAMPLE_RATE` やフィルタ選択を可変化する。
- デプロイ: systemd ユニット例や LV2 インストール先を可変化し、`/home/michihito/...` 固定の記述をなくす。***
