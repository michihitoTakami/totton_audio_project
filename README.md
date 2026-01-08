# Totton Audio Project - 魔法の箱

**全てのヘッドホンユーザーに最高の音を届ける箱**

## ビジョン

究極のシンプルさ
1. 箱をつなぐ
2. 管理画面でポチポチ
3. 最高の音

余計な手順や出力を考えず、ヘッドホンを選んでボタンを押すだけで音質最適化が完了します。

## リポジトリが提供するもの

- **GPUアップサンプリングエンジン**: 640kタップ最小位相FIR + CUDA/cuFFT（SM 7.5/8.7）で最大768kHz出力
- **自動補正EQ**: OPRAデータとKB5000_7ターゲットでヘッドホンごとに自動補正
- **Auto-Negotiation** と **Soft Mute** により、入力レート・DAC能力を考慮した安全なレート切替
- **Crossfeed (HRTF)** や **AI De-Limiter** を含む、プリアンプ/ループバック型データプレーンとの連携
- **Control Plane**: FastAPI + ZeroMQ + Web UI で設定・監視を提供

### アーキテクチャ概要

```
Control Plane (Python/FastAPI)
├── FastAPI Web API / Web UI
├── Filter Generator (scipy)
└── ZeroMQ Command Interface
   ↕
Data Plane (C++/CUDA)
├── ALSA / RTP 入力
├── Rate Detection → Filter Selection
├── GPU FFT Convolution（640k tap + Overlap-Save）
├── Crossfeed + Soft Mute + Headroom Limiter
└── ALSA 出力
```

### クイックスタート（PC開発環境）

1. **フィルタ生成**: `uv sync` 後に `uv run python scripts/filters/generate_minimum_phase.py --taps 2000000` などで係数を構築
2. **ビルド**: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
3. **テスト**: `./build/cpu_tests`, `./build/auto_negotiation_tests`, `./build/gpu_tests` などを実行
4. **デーモン起動**: `./scripts/daemon.sh start` → ALSA 出力を `GPU Upsampler` に設定
5. **Web UI / API**: `docs/api/README.md` を参照し、FastAPI プロセスを起動した後 `http://localhost:8000` へアクセス

**Jetson Orin Nano など実機** でのデプロイやネットワーク入力は `docs/setup/pi_bridge.md` / `docs/jetson/*`（Jetson固有の設計/運用）を参照してください。

## 重要なドキュメントリファレンス

- セットアップ: `docs/setup/pc_development.md`, `docs/setup/pi_bridge.md`, `docs/setup/web_ui.md`, `docs/setup/delimiter.md`
- API: `docs/api/README.md` と `docs/api/openapi.json`
- アーキテクチャ: `docs/architecture/crossfeed_integration.md`, `docs/architecture/two_stage_fir_pipeline.md`
- 仕様: `docs/specifications/opra-sync.md`, `docs/specifications/delimiter_streaming_chunking.md`
- フィルタ解析/検証: `scripts/analysis/verify_frequency_response.py`, `src/daemon/audio_pipeline/headroom_controller.cpp`

## テストと検証

- 各レートのフィルタは `scripts/filters` 下のスクリプトで再生成可能
- 自動化テストは `cmake` でビルドした `./build/*_tests` を利用
- GPU 実装の健全性確認には `clang-tidy` / `diff-based-tests` を走らせる。

## ライセンスとサードパーティデータ

| アイテム | ライセンス | 備考 |
|---------|------------|------|
| 本プロジェクトコード | MB-NCL v1.0 | 非商用・組織評価/PoC
| OPRA EQ データ | CC BY-SA 4.0 | 帰属表示必須
| HUTUBS (Crossfeed) | CC BY 4.0 | 帰属表示必須 |

本プロジェクトでは `OPRA` 由来の EQ データの利用のみを想定し、`oratory1990` のような商用利用不可のデータは含めません。

### 依存ライブラリ

| ライブラリ | 用途 | ライセンス |
|------------|------|------------|
| CUDA/cuFFT | GPU 畳み込み | NVIDIA EULA |
| alsa-lib | ALSA 出力 | LGPL-2.1 |
| ZeroMQ | IPC 通信 | LGPL-3.0 |
| FastAPI | Web API | MIT |
| scipy | IR 生成 | BSD-3-Clause |
