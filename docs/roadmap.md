# Magic Box Project - Development Roadmap

## Vision

**全てのヘッドホンユーザーに最高の音を届ける箱**

- 箱をつなぐ → 管理画面でポチポチ → 最高の音
- ユーザーに余計なことを考えさせない

## Phase Overview

```
Phase 1: Core Engine & Middleware       [====================] 100% ✅ 完了
Phase 2: Control Plane & Web UI         [====================] 100% ✅ 完了
Phase 2.5: Refactoring & Stabilization  [================>   ] 85% 🔄 進行中
Phase 3: Hardware Integration           [====>               ] 20% (準備中)
Phase 4: Commercialization & Deployment [                    ] 0% (計画中)
```

---

## Phase 1: Core Engine & Middleware

**Status:** ✅ 完了（100%）

システムの心臓部であるC++ Audio Engine Daemonが完成。全機能実装済み。

### Completed Tasks ✅

- [x] **GPU Convolution Algorithm**
  - 640k-tap minimum phase FIR filter実装完了
  - ~28x realtime performance on RTX 2070S
  - Overlap-Save方式によるストリーミング処理
  - Partitioned Convolution（低遅延モード対応）

- [x] **Filter Coefficient Generation**
  - scipy.signalによる640kタップフィルタ生成
  - ~160dB stopband attenuation（24bit品質に十分）
  - Kaiser window (β≈28) - 32bit Float実装の量子ノイズ限界に合わせた最適値
  - Minimum Phase / Linear Phase 両対応

- [x] **Phase Type Selection** (#165, #166, #167)
  - Minimum Phase / Linear Phase 切り替え機能
  - `scripts/generate_linear_phase.py` による線形位相フィルタ生成
  - C++/CUDA側の位相タイプ対応（遅延計算含む）
  - 設定システム（`PhaseType` enum）

- [x] **Multi-Rate Support** ✅ Issue #231
  - 全8入力レート対応（44.1k/88.2k/176.4k/352.8k/48k/96k/192k/384k）
  - 動的レート切り替え（グリッチフリー）
  - GPU Upsamplerのマルチレート対応完了

- [x] **Daemon Implementation**
  - PipeWire入力 → GPU処理 → ALSA出力
  - RTP Session Manager統合（ハイレゾ対応）
  - SDP自動パース機能

- [x] **ZeroMQ Communication Layer**
  - Control Plane ↔ Data Plane通信完了
  - REST API経由のコマンド送信
  - リアルタイムステータス取得

- [x] **Safety Mechanisms**
  - **Soft Mute**: レート切り替え時のクロスフェード実装済み
  - **Hot-swap IR loading**: グリッチフリーな係数切り替え
  - **Streaming Cache Reset**: RTPセッション切り替え時のキャッシュフラッシュ

- [x] **Crossfeed/HRTF Engine**
  - バイノーラル処理エンジン実装
  - Web UIからのON/OFF制御

- [x] **EQ Engine**
  - パラメトリックEQ実装
  - Preamp自動推奨機能
  - テキストインポート機能

- [x] **ZeroMQ Communication Layer** ✅
  - 20以上のコマンドタイプ実装完了（LOAD_IR, SET_GAIN, SOFT_RESET, APPLY_EQ, CROSSFEED_*, RTP_*, など）
  - REQ/REP パターン、完全なJSON API
  - Control Plane ↔ Data Plane完全統合
  - 実装: `src/zeromq_interface.cpp`, `src/daemon/control/zmq_server.cpp`

- [x] **Auto-Negotiation Logic** ✅
  - レートファミリー自動判定（44.1k系/48k系）
  - DAC Capability Scan完全実装
  - 最適アップサンプリング率自動選択
  - 実装: `src/auto_negotiation.cpp`, `include/dac_capability.h`

- [x] **Fallback Manager** ✅
  - GPU負荷監視（NVML統合）
  - XRUN自動検出・フォールバック
  - 自動軽量モード移行
  - 実装: `src/fallback_manager.cpp`

- [x] **Error Handling & Logging** ✅
  - エラーコード統一管理（`src/error_codes.cpp`）
  - 構造化ロギング実装済み
  - メトリクス収集（GPU使用率、NVML統合）
  - 実装: `include/logging/logger.h`, `include/logging/metrics.h`

---

## Multi-Rate Support (Critical Feature)

> **Status: ✅ 実装完了** (Issue #231)
>
> GPUUpsamplerは全8入力レートに対応。係数ファイルの生成と配置で動作可能。

### 対応入力レート

| Rate Family | Input Rate | Upsample Ratio | Output Rate | Coefficient File |
|-------------|------------|----------------|-------------|------------------|
| 44.1k系 | 44,100 Hz | 16x | 705,600 Hz | `filter_44k_16x_640k_min_phase.bin` |
| 44.1k系 | 88,200 Hz | 8x | 705,600 Hz | `filter_44k_8x_640k_min_phase.bin` |
| 44.1k系 | 176,400 Hz | 4x | 705,600 Hz | `filter_44k_4x_640k_min_phase.bin` |
| 44.1k系 | 352,800 Hz | 2x | 705,600 Hz | `filter_44k_2x_640k_min_phase.bin` |
| 48k系 | 48,000 Hz | 16x | 768,000 Hz | `filter_48k_16x_640k_min_phase.bin` |
| 48k系 | 96,000 Hz | 8x | 768,000 Hz | `filter_48k_8x_640k_min_phase.bin` |
| 48k系 | 192,000 Hz | 4x | 768,000 Hz | `filter_48k_4x_640k_min_phase.bin` |
| 48k系 | 384,000 Hz | 2x | 768,000 Hz | `filter_48k_2x_640k_min_phase.bin` |

### 実装状況

#### 1. 係数生成 ✅
- [x] 全8構成の最小位相フィルタ生成スクリプト
  ```bash
  uv run python scripts/generate_minimum_phase.py --generate-all --taps 640000
  ```
- [x] 全8構成の線形位相フィルタ生成スクリプト
  ```bash
  uv run python scripts/generate_linear_phase.py --generate-all --taps 640000
  ```
- [x] C++が期待するファイル名パターンに対応 (`filter_{family}_{ratio}x_{taps}_{phase_label}.bin` 例: `_min_phase`)

#### 2. GPUUpsampler Multi-Rate対応 ✅
- [x] `MULTI_RATE_CONFIGS`: 全8構成定義 (`include/convolution_engine.h`)
- [x] `initializeMultiRate()`: 全8構成のFFT事前計算 (`gpu_upsampler_multi_rate.cu`)
- [x] `switchToInputRate()`: グリッチフリー動的切り替え
- [x] ダブルバッファリング（ピンポン方式）

#### 3. 動的レート検知 ✅ (Issue #218)
- [x] PipeWire `param_changed` イベントでのレート検出
- [x] Rate Family判定ロジック (`detectRateFamily()`)
- [x] `handle_rate_change()` による自動切り替え

#### 4. 自動ネゴシエーション ✅
- [x] `AutoNegotiation::negotiate()`: 全8レート対応
- [x] DAC Capability検証
- [x] `requiresReconfiguration` フラグ（ファミリ変更検出）

### データフロー例

```
入力: 96kHz
  │
  ▼
Rate Detection: 48k Family (96000 % 48000 == 0)
  │
  ▼
Load Coefficients: filter_48k_8x_640k_min_phase.bin
  │
  ▼
Strategy: 96k × 8 = 768k (within DAC capability)
  │
  ▼
GPU Processing (640k-tap FIR, 8x upsample)
  │
  ▼
出力: 768kHz
```

### 優先度

**Phase 1の必須タスク**として位置づけ。これがないとMagic Boxとして機能しない。

---

## Phase 2: Control Plane & Web UI

**Status:** ✅ 完了（100%）

システムの頭脳であるPython/FastAPIバックエンドとWeb UIの実装。全機能実装済み。

### Completed Tasks ✅

- [x] **Python/FastAPI Backend**
  - REST API実装（/status, /eq, /daemon, /opra等）
  - OpenAPI自動生成
  - ZeroMQ経由のEngine制御
  - エラーハンドリング統一
  - 詳細: `docs/api/openapi.json`

- [x] **OPRA Integration** (CC BY-SA 4.0)
  - OPRAリポジトリからのEQデータ取得
  - ヘッドホンデータベース構築
  - ブランド・モデル検索機能
  - CC BY-SA 4.0帰属表示対応
  - 実装: `web/routers/opra.py`

- [x] **EQ Engine**
  - Parametric EQ適用（最大10バンド）
  - テキストファイルインポート（AutoEQ形式）
  - Preamp自動推奨機能
  - リアルタイム適用
  - 実装: `src/equalizer.cpp`, `web/routers/eq.py`

- [x] **Web Frontend** ✅
  - **軽量JSフレームワーク**: htmx 1.9.12 + Alpine.js 3.13.10
  - **i18n完全対応**: 日本語/英語切り替え、テスト済み (`web/tests/test_i18n.py`)
  - **レスポンシブデザイン**: モバイルメニュー、タッチ操作対応
  - ヘッドホン選択UI（ブランド・モデル検索）
  - リアルタイムステータス表示（htmx polling）
  - EQプロファイル管理（Drag & Drop対応）
  - クロスフィード/HRTF制御パネル
  - 実装: `web/templates/`, `web/static/css/`

- [x] **テストカバレッジ** ✅
  - i18n テスト (`web/tests/test_i18n.py`)
  - EQ Settings ページテスト (`web/tests/test_eq_settings_page.py`)
  - モデル検証テスト (`web/tests/test_models.py`)

- [x] **Dependencies完全統合**
  - pyzmq（ZeroMQ Python binding）
  - aiofiles（非同期ファイルI/O）
  - httpx（HTTP client）
  - pydantic（データ検証）
  - jinja2（テンプレートエンジン）

### UX Achievement ✅
- **Ultimate Simplicity実現**: ヘッドホンを選ぶ → 適用ボタン → 完了
- 技術的詳細は隠蔽、詳細モードで表示可能
- 多言語対応で国際展開準備完了
- **5000行超のOpenAPI仕様書**自動生成済み (`docs/api/openapi.json`)

---

## Phase 2.5: Refactoring & Stabilization

**Status:** 🔄 進行中（85%）

Phase 2で急速に実装した機能の品質向上・安定化。技術的負債の解消とエッジケース対応。

### 背景

Phase 2で基本機能は完成したが、以下の課題が顕在化：
- 急速な開発による技術的負債
- 巨大化したファイル（`alsa_daemon.cpp`等）の責務不明瞭
- エッジケースでの不安定性
- 可読性・保守性の低下

Phase 3（ハードウェア統合）に進む前に、コードベースを健全化。

### Completed Tasks ✅

- [x] **Web UIフレームワーク刷新**
  - レガシーHTML → **htmx 1.9.12 + Alpine.js 3.13.10**
  - 約半分のコードを作り直し
  - モダンなリアクティブUI実現
  - 実装: `web/templates/base.html`

- [x] **i18n対応実装**
  - 日本語/英語の完全対応
  - テンプレートエンジン統合
  - テストカバレッジ確保
  - 実装: `web/tests/test_i18n.py`

- [x] **コードリファクタリング（部分的）**
  - 可読性向上
  - 関数分割・モジュール化

### In Progress 🔄

- [ ] **alsa_daemon.cpp の責務分割**
  - 現状: 巨大なモノリシックファイル
  - 目標:
    - `audio_input.cpp` - ALSA/PipeWire/RTP入力管理
    - `audio_output.cpp` - ALSA出力管理
    - `processing_pipeline.cpp` - GPU処理パイプライン
    - `daemon_main.cpp` - メインループ・初期化
  - Issue: 要作成

- [ ] **HRTF/Crossfeed バグ修正**
  - 既知のバグ: クロスフェード処理の問題
  - エッジケースでの不安定性
  - Issue: 既存

- [ ] **エッジケーステスト強化**
  - レート切り替え繰り返し
  - RTP接続/切断の頻繁な繰り返し
  - GPU高負荷時の挙動
  - 長時間稼働テスト（メモリリーク検出）

- [ ] **非通常系テスト**
  - デバイス切断時の挙動
  - 不正なRTP/SDP受信時の挙動
  - CUDA/GPU エラー時の復帰

### 技術的負債解消の優先順位

1. **Critical**: HRTF/Crossfeedバグ修正 → Phase 3前に必須
2. **High**: `alsa_daemon.cpp` 分割 → 保守性向上
3. **Medium**: エッジケーステスト強化 → 安定性向上

### Timeline

Phase 2.5は明日（Day 15-16）中に完了予定。Phase 3開始前にコードベースを健全化。

---

## Phase 3: Hardware Integration & Deployment

**Status:** 📋 Planned

**アーキテクチャ:** I/O分離構成
- **Raspberry Pi 5**: UAC2デバイス + RTP送信
- **Jetson Orin Nano**: RTP受信 + GPU処理 + DAC出力

### Tasks

#### Raspberry Pi 5 セットアップ（Universal Audio Input Hub）

##### Phase 3.1: 基本構成
- [ ] **USB Gadget Mode (UAC2)**
  - USB Type-C Device Mode設定
  - Linux ConfigFS設定スクリプト作成
  - **ハイレゾ対応**: 16/24/32-bit, 最大768kHz
  - PCからは「高音質USBサウンドカード」として認識

- [ ] **PipeWire RTP送信**
  - UAC2入力 → PipeWire → **SDP生成** → RTP送信
  - **ハイレゾ透過**: 入力レート/ビット深度をそのままJetsonへ転送
  - 自動サンプルレート検知
  - Jetsonへのネットワーク転送

- [ ] **Docker化 (Raspberry Pi)**
  - PipeWire + RTP Sender コンテナ
  - systemd による自動起動
  - ヘルスチェック機能

##### Phase 3.2: ネットワークオーディオ対応
- [ ] **Spotify Connect統合**
  - librespot を使用
  - 320kbps Ogg Vorbis → 44.1kHz/16bit
  - デバイス名: "Magic Box"

- [ ] **AirPlay 2対応**
  - shairport-sync を使用
  - Apple Lossless (ALAC) → 44.1kHz/16bit
  - iOS/macOS からの無線再生

- [ ] **PipeWire入力ソース管理**
  - 複数入力の自動切り替え（Last Active Wins）
  - または優先順位制御（USB > Roon > Spotify > AirPlay）
  - Web UIでの入力ソース選択機能

##### Phase 3.3: 高度なネットワークオーディオ
- [ ] **Roon Bridge統合**
  - Roon Ready認証（オプション）
  - ハイレゾストリーミング対応
  - Roonコア自動検出

- [ ] **UPnP/DLNA Renderer**
  - 家庭内NASからの再生
  - ハイレゾファイル対応（FLAC/DSD）

- [ ] **入力ソース拡張（オプション）**
  - Bluetooth A2DP (aptX HD対応)
  - Chromecast Audio プロトコル

#### Jetson Orin Nano セットアップ
- [ ] **Jetson移植**
  - CUDA Architecture変更 (SM 7.5 → SM 8.7)
  - CMakeLists.txt の CUDA_ARCHITECTURES 修正
  - パフォーマンス検証・チューニング

- [x] **RTP受信機能（実装済み）** ✅
  - RTP Session Manager統合完了
  - **SDP自動パース**: サンプルレート/ビット深度/チャンネル数を自動認識
  - **ハイレゾ対応**: 16/24/32-bit, 44.1k〜768kHz
  - **動的追従**: RTPストリームのレート変更に自動追従
  - **グリッチフリー切り替え**: Soft Mute機能によるシームレスなレート変更
  - 実装ファイル: `src/rtp_session_manager.cpp`, `src/alsa_daemon.cpp`

- [ ] **ALSA Direct Output**
  - USB DAC直接出力
  - Bit-perfect転送
  - デバイス自動検出

- [ ] **Docker化 (Jetson)**
  - C++ Daemon + CUDA Runtime コンテナ
  - Python Web UI コンテナ
  - docker-compose による統合管理
  - GPU パススルー設定

#### 統合・監視
- [ ] **System Integration**
  - systemd によるDocker自動起動
  - ネットワーク設定（Wi-Fi/Ethernet）
  - mDNS設定（magicbox.local）

- [ ] **Performance Optimization**
  - メモリ帯域最適化（Unified Memory活用）
  - GPU負荷最適化
  - 熱管理（ファン制御、スロットリング回避）
  - 消費電力最適化

- [ ] **Deployment Automation**
  - デプロイスクリプト作成
  - OTA アップデート機構
  - 工場出荷時リセット機能

### Hardware Specifications

#### Raspberry Pi 5 (Input Bridge)
| Item | Specification |
|------|---------------|
| SoC | Broadcom BCM2712 (Quad-core Cortex-A76) |
| Role | USB UAC2デバイス、RTP送信 |
| Input | USB Type-C (UAC2 Device Mode) |
| Output | Ethernet → Jetson |
| Deployment | Docker |

#### Jetson Orin Nano Super (Processing Unit)
| Item | Specification |
|------|---------------|
| SoC | NVIDIA Jetson Orin Nano Super (8GB) |
| CUDA Cores | 1024 |
| CUDA Arch | SM 8.7 (Ampere) |
| Storage | 1TB NVMe SSD (KIOXIA EXCERIA G2) |
| Input | RTP over Ethernet |
| Output | USB Type-A → External USB DAC |
| Network | Wi-Fi / Ethernet |
| Deployment | Docker (CUDA Runtime) |

---

## Phase 4: Commercialization & Deployment Ecosystem

**Status:** 📋 Planned

プロトタイプから製品化への移行。デプロイ自動化とライセンス管理。

### Tasks

#### 4.1: CI/CD & Container Registry

- [ ] **マルチアーキテクチャDockerイメージ**
  - GitHub Actions CI/CD構築
  - `linux/amd64` (開発用) + `linux/arm64` (Jetson用) 同時ビルド
  - GitHub Container Registry (ghcr.io) へのプッシュ
  - バージョンタグ管理 (`latest`, `v1.0.0`, `sha-xxxxx`)

- [ ] **プリビルドバイナリ配布**
  - GitHub Releases経由での配布
  - CUDA Runtime最小化
  - ダウンロードサイズ最適化

- [ ] **自動デプロイスクリプト**
  ```bash
  curl -fsSL https://get.magicbox.io/install.sh | sh
  # → Docker pull → docker-compose up -d
  ```

#### 4.2: ライセンス認証システム

- [ ] **ライセンスサーバ**
  - デバイス固有ID (Jetson Serial Number)
  - ライセンスキー発行API
  - オンライン認証（初回起動時）
  - オフライン猶予期間（30日）

- [ ] **バイナリ保護**
  - デーモンバイナリの暗号化
  - ライセンス検証失敗時の機能制限
  - 試用版モード（機能制限あり、7日間）

- [ ] **ライセンス管理ダッシュボード**
  - 販売店向けライセンス発行UI
  - デバイス登録状況確認
  - 使用統計・テレメトリ（オプトイン）

#### 4.3: OTA (Over-The-Air) アップデート

- [ ] **自動更新機能**
  - Dockerイメージの自動pull
  - ローリングアップデート（無音切断最小化）
  - ロールバック機能

- [ ] **更新通知**
  - Web UI上での更新通知
  - メールアラート（販売店向け）

#### 4.4: 品質保証・テスト

- [ ] **E2Eテスト自動化**
  - ハードウェアインザループテスト
  - オーディオ品質自動検証

- [ ] **パフォーマンスベンチマーク**
  - Jetson実機での性能計測
  - レイテンシ・スループット計測

- [ ] **長時間安定性テスト**
  - 24時間連続動作テスト
  - メモリリーク検出

#### 4.5: ドキュメント・サポート

- [ ] **ユーザーマニュアル**
  - セットアップガイド
  - トラブルシューティング

- [ ] **販売店向けドキュメント**
  - 初期設定手順
  - ライセンス発行手順

- [ ] **開発者向けAPI ドキュメント**
  - OpenAPI仕様書公開
  - SDK提供（カスタマイズ用）

### Timeline
Phase 4は製品リリース後、継続的に改善。初期バージョンは6-8週間で完成目標。

---

## Future Enhancements (Post-Phase 4)

将来の拡張機能（優先度順）

### High Priority
- [ ] **Multiple Headphone Profiles**
  - プロファイル保存・切り替え
  - クロスフェードでのシームレス切り替え

### Medium Priority
- [ ] **Room Correction**
  - マイク測定によるルーム補正

- [ ] **Reverb Engine**
  - 空間系エフェクト
  - プリセット（Hall, Room, etc.）

### Low Priority
- [ ] **Multi-DAC Support**
  - 複数DAC同時出力
  - DAC個別設定

- [ ] **Mobile App**
  - iOS/Android制御アプリ

---

## Legal & License Management

製品化に向けたライセンス管理。**商用利用禁止のライブラリ/データを使用しないこと。**

### EQデータソースのライセンス

| ソース | ライセンス | 商用利用 | 備考 |
|--------|-----------|----------|------|
| **OPRA** | CC BY-SA 4.0 | ✅ OK | 帰属表示必須、派生物も同ライセンス |
| oratory1990 | 独自 | ❌ 禁止 | ライセンス交渉必要 |
| AutoEQ (ソフト) | MIT | ✅ OK | - |
| AutoEQ (データ) | 元データ依存 | ⚠️ 要確認 | oratory1990データ含む場合NG |

### 依存ライブラリのライセンス

| ライブラリ | ライセンス | 商用利用 | 注意点 |
|-----------|-----------|----------|--------|
| CUDA/cuFFT | NVIDIA EULA | ✅ OK | 再配布制限あり |
| libsndfile | LGPL-2.1 | ✅ OK | 動的リンク推奨 |
| libpipewire | MIT | ✅ OK | - |
| alsa-lib | LGPL-2.1 | ✅ OK | 動的リンク推奨 |
| libsoxr | LGPL-2.1 | ✅ OK | 動的リンク推奨 |
| nlohmann/json | MIT | ✅ OK | - |
| scipy/numpy | BSD-3 | ✅ OK | - |
| FastAPI | MIT | ✅ OK | - |
| ZeroMQ | LGPL-3.0 | ✅ OK | 動的リンク推奨 |

### LGPLライブラリの取り扱い

LGPL（libsndfile, alsa-lib, libsoxr, ZeroMQ）は以下の条件で商用利用可能：
- **動的リンク**（.so共有ライブラリとしてリンク）
- ユーザーがライブラリを差し替え可能であること
- LGPLライセンス文の同梱

### 必須タスク

- [ ] **ライセンス監査実施**
  - 全依存ライブラリのライセンス確認
  - ライセンス互換性チェック

- [ ] **帰属表示ファイル作成**
  - NOTICE.md / THIRD_PARTY_LICENSES.md
  - OPRA帰属表示（CC BY-SA 4.0要件）

- [ ] **CI/CDでのライセンスチェック**
  - license-checker / FOSSA 等の導入検討

- [ ] **禁止ライセンスの明文化**
  - GPL（静的リンク時）
  - 商用利用禁止ライセンス
  - 帰属表示漏れ

---

## Infrastructure & Quality

### CI/CD

- [ ] **GitHub Actions設定**
  - C++/CUDAビルドチェック
  - Pythonテスト・lint
  - ライセンスチェック

### Testing

- [ ] **C++ユニットテスト**
  - Google Test導入
  - convolution_engine テスト
  - config_loader テスト

- [ ] **Pythonテスト**
  - pytest導入
  - フィルタ生成テスト
  - Web API テスト

- [ ] **Integration Test**
  - E2Eテスト（入力→出力検証）
  - パフォーマンス回帰テスト

### Documentation

- [ ] **ユーザーガイド**
  - セットアップ手順
  - トラブルシューティング
  - FAQ

- [ ] **API ドキュメント**
  - REST API仕様（OpenAPI/Swagger）
  - ZeroMQ メッセージ仕様

### Deployment

- [ ] **インストールスクリプト**
  - ワンコマンドセットアップ
  - 依存関係自動インストール

- [ ] **リリース自動化**
  - バージョニング（SemVer）
  - CHANGELOGの自動生成

---

## Technical Dependencies

### Phase 1 Dependencies
- CUDA Toolkit 12.x
- cuFFT
- PipeWire (libpipewire)
- ALSA (alsa-lib)
- libsoxr
- ZeroMQ (libzmq)

### Phase 2 Dependencies
- Python 3.11+
- FastAPI
- scipy, numpy
- uvicorn
- aiofiles

### Phase 3 Dependencies
- JetPack SDK 6.x
- Linux ConfigFS (USB Gadget)
- systemd

---

## Reference Documents

- [Architecture Overview](architecture/overview.md)
- [Phase 1 Implementation Report](reports/phase1_implementation_report.md)
- [Phase 2 Implementation Report](reports/phase2_implementation_report.md)
- [Setup Guide](setup/pc_development.md)
