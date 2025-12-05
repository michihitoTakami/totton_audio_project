# Magic Box Project - Development Roadmap

## Vision

**全てのヘッドホンユーザーに最高の音を届ける箱**

- 箱をつなぐ → 管理画面でポチポチ → 最高の音
- ユーザーに余計なことを考えさせない

## Phase Overview

```
Phase 1: Core Engine & Middleware     [=========>          ] 60%
Phase 2: Control Plane & Web UI       [                    ] 0%
Phase 3: Hardware Integration         [                    ] 0%
```

---

## Phase 1: Core Engine & Middleware

**Status:** 🔄 In Progress

システムの心臓部であるC++ Audio Engine Daemonの完成を目指す。

### Completed Tasks

- [x] **GPU Convolution Algorithm**
  - 640k-tap minimum phase FIR filter実装完了
  - ~28x realtime performance on RTX 2070S
  - Overlap-Save方式によるストリーミング処理

- [x] **Filter Coefficient Generation**
  - scipy.signalによる640kタップフィルタ生成
  - ~160dB stopband attenuation（24bit品質に十分）
  - Kaiser window (β≈28) - 32bit Float実装の量子ノイズ限界に合わせた最適値

- [x] **Low-Latency Partition Validation** (#355)
  - `scripts/inspect_impulse.py` / `verify_frequency_response.py` をpartition対応
  - PipeWire/ALSAループバックとXRUN/GPU監視フローを `docs/investigations/low_latency_partition_validation.md` に記録
  - QAチェックリストへ低遅延モードの回帰項目を追加

- [x] **Phase Type Selection** (#165, #166, #167)
  - Minimum Phase / Linear Phase 切り替え機能
  - `scripts/generate_linear_phase.py` による線形位相フィルタ生成
  - C++/CUDA側の位相タイプ対応（遅延計算含む）
  - 設定システム（`PhaseType` enum）

- [x] **Basic Daemon Implementation**
  - PipeWire入力 → GPU処理 → ALSA出力
  - Working prototype動作確認済み

### In Progress

- [ ] **C++ Daemon Refinement**
  - libsoxr統合（可変レートリサンプリング）
  - エラーハンドリング強化
  - メモリ管理最適化

- [ ] **ZeroMQ Communication Layer**
  - Control Plane ↔ Data Plane通信
  - コマンド：係数ロード、ソフトリセット、ステータス取得
  - IPC (Inter-Process Communication) 実装

- [ ] **Auto-Negotiation Logic**
  - DAC Capability Scan（ALSA経由）
  - Input Rate Detection（44.1k vs 48k系）
  - Optimal Upsampling Rate計算

- [x] **Multi-Rate Support (Critical)** ✅ Issue #231
  - 詳細は下記「Multi-Rate Support」セクション参照
  - 44.1k系/48k系両方の係数セット生成・管理
  - 入力レート変更時の動的係数切り替え

### Pending

- [ ] **Safety Mechanisms**
  - Soft Mute（レート切り替え時クロスフェード）
  - Dynamic Fallback（XRUN時の軽量モード移行）
  - Hot-swap IR loading

- [ ] **Logging & Monitoring**
  - 構造化ロギング導入（spdlog推奨）
  - ファイルへのログ出力
  - メトリクス収集（GPU負荷、バッファ状態、XRUN回数）

- [ ] **Error Handling Enhancement**
  - CUDA エラーの適切なハンドリング
  - ALSA/PipeWire エラーからの復帰
  - グレースフルシャットダウン

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

**Status:** 📋 Planned

システムの頭脳であるPython/FastAPIバックエンドとWeb UIの実装。

### Completed (Partial)

- [x] **Basic Web API** (web/main.py)
  - REST API（/status, /settings, /restart等）
  - 埋め込みHTML UI
  - EQプロファイル管理

### Tasks

- [ ] **Python/FastAPI Backend Enhancement**
  - REST API設計の改善
  - WebSocket対応（リアルタイムステータス）
  - ZeroMQ経由のEngine制御（現在はSIGHUP）
  - 認証機能（オプション、ネットワーク公開時）

- [ ] **OPRA Integration** (CC BY-SA 4.0)
  - OPRAリポジトリからのEQデータ取得
  - ヘッドホンデータベース構築（SQLite or JSON）
  - ブランド・モデル検索機能
  - データ更新機能（定期同期）
  - ⚠️ 帰属表示必須（CC BY-SA 4.0要件）

- [ ] **IR Generator**
  - OPRAデータ + KB5000_7ターゲット合成
  - 最小位相IR生成（scipy homomorphic processing）
  - Dual Target Generation（44.1k系/48k系）
  - Filter 11追加: `ON PK Fc 5366 Hz Gain 2.8 dB Q 1.5`
  - 生成済みIRのキャッシュ管理

- [ ] **Web Frontend**
  - ヘッドホン選択UI（シンプルなリスト/検索）
  - ステータス表示（入力レート、出力レート、GPU負荷）
  - 設定変更（ターゲットカーブ調整は将来機能）
  - レスポンシブデザイン（スマホ対応）

- [ ] **Dependencies to Add**
  - pyzmq（ZeroMQ Python binding）
  - aiofiles（非同期ファイルI/O）
  - httpx（AutoEQデータ取得）
  - websockets（リアルタイム通信）

### UX Goal
- ヘッドホンを選ぶ → 適用ボタン → 完了
- 技術的詳細は隠す（詳細モードで表示可能にはする）

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

## Future Enhancements (Post-Phase 3)

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
