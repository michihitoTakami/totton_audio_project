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
  - 2M-tap minimum phase FIR filter実装完了
  - ~28x realtime performance on RTX 2070S
  - Overlap-Save方式によるストリーミング処理

- [x] **Filter Coefficient Generation**
  - scipy.signalによる2Mタップフィルタ生成
  - 197dB stopband attenuation達成
  - Kaiser window (β=55) 適用

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

- [ ] **Multi-Rate Support (Critical)**
  - 詳細は下記「Multi-Rate Support」セクション参照
  - 44.1k系/48k系両方の係数セット生成・管理
  - 入力レート変更時の動的係数切り替え

### Pending

- [ ] **Safety Mechanisms**
  - Soft Mute（レート切り替え時クロスフェード）
  - Dynamic Fallback（XRUN時の軽量モード移行）
  - Hot-swap IR loading

---

## Multi-Rate Support (Critical Feature)

**現状の問題**: 44.1kHz系のみ対応。48kHz系やその倍数（88.2k, 96k, 192k等）が来た時に対応できない。
**Magic Boxとして必須**: どんな入力レートでも自動的に最適処理できないと意味がない。

### 対応すべき入力レート

| Rate Family | Input Rates | Output Target (16x max) |
|-------------|-------------|-------------------------|
| 44.1k系 | 44,100 / 88,200 / 176,400 Hz | 705,600 Hz |
| 48k系 | 48,000 / 96,000 / 192,000 Hz | 768,000 Hz |

### 必要な係数セット

各Rate Family用に別々のFIR係数が必要（ストップバンド周波数が異なるため）

| Family | Passband | Stopband Start | Coefficient File |
|--------|----------|----------------|------------------|
| 44.1k系 | 0-20kHz | 22.05kHz | `filter_44k_2m_min_phase.bin` |
| 48k系 | 0-22kHz | 24.0kHz | `filter_48k_2m_min_phase.bin` |

### 実装タスク

#### 1. 係数生成 (Phase 1)
- [ ] 48kHz系用2M-tap係数の生成
  ```bash
  uv run python scripts/generate_filter.py \
    --input-rate 48000 \
    --stopband-start 24000 \
    --passband-end 22000 \
    --output-prefix filter_48k_2m_min_phase
  ```
- [ ] 両係数セットのバリデーション・検証

#### 2. 動的レート検知 (Phase 1)
- [ ] PipeWire/ALSA入力のサンプルレート変更イベント検知
- [ ] Rate Family判定ロジック
  ```
  if (rate % 44100 == 0) → 44.1k Family
  if (rate % 48000 == 0) → 48k Family
  ```

#### 3. 係数ホットスワップ (Phase 1)
- [ ] Rate Family変更時の係数切り替え
- [ ] ダブルバッファリングによるグリッチレス切り替え
- [ ] Soft Mute（切り替え時のポップノイズ防止）

#### 4. リサンプリング統合 (Phase 1)
- [ ] libsoxrによる入力レート正規化
  - 88.2k → 44.1k → 処理 → 705.6k
  - 96k → 48k → 処理 → 768k
- [ ] または高レート入力を直接処理
  - 96k × 8 = 768k（係数は48k系を使用）

### データフロー例

```
入力: 96kHz
  │
  ▼
Rate Detection: 48k Family (96000 % 48000 == 0)
  │
  ▼
Load Coefficients: filter_48k_2m_min_phase.bin
  │
  ▼
Strategy: 96k × 8 = 768k (within DAC capability)
  │
  ▼
GPU Processing (2M-tap FIR, 8x upsample)
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

### Tasks

- [ ] **Python/FastAPI Backend**
  - REST API設計
  - WebSocket対応（リアルタイムステータス）
  - ZeroMQ経由のEngine制御

- [ ] **oratory1990 Integration**
  - AutoEQデータの取得・パース
  - ヘッドホンデータベース構築
  - 検索・フィルタリング機能

- [ ] **IR Generator**
  - oratory1990データ + KB5000_7ターゲット合成
  - 最小位相IR生成（scipy）
  - Dual Target Generation（44.1k系/48k系）
  - Filter 11追加: `ON PK Fc 5366 Hz Gain 2.8 dB Q 1.5`

- [ ] **Web Frontend**
  - ヘッドホン選択UI（シンプルなリスト/検索）
  - ステータス表示（入力レート、出力レート、GPU負荷）
  - 設定変更（ターゲットカーブ調整は将来機能）

### UX Goal
- ヘッドホンを選ぶ → 適用ボタン → 完了
- 技術的詳細は隠す（詳細モードで表示可能にはする）

---

## Phase 3: Hardware Integration

**Status:** 📋 Planned

Jetson Orin Nano Superへの移植と製品化。

### Tasks

- [ ] **Jetson Orin Nano Migration**
  - CUDA Architecture変更 (SM 7.5 → SM 8.7)
  - NVMLオプショナル化
  - パフォーマンス検証・チューニング

- [ ] **USB Gadget Mode Setup**
  - USB Type-C Device Mode (UAC2)
  - PCからは「高音質USBサウンドカード」として認識
  - Linux ConfigFS設定

- [ ] **ALSA Direct Output**
  - USB DAC直接出力
  - Bit-perfect転送
  - デバイス自動検出

- [ ] **System Integration**
  - Systemdサービス化
  - 自動起動設定
  - ネットワーク設定（Wi-Fi/Ethernet）

- [ ] **Performance Optimization**
  - メモリ帯域最適化
  - GPU負荷最適化
  - 熱管理

### Hardware Specifications

| Item | Specification |
|------|---------------|
| SoC | NVIDIA Jetson Orin Nano Super (8GB) |
| CUDA Cores | 1024 |
| Storage | 1TB NVMe SSD (KIOXIA EXCERIA G2) |
| Input | USB Type-C (UAC2 Device Mode) |
| Output | USB Type-A → External USB DAC |
| Network | Wi-Fi / Ethernet |

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
