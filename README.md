# Magic Box Project - 魔法の箱

**全てのヘッドホンユーザーに最高の音を届ける箱**

## ビジョン

**究極のシンプルさ**:
1. 箱をつなぐ
2. 管理画面でポチポチ
3. 最高の音

ユーザーに余計なことを考えさせない。ヘッドホンを選んで、ボタンを押すだけ。

## 何ができるか

- **究極のアップサンプリング**: 640,000タップ最小位相FIRフィルタ（Kaiser β≈28、~160dB ストップバンド減衰）
- **ヘッドホン補正EQ**: OPRAプロジェクト（CC BY-SA 4.0）のEQデータを活用し、あなたのヘッドホンをターゲットカーブ（KB5000_7）に自動補正
- **Auto-Negotiation**: DAC性能を自動検出し、最適なアップサンプリング倍率を自動算出
- **Multi-Rate対応**: 44.1kHz系/48kHz系を自動判定し、適切な係数に切り替え
- **Crossfeed (HRTF)**: オプションでヘッドホンのステレオ感を自然な音場に補正
- **Soft Mute**: レート切り替え時のフェード制御でポップノイズを防止

---

## システムアーキテクチャ

### 全体構成

```
┌─────────────────────────────────────────────────────────────────┐
│                    Control Plane (Python/FastAPI)               │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │  Web UI  │  │ IR Generator │  │      Orchestrator          │ │
│  │ (React)  │  │   (scipy)    │  │ (ZeroMQ Client)            │ │
│  └────┬─────┘  └──────────────┘  └─────────────┬──────────────┘ │
│       │                                        │                │
│       └────────────── REST/WS ─────────────────┘                │
│                         │                                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │ ZeroMQ IPC (REQ/REP + PUB/SUB)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Plane (C++ Audio Engine)                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Audio Processing Pipeline               │   │
│  │  ALSA Loopback Input (RTP feed) → Rate Detection → GPU FFT → Crossfeed → ALSA  │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌────────────┐ ┌─────────────┐ ┌───────────┐ ┌─────────────┐  │
│  │ Auto-Nego  │ │  Soft Mute  │ │ ZeroMQ Srv│ │ DAC Detect  │  │
│  └────────────┘ └─────────────┘ └───────────┘ └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### オーディオ処理パイプライン

```mermaid
flowchart LR
    subgraph Input
        IN[ALSA Loopback Input<br/>44.1k/48k/96k/... (RTP)]
    end

    subgraph Processing
        RD[Rate Detection<br/>44k系 or 48k系]
        RB1[Ring Buffer<br/>Lock-free]
        GPU[GPU FFT Convolution<br/>640k-tap FIR<br/>cuFFT Overlap-Save]
        CF[Crossfeed<br/>HRTF<br/>Optional]
        SM[Soft Mute<br/>Fade Control]
        RB2[Ring Buffer<br/>Lock-free]
    end

    subgraph Output
        ALSA[ALSA Output<br/>705.6k/768k]
        DAC[USB DAC]
    end

    IN --> RD --> RB1 --> GPU --> CF --> SM --> RB2 --> ALSA --> DAC
```

### ZeroMQ通信フロー

```mermaid
sequenceDiagram
    participant Web as Web UI
    participant Py as Python Controller
    participant ZMQ as ZeroMQ
    participant Cpp as C++ Engine

    Web->>Py: POST /api/eq/apply
    Py->>ZMQ: APPLY_EQ {params}
    ZMQ->>Cpp: Command
    Cpp->>Cpp: Apply EQ
    Cpp->>ZMQ: {status: "ok"}
    ZMQ->>Py: Response
    Py->>Web: 200 OK

    Note over Cpp,ZMQ: Status Updates (PUB/SUB)
    Cpp-->>ZMQ: {input_rate: 44100, gpu_load: 15%}
    ZMQ-->>Py: Status Event
    Py-->>Web: WebSocket Push
```

### ネットワーク入力 (現行: RTP/ALSA)

- Jetson: GStreamerベースの `web/services/rtp_input.py` が RTP/RTCP パイプラインを駆動し、ALSA Loopback playback (`hw:Loopback,0,0`) に音声を注入
- PC: ALSA Loopback capture または直接 ALSA デバイスで稼働 (必要に応じて `raspberry_pi/rtp_sender.py` で送信)

---

## 主要機能

### Auto-Negotiation（自動調停）

DAC性能と入力レートから最適な出力レートを自動算出します。

```
┌─────────────────────────────────────────────────────────────┐
│                    Auto-Negotiation Flow                     │
├─────────────────────────────────────────────────────────────┤
│  1. DAC Capability Scan                                      │
│     └─ ALSA経由で最大サンプリングレート取得 (例: 768kHz)      │
│                                                              │
│  2. Input Rate Detection                                     │
│     └─ 44.1k系: 44100, 88200, 176400 Hz                     │
│     └─ 48k系:   48000, 96000, 192000 Hz                     │
│                                                              │
│  3. Optimal Rate Calculation                                 │
│     └─ target = max(input × N) where input × N ≤ dac_max    │
│     └─ 例: 44100 × 16 = 705600 ≤ 768000 ✓                   │
│                                                              │
│  4. IR Selection                                             │
│     └─ 44.1k系 → filter_44k_2m_minimum.bin                  │
│     └─ 48k系   → filter_48k_2m_minimum.bin                  │
└─────────────────────────────────────────────────────────────┘
```

| 入力レート | DAC Max | 出力レート | 倍率 | 係数セット |
|-----------|---------|-----------|------|-----------|
| 44.1kHz | 768kHz | 705.6kHz | 16x | 44k系 |
| 48kHz | 768kHz | 768kHz | 16x | 48k系 |
| 96kHz | 768kHz | 768kHz | 8x | 48k系 |
| 192kHz | 768kHz | 768kHz | 4x | 48k系 |

### Soft Mute（グリッチ防止）

レート切り替え時のポップノイズを防止するフェード制御機能です。

```
同族内切り替え (44.1k → 88.2k): グリッチレス（即座）
クロス族切り替え (44.1k → 48k):  Soft Mute適用（~50ms fade）
```

### Crossfeed (HRTF)

オプション機能。ヘッドホンの過度なステレオ分離を補正し、より自然なスピーカーライクな音場を実現します。

- 4つの頭サイズ（S, M, L, XL）に対応
- HRTFベースの畳み込み処理
- グリッチレス切り替え（ダブルバッファリング）

---

## 動作環境

### 開発環境（PC）

| 項目 | 仕様 |
|-----|-----|
| GPU | NVIDIA RTX 2070 Super (8GB VRAM) 以上 |
| CUDA | SM 7.5 (Turing) |
| OS | Linux (Ubuntu 22.04+) |
| オーディオ | ALSA Loopback (RTP) |

### 本番環境（Magic Box）

| 項目 | 仕様 |
|-----|-----|
| ハードウェア | NVIDIA Jetson Orin Nano Super (8GB, 1024 CUDA Cores) |
| CUDA | SM 8.7 (Ampere) |
| ストレージ | 1TB NVMe SSD (KIOXIA EXCERIA G2) |
| 入力 | USB Type-C (UAC2 Device Mode) - PCからはUSBオーディオデバイスとして認識 |
| 出力 | USB Type-A → 外部USB DAC |
| 管理 | Wi-Fi / Ethernet 経由のWeb UI |

---

## パフォーマンス

| メトリクス | 目標 | 実績 (RTX 2070S) |
|-----------|------|-----------------|
| Realtime Factor | > 1x | **~28x** |
| GPU Utilization | < 50% | **~15%** |
| Latency | < 100ms | **~50ms** |
| XRUN Rate | 0 | **0** (安定) |
| Stopband Attenuation | ~197dB | **197dB** |

---

## クイックスタート（PC開発環境）

### 1. フィルタ係数生成

```bash
# 環境セットアップ
uv sync

# 44.1k系 2M-tap 最小位相フィルタ（基準）
uv run python scripts/filters/generate_minimum_phase.py --taps 2000000

# 44.1k系 2M-tap 線形位相フィルタ
uv run python scripts/filters/generate_linear_phase.py --taps 2000000

# 全構成（44k/48k × 2x/4x/8x/16x）最小位相フィルタを一括生成
uv run python scripts/filters/generate_minimum_phase.py --generate-all

# 全構成（44k/48k × 2x/4x/8x/16x）線形位相フィルタを一括生成
uv run python scripts/filters/generate_linear_phase.py --generate-all

# 生成されるフィルタ:
# - filter_44k_16x_2m_min_phase.bin (44.1kHz → 705.6kHz)
# - filter_44k_8x_2m_min_phase.bin  (88.2kHz → 705.6kHz)
# - filter_44k_4x_2m_min_phase.bin  (176.4kHz → 705.6kHz)
# - filter_44k_2x_2m_min_phase.bin  (352.8kHz → 705.6kHz)
# - filter_48k_16x_2m_min_phase.bin (48kHz → 768kHz)
# - filter_48k_8x_2m_min_phase.bin  (96kHz → 768kHz)
# - filter_48k_4x_2m_min_phase.bin  (192kHz → 768kHz)
# - filter_48k_2x_2m_min_phase.bin  (384kHz → 768kHz)
# 各フィルタには対応するメタデータJSON（DCゲイン・レート情報含む）が生成されます
```

### 2. ビルド

```bash
# PC用ビルド (SM 7.5)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

#### バックエンド切り替え（CUDA / Vulkan）
- デフォルト: `ENABLE_CUDA=ON` / `ENABLE_VULKAN=OFF`（従来どおりCUDA必須）
- Vulkanのみでのビルド（Pi5やVulkanツールチェーン確認用）
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DENABLE_VULKAN=ON
  cmake --build build -j$(nproc) vkfft_minimal  # VkFFT最小サンプル
  cmake --build build -j$(nproc) vulkan_overlap_save_tool  # Overlap-Saveオフライン検証
  ```
  - `ENABLE_VULKAN=ON` で `GPU_UPSAMPLER_BUILD_VKFFT_SAMPLE` が自動有効化され、VkFFT最小サンプル `vkfft_minimal` でVulkan toolchainを確認できます。
  - Overlap-Save サンプルの詳細は `docs/jetson/vulkan_overlap_save_offline.md` を参照。
- CUDAとVulkanを併用する場合
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DENABLE_VULKAN=ON
  cmake --build build -j$(nproc)
  ```

Vulkan環境確認（Jetson Orin Nano / PC 共通の抜粋）
- ドライバ/ICD: Jetsonは JetPack 6.x(L4T r36.x)、PCは推奨版 `nvidia-driver-XXX`。`ls /usr/share/vulkan/icd.d/nvidia_icd.json` でICD確認。
- ツール: `sudo apt update && sudo apt install -y vulkan-tools libvulkan1`
- 動作確認: `vulkaninfo --summary | grep -E "deviceName|Compute"` で compute queue が見えること。
- トラブルシュート: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` を指定、/dev/nvidia* の権限は `sudo usermod -aG video,audio $USER` 後に再ログイン。
- Pi5向けDockerビルドでも `-DENABLE_CUDA=OFF -DENABLE_VULKAN=ON` を渡せば CUDAToolkit 非依存でCMakeが通る構成にしています。

### 3. テスト実行

```bash
# CPUテスト（GPU不要）
./build/cpu_tests
./build/zmq_tests
./build/auto_negotiation_tests
./build/soft_mute_tests

# GPUテスト（GPU必要）
./build/gpu_tests
./build/crossfeed_tests
```
※ `ENABLE_CUDA=OFF` で構成した場合、CUDA依存の `cpu_tests` / `auto_negotiation_tests` / `gpu_tests` / `crossfeed_tests` は生成されず、ZeroMQ/Base64/SoftMuteなどのCPUテストのみが有効になります。

### 4. デーモン起動

```bash
./scripts/daemon.sh start

# サウンド設定で出力デバイスを「GPU Upsampler」に選択
```

---

## 主要仕様

### フィルタ仕様

| 項目 | 仕様 |
|-----|-----|
| FIRフィルタ | 2,000,000タップ |
| 位相タイプ | ハイブリッド（デフォルト）/ 最小位相（オプション） |
| ストップバンド減衰 | ハイブリッド: 150dB級（100Hzクロスオーバ） |
| ウィンドウ関数 | Kaiser (β=25, Float32 GPU実装に最適化) |
| DCゲイン | アップサンプリング倍率 × 0.99（全レートで音量統一） |
| メタデータ | 各フィルタに`.json`ファイル（DCゲイン、入力/出力レート、倍率を含む） |

### アップサンプリング仕様（マルチレート対応）

システムは入力レートとDAC性能に基づいて自動的に最適なアップサンプリング倍率を選択します。

| 入力レート | 出力レート | 倍率 | 使用フィルタ |
|-----------|----------|------|------------|
| 44.1kHz | 705.6kHz | 16x | filter_44k_16x_2m_min_phase.bin |
| 88.2kHz | 705.6kHz | 8x | filter_44k_8x_2m_min_phase.bin |
| 176.4kHz | 705.6kHz | 4x | filter_44k_4x_2m_min_phase.bin |
| 352.8kHz | 705.6kHz | 2x | filter_44k_2x_2m_min_phase.bin |
| 48kHz | 768kHz | 16x | filter_48k_16x_2m_min_phase.bin |
| 96kHz | 768kHz | 8x | filter_48k_8x_2m_min_phase.bin |
| 192kHz | 768kHz | 4x | filter_48k_4x_2m_min_phase.bin |
| 384kHz | 768kHz | 2x | filter_48k_2x_2m_min_phase.bin |

**フィルタ配布**: 各レート・倍率に対応するフィルタ係数ファイル（`.bin`）とメタデータ（`.json`）が`data/coefficients/`に配置されています。メタデータにはDCゲイン、入力/出力レート、アップサンプリング倍率が記載されています。

**ゲイン設定**: `config.json`の`gain`パラメータはフィルタの`max_coefficient_amplitude`に応じて設定してください。ハイブリッド係数では `calculate_safe_gain` の推奨値（例: 44k/16xで約`0.285`）を採用することで、全レートでクリッピングを防げます。

**ヘッドルーム設定**: `headroomTarget`（デフォルト`0.92`）は出力リミッターのターゲットピークを線形値で指定します。サンプリングレートや使用フィルタに依存せず一定音量を維持しつつ、ピークを0dBFS未満に抑えてクリッピングを防ぎます。0.5–0.999の範囲で設定可能です。

### EQ仕様

| 項目 | 仕様 |
|-----|-----|
| ターゲットカーブ | KB5000_7 |
| EQソース | OPRA Project (CC BY-SA 4.0) |
| 追加フィルタ | PK Fc 5366 Hz Gain 2.8 dB Q 1.5 |

### 低遅延パーティションモード

`config.json` の `partitionedConvolution` セクションで GPU パーティション処理を制御できます。
サポートされるキー:

| キー | 説明 | デフォルト |
|------|------|------------|
| `enabled` | 低遅延モードの有効/無効 | `false` |
| `fastPartitionTaps` | 即時出力用パーティションのタップ数 | `32768` |
| `minPartitionTaps` | tail パーティションの最小タップ数 | `32768` |
| `maxPartitions` | 生成する最大パーティション数 | `4` |
| `tailFftMultiple` | tail パーティション FFT サイズをタップ数の何倍にするか（2〜16） | `2` |

```
"partitionedConvolution": {
  "enabled": true,
  "fastPartitionTaps": 32768,
  "minPartitionTaps": 32768,
  "maxPartitions": 4,
  "tailFftMultiple": 4
}
```

クロスフィードは現在の低遅延パスでは未サポートです。クロスフィード設定が有効な場合はデーモン起動時に警告を出して自動的に従来モードへフォールバックします。EQ は低遅延モードでも利用できます。

#### 検証ワークフロー

1. **パーティション構成の確認**
   ```bash
   uv run python scripts/analysis/inspect_impulse.py \
     --coeff data/coefficients/filter_44k_16x_2m_min_phase.bin \
     --metadata data/coefficients/filter_44k_16x_2m_min_phase.json \
     --config config.json --enable-partition \
     --summary-json plots/analysis/partition_summary.json
   ```
   - fast/tailごとのタップ数・FFTサイズ・エネルギー配分をテーブル化
   - 推定遅延（fastウィンドウ / tail合流ウィンドウ）を自動計算
2. **周波数応答の比較**
   ```bash
   uv run python scripts/analysis/verify_frequency_response.py \
     test_data/low_latency/test_sweep_44100hz.wav \
     test_output/lowlat_sweep.wav \
     --metadata data/coefficients/filter_44k_16x_2m_min_phase.json \
     --config config.json \
     --analysis-window-seconds 1.5 \
     --compare-fast-tail
   ```
   - 自動的に tail 合流までの秒数をスキップし、fast 区間とのスペクトル差分を表示
   - stopband 減衰とピーク一致を既存モードと比較

詳しいループバック手順やXRUN/GPU監視のポイントは `docs/investigations/low_latency_partition_validation.md` を参照してください。

---

## Web API

主要エンドポイント一覧（詳細は [docs/api/README.md](docs/api/README.md) 参照）

| エンドポイント | メソッド | 説明 |
|--------------|--------|------|
| `/api/status` | GET | システムステータス取得 |
| `/api/daemon/start` | POST | デーモン起動 |
| `/api/daemon/stop` | POST | デーモン停止 |
| `/api/eq/profiles` | GET | EQプロファイル一覧 |
| `/api/eq/apply` | POST | EQ適用 |
| `/api/opra/search` | GET | OPRAヘッドホン検索 |
| `/api/dac/capability/{id}` | GET | DAC性能取得 |

> Note: OPRAから適用したプロファイルの履歴はSaved EQ Profilesには表示せず、`/api/eq/profiles` のレスポンスからも除外されます。

### ZeroMQコマンド

| コマンド | 説明 |
|---------|------|
| `LOAD_IR` | IR係数ロード |
| `SET_GAIN` | ゲイン設定 |
| `SOFT_RESET` | バッファクリア |
| `GET_STATUS` | ステータス取得 |
| `SWITCH_RATE` | レートファミリー切り替え |
| `APPLY_EQ` | EQ適用 |
| `RESTORE_EQ` | EQ解除 |

---

## プロジェクト構成

```
gpu_os/
├── README.md              # このファイル
├── CLAUDE.md              # AI開発ガイド（詳細仕様）
├── CMakeLists.txt         # C++/CUDAビルド設定
├── config.json            # ランタイム設定
├── pyproject.toml         # Python依存関係
│
├── src/                   # C++/CUDAソースコード
│   ├── gpu/               # GPU処理（convolution, crossfeed）
│   ├── zeromq_interface.cpp
│   ├── auto_negotiation.cpp
│   ├── soft_mute.cpp
│   └── ...
│
├── include/               # C++ヘッダー
├── web/                   # Python/FastAPI Web API
│   ├── main.py
│   ├── models.py
│   └── routers/
│
├── scripts/               # ユーティリティスクリプト
│   ├── generate_minimum_phase.py # 最小位相フィルタ生成
│   ├── generate_linear_phase.py  # 線形位相フィルタ生成
│   └── daemon.sh               # デーモン管理
│
├── data/
│   ├── coefficients/      # FIRフィルタ係数
│   └── hrtf/              # HRTF係数（Crossfeed用）
│
├── docs/                  # ドキュメント
│   ├── api/               # REST API仕様
│   ├── architecture/      # システム設計
│   └── roadmap.md         # 開発ロードマップ
│
└── tests/                 # テストコード
    ├── cpp/               # C++ユニットテスト
    └── gpu/               # GPUテスト
```

---

## 開発ロードマップ

| Phase | 内容 | 進捗 |
|-------|-----|------|
| **1** | Core Engine & Middleware | **60%** |
| | - GPU FFT Convolution | ✅ 完了 (~28x realtime) |
| | - ZeroMQ通信層 | ✅ 完了 |
| | - Auto-Negotiation | ✅ 完了 |
| | - Soft Mute | ✅ 完了 |
| | - Crossfeed (HRTF) | ✅ 完了 |
| | - 48k系係数生成 | ⬜ 未実装 |
| | - 動的レート検知 | ⬜ 未実装 |
| **2** | Control Plane & Web UI | **20%** |
| | - FastAPI基本構造 | ✅ 完了 |
| | - OPRA統合 | ⬜ 未実装 |
| | - Webフロントエンド | ⬜ 未実装 |
| **3** | Hardware Integration | **0%** |
| | - Jetson移植 | ⬜ 未実装 |
| | - USB Gadget Mode | ⬜ 未実装 |

詳細は [docs/roadmap.md](docs/roadmap.md) を参照。

---

## ドキュメント

- [セットアップガイド](docs/setup/pc_development.md) - PC開発環境構築
- [アーキテクチャ概要](docs/architecture/overview.md) - システム設計詳細
- [REST API仕様](docs/api/README.md) - Web APIドキュメント
- [ロードマップ](docs/roadmap.md) - 開発計画と進捗

---

## ライセンス

TBD

### 使用ライブラリ

| ライブラリ | 用途 | ライセンス |
|-----------|------|----------|
| CUDA/cuFFT | GPU計算 | NVIDIA EULA |
| alsa-lib | オーディオ出力 | LGPL-2.1 |
| ZeroMQ | IPC通信 | LGPL-3.0 |
| FastAPI | Web API | MIT |
| scipy | フィルタ生成 | BSD-3-Clause |

### EQデータソース

| ソース | ライセンス | 商用利用 |
|--------|----------|----------|
| **OPRA** | CC BY-SA 4.0 | ✅ OK（帰属表示必須） |
| oratory1990 | 独自 | ❌ 禁止 |

### HRTFデータソース（Crossfeed機能）

| ソース | ライセンス | 商用利用 |
|--------|----------|----------|
| **HUTUBS** | CC BY 4.0 | ✅ OK（帰属表示必須） |

**HUTUBS (Head-related Transfer Function Database of TU Berlin)**
- 提供元: Technische Universität Berlin
- URL: https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960
- 著者: Brinkmann, Dinakaran, Pelzer, Wohlgemuth, Seipel, Voss, Grosche, Weinzierl
- 帰属表示例: "HRTF data from HUTUBS database, TU Berlin (CC BY 4.0)"
