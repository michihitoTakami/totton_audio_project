# システムアーキテクチャ概要

## 設計思想

Totton Audioは**Control Plane / Data Plane分離アーキテクチャ**を採用しています。

- **Control Plane (Python/FastAPI)**: ユーザー操作、設定管理、係数生成
- **Data Plane (C++/CUDA)**: リアルタイムオーディオ処理

この分離により、UIの応答性とオーディオ処理の低レイテンシを両立します。

---

## システム構成図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Totton Audio                                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        USB Composite Gadget                          │    │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                 │    │
│  │  │   UAC2 (Audio)      │    │   ECM (Ethernet)    │                 │    │
│  │  │   44.1k/48k, 2ch    │    │   192.168.55.1/24   │                 │    │
│  │  └──────────┬──────────┘    └──────────┬──────────┘                 │    │
│  └─────────────┼──────────────────────────┼────────────────────────────┘    │
│                │                          │                                  │
│                ▼                          ▼                                  │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                 │
│  │    ALSA Capture         │    │    systemd-networkd     │                 │
│  │    (UAC2 Gadget)        │    │    (DHCP Server)        │                 │
│  └──────────┬──────────────┘    └──────────┬──────────────┘                 │
│             │                              │                                 │
│             ▼                              ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Data Plane (C++)                              │   │
│  │  ┌────────────┐   ┌────────────────┐   ┌────────────────┐            │   │
│  │  │ Ring Buffer│──>│ GPU Convolution│──>│ ALSA Playback  │──> DAC     │   │
│  │  │ (Input)    │   │ 640k-tap FIR   │   │ (Output)       │            │   │
│  │  └────────────┘   └────────────────┘   └────────────────┘            │   │
│  │        ▲                  ▲                                           │   │
│  │        │                  │                                           │   │
│  │        │     ┌────────────┴────────────┐                             │   │
│  │        │     │    Filter Coefficients   │                             │   │
│  │        │     │    (GPU Memory)          │                             │   │
│  │        │     └──────────────────────────┘                             │   │
│  │        │                  ▲                                           │   │
│  │  ┌─────┴──────────────────┴─────────────────────────────────────┐    │   │
│  │  │                    ZeroMQ IPC                                 │    │   │
│  │  │                 ipc:///tmp/gpu_os.sock                        │    │   │
│  │  └──────────────────────────┬───────────────────────────────────┘    │   │
│  └─────────────────────────────┼────────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Control Plane (Python)                          │   │
│  │  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐        │   │
│  │  │ FastAPI        │   │ IR Generator   │   │ Config Manager │        │   │
│  │  │ Web UI (80)    │   │ (scipy)        │   │ (JSON)         │        │   │
│  │  └────────────────┘   └────────────────┘   └────────────────┘        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                ▲                                             │
│                                │ HTTP (port 80)                             │
│                                │                                             │
└────────────────────────────────┼─────────────────────────────────────────────┘
                                 │
                          ┌──────┴──────┐
                          │   Browser   │
                          │   (User)    │
                          └─────────────┘
```

---

## コンポーネント詳細

### 1. USB Composite Gadget

単一のUSB Type-Cポートで2つの機能を提供：

| Function | 用途 | 仕様 |
|----------|------|------|
| **UAC2** | オーディオ入力 | 44.1kHz/48kHz, 2ch, 32-bit |
| **ECM/NCM** | 管理アクセス | Ethernet over USB |

詳細: [usb-gadget/composite-gadget.md](../usb-gadget/composite-gadget.md)

### 2. Data Plane (gpu_upsampler_alsa)

リアルタイムオーディオ処理を担当するC++デーモン。

| モジュール | 役割 |
|-----------|------|
| **Ring Buffer** | 入力バッファリング（Lock-free） |
| **Rate Detection** | 44.1kHz/48kHz系の自動判定 |
| **GPU Convolution** | cuFFT Overlap-Save法による畳み込み |
| **Soft Mute** | レート切り替え時のクロスフェード |
| **ALSA Output** | Bit-perfect出力 |

詳細: [data-flow.md](./data-flow.md)

### 3. Control Plane (FastAPI Web UI)

ユーザーインターフェースと設定管理。

| エンドポイント | 機能 |
|---------------|------|
| `GET /` | 管理画面 |
| `GET /status` | システムステータス |
| `POST /eq/activate/{name}` | EQプロファイル適用 |
| `GET /opra/vendors` | ヘッドホンメーカー一覧 |
| `POST /daemon/restart` | デーモン再起動 |

詳細: [../../../web/](../../../web/) および [../../../docs/api/](../../../docs/api/)

### 4. ZeroMQ IPC

Control PlaneとData Plane間の通信。

| パターン | 用途 |
|---------|------|
| REQ/REP | コマンド送受信 |
| PUB/SUB | ステータス通知（将来） |

**エンドポイント**: `ipc:///tmp/gpu_os.sock`

---

## ブート シーケンス

```
電源ON
   │
   ├─> systemd multi-user.target
   │       │
   │       ├─> magicbox-gadget.service     [USB Gadget初期化]
   │       │       │
   │       │       ├─> ConfigFS設定
   │       │       └─> UDC有効化
   │       │
   │       ├─> systemd-networkd            [ネットワーク設定]
   │       │       │
   │       │       └─> usb0: 192.168.55.1/24 + DHCP Server
   │       │
   │       ├─> avahi-daemon                [mDNS]
   │       │       │
   │       │       └─> magicbox.local 登録
   │       │
   │       ├─> gpu-upsampler.service       [音声エンジン]
   │       │       │
   │       │       ├─> フィルタ係数ロード (GPU)
   │       │       ├─> ALSA出力デバイス初期化
   │       │       ├─> ZeroMQ IPC開始
   │       │       └─> 入力待機
   │       │
   │       └─> magicbox-web.service        [Web UI]
   │               │
   │               └─> uvicorn :80 起動
   │
   └─> Ready (約30秒)
```

詳細: [../systemd/service-design.md](../systemd/service-design.md)

---

## リソース使用量（推定）

### メモリ

| コンポーネント | 使用量 |
|---------------|--------|
| Linux Kernel + Systemd | ~200 MB |
| GPU Upsampler Daemon | ~500 MB |
| Filter Coefficients (GPU) | ~64 MB |
| Web UI (Python) | ~100 MB |
| **合計** | **~900 MB** (8GB中) |

### GPU

| リソース | 使用量 |
|---------|--------|
| CUDA Cores | ~50% (推定) |
| VRAM | ~200 MB |
| Memory Bandwidth | 制約要因（68 GB/s） |

### ストレージ

| 項目 | サイズ |
|------|--------|
| OS + アプリ | ~10 GB |
| フィルタ係数 | ~512 MB (全構成) |
| ログ (最大) | ~100 MB |
| **合計** | **~11 GB** (1TB中) |

---

## セキュリティ境界

```
┌─────────────────────────────────────────┐
│             Trusted Zone                 │
│  (Totton Audio内部)                        │
│                                          │
│  ┌──────────┐  ┌──────────┐             │
│  │ Data     │  │ Control  │             │
│  │ Plane    │  │ Plane    │             │
│  └──────────┘  └──────────┘             │
│                     │                    │
└─────────────────────┼────────────────────┘
                      │
        USB Ethernet (192.168.55.0/24)
                      │
┌─────────────────────┼────────────────────┐
│             Untrusted Zone               │
│  (ユーザーPC)                            │
│                                          │
│  ┌──────────┐                            │
│  │ Browser  │                            │
│  └──────────┘                            │
│                                          │
└──────────────────────────────────────────┘
```

**設計原則**:
- USB Ethernet経由のアクセスのみ許可
- 不要なポートは閉じる（ファイアウォール）
- ローカルアクセス想定のため認証は将来拡張

詳細: [../systemd/security-hardening.md](../systemd/security-hardening.md)

---

## 拡張ポイント

将来の機能拡張を見据えた設計：

| 拡張機能 | 対応方針 |
|---------|---------|
| WiFi AP | hostapd追加、Captive Portal |
| 複数DAC | ALSA出力先の動的選択 |
| リバーブ | GPUプラグインアーキテクチャ |
| リモート診断 | ZeroMQ外部公開 or REST API |
| クラウド連携 | OTA更新サーバ、テレメトリ |

---

## 関連ドキュメント

- [data-flow.md](./data-flow.md) - オーディオデータフロー詳細
- [hardware-specification.md](./hardware-specification.md) - ハードウェア仕様
- [software-stack.md](./software-stack.md) - ソフトウェアスタック
