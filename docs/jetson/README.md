# Magic Box - Jetson Orin Nano 組み込みガイド

## 製品ビジョン

**全てのヘッドホンユーザーに最高の音を届ける箱**

```
1. 箱をつなぐ
2. 管理画面でポチポチ
3. 最高の音
```

ユーザーに余計なことを考えさせない。ヘッドホンを選んで、ボタンを押すだけ。

---

## システム概要

Magic Boxは、2,000,000タップの最小位相FIRフィルタによる究極のアップサンプリングと、ヘッドホン周波数特性の自動補正を提供するオーディオプロセッサです。

### 主要機能

| 機能 | 説明 |
|------|------|
| **Ultimate Upsampling** | 2Mタップ最小位相FIRで最大16倍アップサンプリング（44.1kHz→705.6kHz） |
| **Headphone Correction** | OPRAデータベースに基づくヘッドホン周波数特性補正 |
| **Seamless Operation** | 入力レート自動検知、DAC性能に応じた最適化 |
| **Simple UI** | Web UIでヘッドホンを選んでポチポチするだけ |

---

## ハードウェア構成

### 本体仕様

| 項目 | 仕様 |
|------|------|
| **SoC** | NVIDIA Jetson Orin Nano Super (8GB, 1024 CUDA Cores) |
| **CUDA Arch** | SM 8.7 (Ampere) |
| **Storage** | 1TB NVMe SSD (KIOXIA EXCERIA G2) |
| **入力** | USB Type-C (UAC2 Device Mode) |
| **出力** | USB Type-A → 外部USB DAC |
| **管理** | USB Ethernet (同一ケーブルでPCから管理) |

### 接続図

```
┌─────────────────────────────────────────────────────────────┐
│                       Magic Box                              │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ USB Type-C  │───>│ Jetson Orin │───>│ USB Type-A   │    │
│  │ (入力+管理) │    │    Nano     │    │ (DAC出力)    │    │
│  └─────────────┘    └──────────────┘    └──────────────┘    │
│        │                   │                    │           │
│        │              GPU処理               外部DAC         │
│        │           2Mタップ畳み込み                         │
│        │                                                    │
│   ┌────┴────┐                                               │
│   │ UAC2    │  PCからのオーディオ入力                       │
│   │ ECM     │  Web UI管理アクセス (192.168.55.1)            │
│   └─────────┘                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## クイックスタート

### 評価者向け（ソース不要 / Docker）

- **Jetson評価版を最短で起動**したい場合は、まずこちらを参照してください:
  - [Jetson 評価者向け導入ガイド（ソース不要 / Docker）](./evaluator_guide_docker.md)

### 1. 接続

```
PC  ──USB Type-C──>  Magic Box  ──USB Type-A──>  外部DAC  ──>  ヘッドホン
```

### 2. 管理画面にアクセス

USBケーブル接続後、ブラウザで以下にアクセス：

- `http://192.168.55.1/`
- または `http://magicbox.local/`（mDNS対応環境）

### 3. ヘッドホンを選択

1. 管理画面で「ヘッドホン選択」
2. メーカー・モデルを選択
3. 「適用」をクリック
4. 音楽を再生

---

## ドキュメント構成

| ディレクトリ | 内容 |
|-------------|------|
| [architecture/](./architecture/) | システムアーキテクチャ設計 |
| [usb-gadget/](./usb-gadget/) | USB Composite Gadget設計 |
| [systemd/](./systemd/) | Systemdサービス設計 |
| [network/](./network/) | ネットワーク設計 |
| [deployment/](./deployment/) | デプロイメント・OTA設計 |
| [reliability/](./reliability/) | 信頼性・エラー回復設計 |
| [quality/](./quality/) | 品質基準・テスト |
| [dev-autostart.md](./dev-autostart.md) | Jetson開発機向け自動起動セットアップ |
| [troubleshooting.md](./troubleshooting.md) | トラブルシューティング |

---

## 技術仕様

### オーディオ処理

| パラメータ | 値 |
|-----------|-----|
| フィルタタップ数 | 2,000,000 (2M) |
| 位相タイプ | ハイブリッド (default) / 最小位相 (legacy) |
| ストップバンド減衰 | 50–67dB (ハイブリッド) |
| 入力レート | 44.1kHz / 48kHz |
| 出力レート | 最大 705.6kHz / 768kHz |
| ビット深度 | 32-bit float (内部処理) |

### パフォーマンス目標

| 指標 | 目標値 |
|------|--------|
| 処理速度 | > 1x realtime |
| レイテンシ | < 100ms (最小位相時) |
| XRUN | 0 (10分連続再生) |
| 起動時間 | < 30秒 (電源ON→音声出力可能) |

---

## ライセンス・帰属

### EQデータソース

- **OPRA Project** (CC BY-SA 4.0) - ヘッドホンEQデータ
- 商用利用可能、帰属表示必須

### ソフトウェア

- GPU Upsampler Core - Proprietary
- FastAPI, CUDA, ALSA - 各ライセンスに従う

---

## サポート

- **GitHub Issues**: https://github.com/michihitoTakami/gpu_os/issues
- **技術ドキュメント**: 本ディレクトリ配下

---

## 改訂履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 0.1.0 | 2025-11-26 | 初版作成 |
