# 既知の制限と代替案

## 概要

Jetson Orin NanoでのUSB Audio Gadget (UAC2) 実装には、いくつかの既知の制限があります。本ドキュメントでは、これらの制限と回避策・代替案を説明します。

---

## 既知の制限

### 1. tegra-xudc ドライバの Isochronous 転送制限

#### 問題

Jetson OrinのUSBデバイスコントローラ (tegra-xudc) において、isochronous転送（リアルタイムオーディオに必須）が正常に動作しない可能性があります。

**エラー例**:
```
udc 3550000.usb: failed to start g1: -19
```

#### 原因

- L4Tカーネルのtegra-xudcドライバがisochronous endpointを完全にサポートしていない
- NVIDIAの公式サポート対象外の機能

#### 現状

- NVIDIAフォーラムで問題報告あり
- 公式な修正のタイムラインは不明
- 実機検証が必須

**参考**: [NVIDIA Developer Forums - USB Audio gadget on Jetson Orin](https://forums.developer.nvidia.com/t/usb-audio-gadget-on-jetson-orin-devkit-64g/323509)

---

### 2. USB 3.0 デバイスモードの不安定性

#### 問題

USB 3.0ポートでのデバイスモード動作が不安定な報告があります。

#### 症状

- デバイスが認識されない
- 接続が断続的に切れる
- `invalid argument` エラー

#### 回避策

**USB 2.0での動作を基本とする**

```bash
# デバイスツリーでUSB 2.0強制（将来対応）
# または、USB 2.0ケーブル使用
```

---

### 3. 複合ガジェットの関数順序

#### 問題

Composite Gadgetで複数の機能を組み合わせる際、関数の追加順序によってはホストで認識されないことがあります。

#### 回避策

```bash
# 推奨順序: Audio → Ethernet
ln -s functions/uac2.usb0 configs/c.1/
ln -s functions/ecm.usb0 configs/c.1/

# NG: Ethernet → Audio だと認識しない場合あり
```

---

## 代替案

UAC2が動作しない場合の代替入力方式:

### 代替案1: USB DDC経由入力

```
PC  ──USB──>  USB-DDC  ──S/PDIF──>  Magic Box  ──USB──>  DAC
                         (同軸/光)
```

#### メリット

- 確実に動作する（一般的なオーディオ入力）
- 高品質なDDCなら低ジッター

#### デメリット

- 追加コスト（USB-DDC: ¥5,000〜¥50,000）
- ケーブルが増える
- S/PDIF入力対応が必要（ADCボード追加）

#### 必要な変更

```cpp
// S/PDIF入力用のALSAキャプチャ設定
snd_pcm_open(&capture_handle, "hw:SPDIF", SND_PCM_STREAM_CAPTURE, 0);
```

---

### 代替案2: I2S直接入力

```
                    ┌─────────────────────────┐
PC  ──USB──>  DAC  ─┤ I2S (BCLK, LRCK, DATA) ├──>  Magic Box (GPIO)
                    └─────────────────────────┘
```

#### メリット

- 低レイテンシ
- 追加コスト低い（ケーブルのみ）
- 高品質（デジタル直結）

#### デメリット

- I2S出力可能なDACが必要
- GPIO配線作業が必要
- PCからの直接接続不可

#### Jetson Orin Nano I2S ピンアウト

Jetson Orin Nano DevKit（キャリアボード）の **J12（40pin）** は、I2S0系の信号を利用できる。詳細は `docs/jetson/i2s/i2s_migration_spec_820.md` を参照。

| 信号 | J12 物理ピン | 備考 |
|---|---:|---|
| I2S0_SCLK（BCLK） | 12 | |
| I2S0_FS（LRCLK） | 35 | |
| I2S0_DIN | 38 | Pi→Jetson（MVP）で使用 |
| I2S0_DOUT | 40 | MVPでは未使用 |
| GND | 39 | 推奨（DATA隣） |

---


```
                  (WiFi/LAN)
```

#### メリット

- ワイヤレス対応可能
- PC側の設定が容易
- 柔軟なルーティング

#### デメリット

- レイテンシ大（数十〜数百ms）
- ネットワーク依存（パケットロス）
- リアルタイム性が低い

#### 設定例（廃止）

> ⚠️ RTP/PipeWire経路は #690 で廃止し、TCP PCM入力に一本化しました。
> 本節の設定は現在サポートされていません（歴史的記録のみ）。

---

### 代替案4: USB Audio Host Mode + 専用入力デバイス

```
PC  ──USB──>  USB Audio Interface  ──USB──>  Magic Box (Host)  ──>  DAC
              (UAC2対応)
```

#### メリット

- 確実に動作
- 高品質な入力インターフェース使用可能

#### デメリット

- USBポート2つ使用（入力用 + DAC出力用）
- 追加機器コスト
- 接続が複雑化

---

## 推奨アプローチ

### Phase 1: UAC2検証

1. 実機でカーネルモジュールビルド
2. UAC2 Gadget単体でのテスト
3. Composite Gadget (UAC2 + ECM) テスト
4. 安定性・音質評価

### Phase 2: 代替案準備

UAC2が動作しない/不安定な場合:

1. **短期**: USB DDC経由入力を採用
2. **中期**: I2S入力対応を追加開発
3. **長期**: NVIDIAのドライバ修正を待つ

---

## 判断フローチャート

```
UAC2 Gadget テスト
        │
        ├── 成功 ──> UAC2採用
        │
        └── 失敗 ──> Isochronous問題？
                            │
                            ├── Yes ──> 代替案1: USB-DDC
                            │           または
                            │           代替案2: I2S
                            │
                            └── No ──> カーネルモジュール再確認
                                        ↓
                                    設定問題を修正
```

---

## 実機検証チェックリスト

- [ ] カーネルモジュール (u_audio, usb_f_uac2) ロード成功
- [ ] ConfigFS Gadget作成成功
- [ ] UDCバインド成功
- [ ] ホストPCでデバイス認識
- [ ] ALSAデバイス表示 (`arecord -l`)
- [ ] 音声キャプチャテスト
- [ ] 長時間安定性テスト（10分）
- [ ] ECM同時使用テスト

---

## 関連ドキュメント

- [composite-gadget.md](./composite-gadget.md) - Composite Gadget設計
- [kernel-modules.md](./kernel-modules.md) - カーネルモジュールビルド
- [../troubleshooting.md](../troubleshooting.md) - トラブルシューティング
