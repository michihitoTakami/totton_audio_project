# GPU Upsampler バイパス機能 実装試行記録

## 目的
アップサンプリングのON/OFF切り替え機能を実装し、OFFの場合はEasy EffectsのEQを維持しつつGPUアップサンプリングをバイパスする。

## 試行した手法

### 手法1: デーモン内でのバイパス（サンプル複製）
- **実装内容**: `enabled=false`の時、GPUコンボリューションをスキップしてサンプルを16倍複製
- **結果**: 失敗
- **問題**: 出力は16xレートのままなので、PipeWireでEasy Effectsの前段に戻すと、Easy EffectsのEQもバイパスされる（Easy Effectsはgpu_upsampler_sinkの後段にある）

### 手法2: pw-linkでルーティング切り替え
- **実装内容**:
  - ON: `ee_soe_output_level` → `gpu_upsampler_sink`
  - OFF: `ee_soe_output_level` → 直接ALSA sink
- **結果**: 部分的に成功、実用は困難

#### 試行2a: 別のALSA sinkに接続
- **問題**: オンボードaudio(`alsa_output.pci-0000_00_1f.3.analog-stereo`)に出力されるが、SMSLのUSB DACに接続されていないため音が出ない

#### 試行2b: USB DAC(SMSL)に直接接続
- **問題**: GPU Upsamplerデーモンが直接ALSAでUSB DACを占有しているため、PipeWire/WirePlumberがUSB DACをsinkとして認識しない

#### 試行2c: デーモン停止 + WirePlumber再起動
- **実装内容**:
  1. デーモンを`kill -9`で停止
  2. `systemctl --user restart wireplumber`でWirePlumber再起動
  3. USB DAC(`alsa_output.usb-SMSL_SMSL_USB_AUDIO-00.analog-stereo`)が出現
  4. Easy Effectsの出力をUSB DACに接続
- **結果**: バイパスOFF→成功（音が出る）
- **問題**: ON復帰が困難
  - WirePlumber再起動で`gpu_upsampler_sink`も消える
  - デーモン再起動してもPipeWireとの再接続が不安定
  - Easy Effectsの出力接続も再設定が必要

## 根本的な問題

### アーキテクチャ上の制約
1. **ALSA排他利用**: GPU Upsamplerデーモンは高レート出力のためALSAデバイスを直接排他オープンしている
2. **PipeWireとの共存不可**: 同じUSB DACをPipeWire経由でアクセスすることができない
3. **WirePlumber依存**: デバイスの認識/解放にWirePlumber再起動が必要で、全体のオーディオグラフが不安定になる

### 解決に必要な変更
- GPU UpsamplerをPipeWire nativeなsinkとして実装し直す（大規模な改修）
- または、バイパス専用の別出力デバイスを用意する（オンボードaudioなど）

## 結論
バイパス機能は現在のアーキテクチャでは実用的に実装できない。
アップサンプリングのON/OFFが必要な場合は、別途オンボードaudioなどの代替出力先を用意するか、デーモンの手動停止/起動で対応する。

## 削除した実装
- `web/main.py`:
  - `check_upsampling_enabled()` 関数
  - `get_bypass_sink()` 関数
  - `switch_audio_routing()` 関数のバイパスロジック
  - `/upsampling` エンドポイント
  - `UpsamplingToggle` モデル
  - HTML UIのトグルスイッチ
  - JavaScript のトグルハンドラー
