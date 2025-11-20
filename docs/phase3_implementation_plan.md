# Phase 3 実装計画: PipeWire Filter-Chain + LV2プラグイン統合

## 1. プロジェクト概要

### 目標
100万タップGPU FIRフィルタをPipeWireシステムに統合し、Easy Effectsのイコライザと連携させながら、システム全体の高音質オーディオ処理を実現する。

### アーキテクチャ
```
[音楽プレイヤー: 44.1kHz]
  ↓
[Easy Effects Sink]
  ├─ イコライザ適用 (44.1kHz)
  ├─ リミッター
  └─ その他エフェクト
  ↓
[GPU Upsampler Filter-Chain (LV2プラグイン)]
  ├─ CUDA FFT畳み込み
  ├─ 100万タップ minimum phase FIR
  └─ 44.1kHz → 705.6kHz アップサンプリング
  ↓
[SMSL USB AUDIO DAC: 705.6kHz出力]

[システム音・ブラウザ等: 48kHz]
  ↓
[内蔵スピーカー: 48kHz そのまま]
```

### ユーザー要件
1. ✅ Easy Effectsのイコライザは常時使用
2. ✅ 処理順序: イコライザ(44.1kHz) → GPU Upsampler(→705.6kHz)
3. ✅ 出力デバイスで分ける
   - 高音質DAC (SMSL): GPU処理 + 705.6kHz
   - 内蔵スピーカー: 通常処理 + 48kHz

### DAC情報
- **製品名**: SMSL USB AUDIO
- **ノード名**: `alsa_output.usb-SMSL_SMSL_USB_AUDIO-00.analog-stereo`
- **現在のレート**: 44.1kHz (将来的に705.6kHzに変更)
- **フォーマット**: s32le (32bit signed)
- **チャンネル**: 2ch stereo

---

## 2. 実装コンポーネント

### 2.1 LV2プラグイン (`gpu_upsampler.lv2`)

#### 目的
Phase 2で実装したCUDA FFT畳み込みエンジンをLV2プラグインとしてラップし、PipeWireから利用可能にする。

#### 主要機能
- **入力**: 44.1kHz または 48kHz stereo (float32)
- **出力**: 16倍アップサンプリング (705.6kHz / 768kHz) stereo (float32)
- **処理**: CUDA cuFFT + 100万タップFIRフィルタ
- **非同期処理**: GPU処理専用スレッド + Lock-freeリングバッファ
- **レイテンシ報告**: LV2 Worker APIを使用

#### ファイル構成
```
src/lv2_plugin/
├── gpu_upsampler_lv2.cpp      # LV2記述子実装
├── manifest.ttl                # プラグイン登録情報
├── gpu_upsampler.ttl           # プラグインメタデータ
└── CMakeLists.txt              # ビルド設定

インストール先:
/usr/local/lib/lv2/gpu_upsampler.lv2/
├── gpu_upsampler.so
├── manifest.ttl
└── gpu_upsampler.ttl
```

#### LV2 API実装
```cpp
// 必須関数
LV2_Handle instantiate()      // GPU初期化、バッファ確保
void activate()                // サンプルレート取得、処理準備
void run()                     // オーディオ処理（リアルタイムスレッド）
void deactivate()              // 処理停止
void cleanup()                 // リソース解放
```

#### 非同期処理設計
```
[LV2 run() - リアルタイムスレッド]
  ↓ Lock-free push
[入力リングバッファ (3ブロック分, ~48kサンプル)]
  ↓
[GPU処理スレッド]
  1. cudaMemcpy H2D (PCIe転送)
  2. Zero-padding kernel
  3. Overlap-Save FFT畳み込み
  4. cudaMemcpy D2H
  ↓
[出力リングバッファ (3ブロック分, ~768kサンプル)]
  ↓ Lock-free pop
[LV2 run() - 出力]
```

#### レイテンシ計算
```
入力バッファリング:    8192サンプル @ 44.1kHz  ≈ 186ms
GPU処理時間:          ~5ms (Phase 2実測値の1/4ブロックサイズ)
PCIe転送:             ~2ms
出力バッファリング:    131072サンプル @ 705.6kHz ≈ 186ms
────────────────────────────────────────────────
合計レイテンシ:       ~380ms (許容範囲内: 音楽リスニング用途)
```

### 2.2 Filter-Chain設定

#### 目的
GPU Upsampler LV2プラグインをPipeWire Sinkとして公開し、システムレベルで利用可能にする。

#### 設定ファイル
**パス**: `~/.config/pipewire/pipewire.conf.d/90-gpu-upsampler-sink.conf`

**内容**:
```lua
context.modules = [
    {
        name = libpipewire-module-filter-chain
        args = {
            # Sink基本情報
            node.description = "GPU Upsampler (1M tap FIR, 705.6kHz)"
            node.name        = "gpu_upsampler_sink"
            media.name       = "GPU Upsampler"
            media.class      = "Audio/Sink"

            # 入力設定 (Easy Effectsから44.1kHz音声を受信)
            capture.props = {
                audio.rate      = 44100
                audio.position  = [ FL FR ]
                node.passive    = false     # アクティブSinkとして動作
            }

            # 出力設定 (705.6kHzでSMSL DACへ)
            playback.props = {
                audio.rate      = 705600
                audio.position  = [ FL FR ]
                node.passive    = true      # ターゲットに従属
                target.object   = "alsa_output.usb-SMSL_SMSL_USB_AUDIO-00.analog-stereo"
                node.name       = "gpu_upsampler_sink.output"
            }

            # LV2プラグイン読み込み
            filter.graph = {
                nodes = [
                    {
                        type   = lv2
                        name   = gpu_upsampler
                        plugin = "http://github.com/michihito/gpu_upsampler"
                        label  = gpu_upsampler
                        control = {
                            "Filter Path" = "/usr/local/share/gpu_upsampler/filter_1m_min_phase.bin"
                            "Block Size"  = 8192
                        }
                    }
                ]
            }
        }
    }
]
```

#### 動作フロー
1. PipeWire起動時に `90-gpu-upsampler-sink.conf` を読み込み
2. Filter-Chainモジュールが `gpu_upsampler_sink` ノードを作成
3. LV2プラグイン `gpu_upsampler.so` をロード
4. 入力ポート: 44.1kHz受信待機
5. 出力ポート: SMSL DACに705.6kHzで接続

### 2.3 WirePlumber自動ルーティング

#### 目的
アプリケーションを適切な出力先に自動接続する。

#### 設定ファイル
**パス**: `~/.config/wireplumber/main.lua.d/51-gpu-upsampler-routing.lua`

**内容**:
```lua
-- Easy Effectsの出力を GPU Upsampler Sinkに自動接続
rule = {
  matches = {
    {
      { "node.name", "equals", "easyeffects_sink" },
    },
  },
  apply_properties = {
    ["node.target"] = "gpu_upsampler_sink",
  },
}
table.insert(alsa_monitor.rules, rule)

-- 音楽プレイヤーを Easy Effectsへ自動接続
music_players = {
  "spotify", "rhythmbox", "clementine", "audacious",
  "vlc", "mpv", "Spotify", "tidal-hifi", "deadbeef"
}

for _, app in ipairs(music_players) do
  rule = {
    matches = {
      {
        { "application.name", "matches", app },
      },
    },
    apply_properties = {
      ["node.target"] = "easyeffects_sink",
    },
  }
  table.insert(alsa_monitor.rules, rule)
end

-- システム音は内蔵スピーカーへ (デフォルト動作のまま)
-- 明示的なルールは不要
```

#### ルーティング結果
```
Spotify → Easy Effects → GPU Upsampler → SMSL DAC (705.6kHz)
VLC → Easy Effects → GPU Upsampler → SMSL DAC (705.6kHz)
Chrome/Firefox → 内蔵スピーカー (48kHz)
通知音 → 内蔵スピーカー (48kHz)
```

### 2.4 Easy Effects設定

#### 目的
PipeWireの自動リサンプリングを回避し、44.1kHz入力を保持する。

#### 設定コマンド
```bash
# Easy Effectsの出力先をGPU Upsampler Sinkに固定
gsettings set com.github.wwmm.easyeffects.streamoutputs use-default-sink false
gsettings set com.github.wwmm.easyeffects.streamoutputs target-object "gpu_upsampler_sink"

# 入力レートを保持（自動変換無効）
gsettings set com.github.wwmm.easyeffects input-device "preserve-rate"
```

#### Easy Effects処理チェーン
```
入力: 44.1kHz → イコライザ → リミッター → その他エフェクト → 出力: 44.1kHz
```

**重要**: Easy Effects内では44.1kHzのまま処理されるため、CPU負荷が最小限。

---

## 3. 実装手順（4週間計画）

### Week 1: LV2プラグイン基本構造 (5日間)

#### Day 1-2: Phase 2エンジンのライブラリ化
**タスク**:
- [ ] `src/convolution_engine.cu` を静的ライブラリとして分離
- [ ] `libgpu_upsampler_core.a` 作成
- [ ] CMakeLists.txt更新（`add_library(gpu_upsampler_core STATIC ...)`）
- [ ] リンクテスト（既存のgpu_upsamplerコマンドが動作すること）

**成果物**: `build/libgpu_upsampler_core.a`

#### Day 3-4: LV2記述子実装（パススルー版）
**タスク**:
- [ ] `src/lv2_plugin/gpu_upsampler_lv2.cpp` 作成
- [ ] LV2_Descriptor構造体実装
- [ ] instantiate/activate/run/deactivate/cleanup関数実装
- [ ] 初期版: 入力をそのまま出力（GPU処理なし）
- [ ] ビルド設定（CMakeLists.txt）
- [ ] インストールパス設定（/usr/local/lib/lv2/）

**検証コマンド**:
```bash
make install
lv2ls | grep gpu_upsampler
lv2info http://github.com/michihito/gpu_upsampler
```

#### Day 5: TTLファイル作成
**タスク**:
- [ ] `manifest.ttl` 作成（プラグイン登録）
- [ ] `gpu_upsampler.ttl` 作成（ポート定義、パラメータ）
- [ ] URI定義: `http://github.com/michihito/gpu_upsampler`
- [ ] ポート定義: Input (stereo), Output (stereo)
- [ ] コントロールポート: Filter Path, Block Size

**検証**: `lv2_jack_host` でプラグインをロード、パススルー動作確認

---

### Week 2: GPU統合（同期処理版） (5日間)

#### Day 6-7: CUDAコンテキスト統合
**タスク**:
- [ ] `instantiate()` でCUDA初期化
- [ ] `GPUUpsampler` クラスインスタンス化
- [ ] フィルタ係数ロード（`filter_1m_min_phase.bin`）
- [ ] CUDA Streamは使わず、同期処理版として実装
- [ ] エラーハンドリング（GPU初期化失敗時の処理）

**検証**: プラグイン起動時にGPUメモリ使用量が増加すること（nvidia-smi確認）

#### Day 8-9: サンプルレート変換実装
**タスク**:
- [ ] `activate()` でホストのサンプルレートを取得
- [ ] 入力レート判定（44.1kHz / 48kHz）
- [ ] 16倍アップサンプリング設定
- [ ] `run()` 内でGPU処理呼び出し
- [ ] 出力バッファへの書き込み

**検証**:
```bash
# テスト用WAVファイルでオフライン処理
lv2_jack_host http://github.com/michihito/gpu_upsampler
# 出力が705.6kHzであることを確認
```

#### Day 10: 動作確認とデバッグ
**タスク**:
- [ ] メモリリークチェック（valgrind）
- [ ] GPU処理時間計測
- [ ] 音質確認（周波数解析）
- [ ] 長時間動作テスト（1時間以上）

**目標パフォーマンス**:
- 処理速度: 10倍速以上（同期処理でも許容範囲）
- メモリ: 安定使用（リークなし）
- 音質: 周波数特性が-187dB以上

---

### Week 3: Filter-Chain統合 (5日間)

#### Day 11-12: Filter-Chain設定ファイル作成
**タスク**:
- [ ] `~/.config/pipewire/pipewire.conf.d/90-gpu-upsampler-sink.conf` 作成
- [ ] SMSL DACの実際のnode.nameを確認して設定
- [ ] PipeWire再起動
- [ ] `pw-dump` でノードグラフ確認

**検証コマンド**:
```bash
systemctl --user restart pipewire pipewire-pulse
pw-dump | grep -A 20 "gpu_upsampler"
```

#### Day 13: WirePlumber自動ルーティング設定
**タスク**:
- [ ] `~/.config/wireplumber/main.lua.d/51-gpu-upsampler-routing.lua` 作成
- [ ] Easy Effects → GPU Upsampler ルール
- [ ] 音楽プレイヤー → Easy Effects ルール
- [ ] WirePlumber再起動
- [ ] 接続確認

**検証**: `pw-link` または `helvum` (グラフィカルツール) で接続グラフ確認

#### Day 14-15: Easy Effects統合テスト
**タスク**:
- [ ] Easy Effects設定（gsettings）
- [ ] イコライザ設定確認
- [ ] Spotifyで音楽再生テスト
- [ ] フルチェーン動作確認:
  ```
  Spotify → Easy Effects → GPU Upsampler → SMSL DAC
  ```
- [ ] 処理順序の検証（イコライザが先に適用されているか）

**検証方法**:
1. イコライザで特定周波数を大きく増幅
2. FFTアナライザで出力確認
3. GPU Upsamplerが増幅後の信号を処理していることを確認

---

### Week 4: 非同期化と最適化 (5日間)

#### Day 16-18: 非同期GPU処理実装
**タスク**:
- [ ] Lock-freeリングバッファ実装（boost::lockfree::spsc_queue検討）
- [ ] GPU処理専用スレッド作成
- [ ] `run()` からリングバッファへpush
- [ ] GPU threadでpop → 処理 → 出力バッファへpush
- [ ] レイテンシ測定

**目標**:
- リアルタイムスレッド (`run()`) のブロッキング時間: <100μs
- 総レイテンシ: <50ms

#### Day 19: レイテンシ最適化
**タスク**:
- [ ] バッファサイズ調整（2ブロック vs 3ブロック）
- [ ] CUDA Stream使用検討
- [ ] ステレオ並列処理の有効化
- [ ] LV2レイテンシ報告実装

**最適化項目**:
```cpp
// レイテンシ報告
float get_latency() {
    return (ring_buffer_size + gpu_processing_samples) / sample_rate;
}
```

#### Day 20: 最終テストと文書化
**タスク**:
- [ ] 全機能統合テスト
- [ ] 長時間安定性テスト（24時間連続再生）
- [ ] CPU/GPU使用率測定
- [ ] ユーザーマニュアル作成
- [ ] インストールスクリプト作成

**最終検証項目**:
- ✅ 音楽再生が正常に動作
- ✅ Easy Effectsのイコライザが適用されている
- ✅ システム音は内蔵スピーカーから出力
- ✅ GPU処理が安定動作（メモリリークなし）
- ✅ レイテンシが許容範囲内（<50ms）

---

## 4. ビルドシステム

### CMakeLists.txt更新内容

```cmake
# Phase 2エンジンを静的ライブラリ化
add_library(gpu_upsampler_core STATIC
    src/convolution_engine.cu
    src/audio_io.cpp
)

target_link_libraries(gpu_upsampler_core
    CUDA::cudart
    CUDA::cufft
    CUDA::nvml
    ${SNDFILE_LIBRARIES}
)

# 既存のコマンドラインツール（変更なし）
add_executable(gpu_upsampler
    src/main.cpp
)

target_link_libraries(gpu_upsampler
    gpu_upsampler_core
)

# 新規: LV2プラグイン
add_library(gpu_upsampler_lv2 MODULE
    src/lv2_plugin/gpu_upsampler_lv2.cpp
)

set_target_properties(gpu_upsampler_lv2 PROPERTIES
    PREFIX ""  # libをつけない
    OUTPUT_NAME "gpu_upsampler"
)

target_link_libraries(gpu_upsampler_lv2
    gpu_upsampler_core
)

# LV2インストール
install(TARGETS gpu_upsampler_lv2
    LIBRARY DESTINATION /usr/local/lib/lv2/gpu_upsampler.lv2/
)

install(FILES
    src/lv2_plugin/manifest.ttl
    src/lv2_plugin/gpu_upsampler.ttl
    DESTINATION /usr/local/lib/lv2/gpu_upsampler.lv2/
)

# フィルタ係数インストール
install(FILES
    data/coefficients/filter_1m_min_phase.bin
    DESTINATION /usr/local/share/gpu_upsampler/
)
```

---

## 5. テスト計画

### 5.1 単体テスト

#### LV2プラグイン単体
```bash
# プラグイン認識確認
lv2ls | grep gpu_upsampler

# メタデータ確認
lv2info http://github.com/michihito/gpu_upsampler

# JACK経由でテスト
jalv.gtk http://github.com/michihito/gpu_upsampler

# コマンドライン版（デバッグ用）
lv2_jack_host http://github.com/michihito/gpu_upsampler
```

#### GPU処理単体
```bash
# 既存のコマンドラインツールで動作確認
./build/gpu_upsampler test.wav output.wav \
    --filter data/coefficients/filter_1m_min_phase.bin
```

### 5.2 統合テスト

#### PipeWireノード確認
```bash
# ノード一覧
pw-cli list-objects Node

# GPU Upsampler Sink確認
pw-dump | grep -A 30 "gpu_upsampler_sink"

# Easy Effectsからの接続確認
pw-link --list | grep -A 5 easyeffects
```

#### 音声ルーティング確認
```bash
# 音楽プレイヤー起動
spotify &

# 接続状態確認（Spotifyがgpu_upsampler_sinkに接続されているか）
pw-link --list | grep spotify

# SMSL DACの入力確認
pw-link --list | grep SMSL
```

### 5.3 性能テスト

#### レイテンシ測定
```bash
# jack_iodelay使用（要jack-audio-tools）
jack_iodelay

# PipeWireレイテンシ確認
pw-metadata -n settings 0 | grep latency
```

#### GPU使用率モニタリング
```bash
# リアルタイムモニタリング
watch -n 0.5 nvidia-smi

# ログ記録
nvidia-smi dmon -s u -c 1000 > gpu_usage.log &
# 音楽再生中のGPU使用率を記録
```

#### CPU使用率
```bash
# PipeWireプロセスのCPU使用率
top -p $(pgrep -f pipewire)

# GPU処理スレッドのCPU使用率（プラグイン動作中）
top -H -p $(pgrep -f pipewire) | grep gpu
```

### 5.4 音質テスト

#### 周波数特性測定
```python
# Python + scipy で周波数解析
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

rate, data = wav.read('output_705600hz.wav')
f, pxx = signal.welch(data[:, 0], rate, nperseg=65536)

# -187dB以下の減衰を確認（22.05kHz以上）
import matplotlib.pyplot as plt
plt.semilogy(f/1000, pxx)
plt.xlim(0, 50)
plt.ylim(1e-20, 1e0)
plt.xlabel('Frequency (kHz)')
plt.ylabel('PSD')
plt.axvline(22.05, color='r', linestyle='--', label='Stopband start')
plt.legend()
plt.savefig('frequency_response_test.png')
```

#### THD+N測定（オプション）
- Room EQ Wizard (REW) 使用
- ループバック測定（DACアナログ出力 → ADC入力）
- 期待値: THD+N < 0.001% (1kHz, -3dBFS)

---

## 6. トラブルシューティング

### 6.1 プラグインが認識されない

**症状**: `lv2ls` でプラグインが表示されない

**確認事項**:
```bash
# インストール先確認
ls -la /usr/local/lib/lv2/gpu_upsampler.lv2/

# TTLファイルの文法チェック
lv2_validate /usr/local/lib/lv2/gpu_upsampler.lv2/manifest.ttl

# LV2_PATHに含まれているか
echo $LV2_PATH
export LV2_PATH=/usr/local/lib/lv2:$LV2_PATH
```

### 6.2 GPU Upsampler Sinkが作成されない

**症状**: `pw-cli list-objects` に `gpu_upsampler_sink` が表示されない

**確認事項**:
```bash
# PipeWire設定ファイル構文チェック
pipewire -c ~/.config/pipewire/pipewire.conf

# Filter-Chainモジュールがロードされているか
pw-cli list-objects Module | grep filter-chain

# ログ確認
journalctl --user -u pipewire -f
```

### 6.3 Easy Effectsが接続されない

**症状**: Easy Effectsの音声がGPU Upsampler Sinkに流れない

**確認事項**:
```bash
# Easy Effectsの出力先設定
gsettings get com.github.wwmm.easyeffects.streamoutputs target-object

# WirePlumberルールの確認
cat ~/.config/wireplumber/main.lua.d/51-gpu-upsampler-routing.lua

# WirePlumber再起動
systemctl --user restart wireplumber
```

### 6.4 音が出ない

**症状**: 音楽を再生しても無音

**チェックリスト**:
1. GPU処理が実行されているか
   ```bash
   nvidia-smi  # GPU使用率が上がっているか
   ```

2. SMSL DACが選択されているか
   ```bash
   pactl list sinks short
   pactl set-default-sink alsa_output.usb-SMSL_SMSL_USB_AUDIO-00.analog-stereo
   ```

3. ミュートされていないか
   ```bash
   pactl list sinks | grep -A 10 SMSL | grep Mute
   ```

4. サンプルレート不一致
   ```bash
   pw-metadata -n settings 0 clock.rate  # 705600であるべき
   ```

### 6.5 レイテンシが大きすぎる

**症状**: 動画とリップシンクがずれる（>100ms）

**対策**:
```bash
# バッファサイズ削減
# ~/.config/pipewire/pipewire.conf
default.clock.quantum = 4096  # デフォルト: 8192

# PipeWire再起動
systemctl --user restart pipewire
```

---

## 7. 成功基準

### Phase 3完了の定義

以下のすべての項目が満たされた時点でPhase 3完了とする。

#### 機能要件
- [x] LV2プラグインが `lv2ls` で認識される
- [ ] Filter-ChainでGPU Upsampler Sinkが作成される
- [ ] 音楽プレイヤーの音声が自動的にEasy Effectsへルーティングされる
- [ ] Easy Effectsの音声がGPU Upsampler Sinkへルーティングされる
- [ ] GPU処理が正常に動作する（音声出力が確認できる）
- [ ] イコライザが44.1kHzで動作する
- [ ] 最終出力が705.6kHzである

#### 性能要件
- [ ] GPU処理速度: 10倍速以上（リアルタイム処理可能）
- [ ] レイテンシ: 50ms以下
- [ ] GPU使用率: 平均30%以下（RTX 2070 Super）
- [ ] CPU使用率: PipeWireプロセス 10%以下
- [ ] メモリリーク: なし（24時間連続動作）

#### 音質要件
- [ ] 周波数特性: 22.05kHz以上で-187dB以下の減衰
- [ ] THD+N: 0.01%以下（測定可能な場合）
- [ ] ノイズフロア: -140dBFS以下
- [ ] 主観評価: プリリンギングなし、クリアな音質

---

## 8. 次ステップ（Phase 4以降）

Phase 3完了後の発展的な改善項目。

### 8.1 追加最適化
- [ ] CUDA Graphによる更なる高速化
- [ ] Multi-rate対応（48kHz, 88.2kHz, 96kHz入力）
- [ ] Dynamic blockSize調整（レイテンシ自動最適化）
- [ ] Pinned memoryによるPCIe転送高速化

### 8.2 機能拡張
- [ ] GUIコントロールパネル（GTK4）
- [ ] リアルタイム周波数アナライザ
- [ ] プリセット機能（音楽ジャンル別設定）
- [ ] ABXテスト用比較機能

### 8.3 互換性向上
- [ ] AMD GPU対応（ROCm/HIP移植）
- [ ] Intel GPU対応（Level Zero API）
- [ ] Vulkan Compute移植（ベンダー非依存）
- [ ] ARM64対応（Jetson等）

### 8.4 配布
- [ ] AURパッケージ作成（Arch Linux）
- [ ] PPAリポジトリ（Ubuntu/Debian）
- [ ] Flatpakパッケージ
- [ ] 公式ドキュメントサイト構築

---

## 9. 参考資料

### LV2プラグイン開発
- LV2公式仕様: https://lv2plug.in/
- LV2 Book: https://lv2plug.in/book/
- Jalv（LV2ホスト）: https://drobilla.net/software/jalv.html

### PipeWire
- 公式ドキュメント: https://docs.pipewire.org/
- Filter-Chain設定例: https://gitlab.freedesktop.org/pipewire/pipewire/-/wikis/Filter-Chain
- WirePlumber: https://pipewire.pages.freedesktop.org/wireplumber/

### 類似プロジェクト
- GPU Impulse Reverb (OpenCL畳み込みリバーブ)
- HQPlayer (商用高品質アップサンプラー)
- CamillaDSP (CPU版FIRフィルタエンジン)

---

**ドキュメント作成日**: 2025-11-21
**対象Phase**: Phase 3
**推定工数**: 4週間（20営業日）
**必須スキル**: C++17, CUDA, LV2 API, PipeWire設定, Lua
