# GPU Audio Upsampler

高品位なFIRフィルタをCUDAで並列実行し、44.1kHzオーディオを最大16倍 (705.6/352.8kHz) にリアルタイム/オフラインでアップサンプリングするプロジェクトです。PipeWire経由の入力をGPUで処理し、ALSA/DACへ高サンプルレート出力するデーモンと、オフライン変換用CLI/LV2プラグインを同梱します。

## アーキテクチャ概要
- `src/convolution_engine.cu`: Overlap-Save方式の1Mタップ最小位相FIRをCUDA/cuFFTで実装するコア。ステレオ並列ストリーム、オーバーラップ保持付きストリーミングAPIを提供。
- `src/alsa_daemon.cpp`: PipeWireから44.1kHz floatを受信し、GPUで16xアップサンプル後、ALSA `hw:3,0` へ705.6kHz S32_LE出力するデーモン（SMSL D400EX想定）。
- `src/pipewire_daemon.cpp`: PipeWire→GPU→PipeWireのラウンドトリップ用デーモン。
- `src/lv2_plugin/`: LV2プラグイン実装（ホスト側が705.6kHz入力を提供する構成）。
- `scripts/`: フィルタ生成/解析ツール（例: `scripts/analyze_waveform.py` でクリック検出）。
- `data/coefficients/`: 1MタップFIR係数 (`filter_1m_min_phase.bin`) とメタデータ。
- `docs/`: 調査・セットアップ資料（`setup_guide.md`, `crackling_noise_investigation.md` など）。

## フィルタとサンプルレート
- 44.1kHz入力: `data/coefficients/filter_1m_min_phase.bin` を自動選択 (16x)。
- 48kHz入力: `data/coefficients/filter_48k_1m_min_phase.bin` が存在すれば自動選択 (16x)。未生成の場合は明示的に `--filter` で指定するか、以下で生成:
  ```bash
  python scripts/generate_filter.py --input-rate 48000 --stopband-start 24000 --passband-end 21500 --output-prefix filter_48k_1m_min_phase
  ```
- 48kHz用のバイナリが無い場合は、警告の上で44.1kHz用フィルタにフォールバックします（生成を推奨）。
- その他レートは非サポート。必要に応じて適合フィルタを生成してください。

## 設定ファイルと再読込
- `config.json` をルートに置くと、CLIのデフォルト値と `gpu_upsampler_alsa` の起動設定（ALSAデバイス、バッファ、フィルタ、ゲインなど）に適用されます。  
- `gpu_upsampler_alsa` は SIGHUP で設定を再読込し、内部を再初期化します。Web UI の `/restart` も SIGHUP を送ります。手動で再読込したい場合:
  ```bash
  pkill -HUP -f gpu_upsampler_alsa   # または PID を指定
  ```

### 信号フロー (ALSAデーモン)
```
App (PipeWire) -> gpu_upsampler_sink.monitor -> GPU Upsampler (CUDA 16x) -> ALSA hw:3,0 -> USB DAC -> Analog
```

## ビルド
CUDA ToolkitとPipeWire/ALSAヘッダを用意した上で:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## 実行
- オフライン変換:
  ```bash
  ./build/gpu_upsampler input.wav output.wav --ratio 16 --block 4096
  ```
- PipeWire→ALSAリアルタイム出力:
  ```bash
  ./build/gpu_upsampler_alsa
  # 別ターミナルで必要ならモニター接続
  pw-link gpu_upsampler_sink:monitor_FL "GPU Upsampler Input:input_FL"
  pw-link gpu_upsampler_sink:monitor_FR "GPU Upsampler Input:input_FR"
  ```
- PipeWireラウンドトリップ:
  ```bash
  ./build/gpu_upsampler_daemon
  ```
- LV2プラグインはホストから通常の方法で読み込んでください。

### 再起動後のクイック手順
1. デーモン起動（必要ならデバイス指定）  
   ```bash
   ./scripts/restart_alsa_daemon.sh hw:3,0
   ```
2. PipeWireシンク作成・配線（アプリ名は任意、デフォルトspotify）  
   ```bash
   ./scripts/setup_pw_links.sh spotify
   ```
3. サウンド設定で出力デバイスを「GPU Upsampler (705.6kHz)」に選択。  

※ Easy Effects を経由する場合も同じ手順で、アプリは Easy Effects Sink に向ける。

### Easy Effects を通す場合のPipeWire配線例
- 出力経路にイコライザ等を挿入したいときは、再生側を `easyeffects_sink` に向け、Easy Effects のモニター出力を本プロジェクトのシンクへ接続します。
  ```bash
  # 再生アプリ → Easy Effects
  pw-link <app>:output_FL easyeffects_sink:playback_0
  pw-link <app>:output_FR easyeffects_sink:playback_1

  # Easy Effects モニタ → GPU Upsampler
  pw-link easyeffects_sink:monitor_0 gpu_upsampler_sink:playback_0
  pw-link easyeffects_sink:monitor_1 gpu_upsampler_sink:playback_1

  # GPU Upsampler monitor → 処理入力
  pw-link gpu_upsampler_sink:monitor_0 "GPU Upsampler Input:input_0"
  pw-link gpu_upsampler_sink:monitor_1 "GPU Upsampler Input:input_1"
  ```
  Easy Effects 内部の「出力デバイス」を `Easy Effects Sink`、「モニター」を有効化しておくと上記配線が活きます。

## 主要仕様
- アップサンプル比: 16x（44.1kHz→705.6kHz）/ オプションで8x。
- FIRフィルタ: 1,000,000タップ最小位相、FFTサイズ1,048,576。
- Overlap-Save: オーバーラップ999,999サンプル、有効出力48,577サンプル/ブロック。
- 出力形式: S32_LE (デーモン), float32 (PipeWire/LV2内部)。

## トラブルシュートの入口
- プチプチ音・クリック: `docs/crackling_noise_investigation.md` を参照し、オーバーラップ保存とストリーミング設定を確認。
- セットアップ全般: `docs/setup_guide.md` に PipeWire null sink 作成や接続手順を記載。
