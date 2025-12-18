# 4ch FIR統合最小アーキテクチャ設計

## 背景
EPIC #884 で掲げた既存畳み込みエンジンへの Crossfeed/HRTF 4ch FIR統合は、まず最小限の正しさと安全性を担保しながら次段階を着手できるかたちでまとめる必要があります。本 Issue #885 では実装可能なデータフローと切替ポリシー、API境界を明示して後続タスクにブロッキングが出ない状態を目指します。

## 前提（譲れないこと）
- **HUTUBS 由来の固定HRTF（±30°固定 / 4ch: LL/LR/RL/RR）は必ず維持**する（簡易クロスフィードへの置換はしない）。
- `scripts/filters/generate_hrtf.py` が生成する `data/crossfeed/hrtf/hrtf_{xs,s,m,l,xl}_{44k,48k}.bin/.json` を利用する。
- head size は **5段階（xs/s/m/l/xl）を必ず使い分ける**（xs を s に丸めない）。
- 今回の統合スコープは「安定稼働の一本化」が主目的で、**combined filter / woodworth / ホットスワップ / 最適化**は対象外（必要なら別Issueで追加）。

## データフロー（既存エンジンとの接続点）
1. `src/daemon/audio_pipeline/audio_pipeline.cpp` の `AudioPipeline::process` が ALSA からの平行ステレオ入力を受け取り、アップサンプラ（`deps_.upsampler.process`）で Hi-Rate（705.6kHz / 768kHz）ブロックを生成します。ここまでが既存の主系統です。
2. Crossfeed を ON にすると、4ch FIR 用のストリーミングバッファ（`deps_.cfStreamInput*`, `deps_.cfStreamAccumulated*`）にアップサンプラ出力を蓄積し、**統合版の 4ch FIR 実行パス**が `outputL/R` を生成して初めて `playback_buffer` にエンキューします。これにより未処理のアップサンプル音声と処理済み音声の混在を防ぎます。
3. 4ch FIR は HUTUBS 固定HRTF（LL/LR/RL/RR）を用い、`Out_L = In_L*LL + In_R*RL`, `Out_R = In_L*LR + In_R*RR` を GPU 上で合成します。フィルタ係数は `data/crossfeed/hrtf` の 5サイズ×2レートの生成物（channel-major: LL→LR→RL→RR）を読み込みます。
4. 出力後は `PlaybackBufferManager` のリングバッファに戻り、`renderOutput` 側で SoftMute / リミッタを通して ALSA に渡されます。必要であれば `streaming_cache::StreamingCacheManager` がバッファリセットや soft mute を差し込みます。

## ブロック境界とストリーミングしきい値
- `include/io/playback_buffer.h` の `PlaybackBuffer::computeReadyThreshold` は、クロスフィード有効時に **統合版 4ch FIR の入力ブロックサイズ**（「何サンプル蓄積すると1ブロック出力できるか」）をしきい値として使います。これにより GPU が一定量の 4ch FIR を処理し終えた段階で ALSA 側の待ち合わせを解除し、過剰な蓄積を防ぎます。
- `g_upsampler->getStreamValidInputPerBlock()` とアップサンプル比率から、実際に `playback_buffer` に溜まるフレーム数（producer block）が算出され、`computeReadyThreshold` の `producerBlockSize` として渡されます。クロスフィード OFF 時は periodSize の 3倍前後を目安に ready threshold が決まり、ON 時は `max(periodSize, crossfeedBlock)` に制限されます。
- Crossfeed で生成される出力も `enqueueOutputFramesLocked` を通じて `PlaybackBufferManager::enqueue` されるため、バッファはアップサンプル出力と同一のキャパシティ管理ロジックを共有します。

## ON/OFF 切替と状態リセット
1. `alsa_daemon.cpp` の `reset_crossfeed_stream_state_locked` はクロスフィードのストリーミングバッファ（蓄積量のみ）と `ConvolutionEngine::FourChannelFIR::resetStreaming()` をクリアし、切替時の残留データから生じる軋み音やバッファ膨張を防ぎます。
2. `control_plane`（`src/daemon/control/control_plane.cpp`）では `deps_.crossfeed.enabledFlag`/`mutex` を介して `ConvolutionEngine::FourChannelFIR` へのアクセスを保護しており、`CROSSFEED_ENABLE/DISABLE` API は `resetStreamingState` を呼ぶことで状態を同期させます。
3. Crossfeed を有効化するタイミングでは `StreamingCacheManager::onCrossfeedReset` を通じて playback buffer をリセットし、ソフトミュートと `softMute::Controller` が `renderOutput` 側で連動してポップ音を吸収します。
4. 無効化後は `deps_.cfStreamInput*` の蓄積量をリセットして新しいアップサンプルブロックを待ち、次回有効化時にも `processStreamBlock` が新しいブロックから再スタートするよう整合性をとります。

## 内部インターフェースと CROSSFEED_* API の保持
- 既存の ZeroMQ API である `CROSSFEED_ENABLE`, `CROSSFEED_DISABLE`, `CROSSFEED_GET_STATUS`, `CROSSFEED_*` のエンドポイント（`include/network/zeromq_interface.h` / `src/network/zeromq_interface.cpp`）はそのまま維持し、`control_plane` から `deps_.crossfeed` を通じて `ConvolutionEngine::FourChannelFIR` の状態を切り替えます。
- 今回は `CROSSFEED_SET_COMBINED` / `CROSSFEED_GENERATE_WOODWORTH` は対象外（将来の拡張候補として残す）。まずは `data/crossfeed/hrtf` の固定HRTFを確実に使い、安定稼働を優先する。
- GPU 側の実行経路（`audio_pipeline` での 4ch FIR 呼び出し）と `CROSSFEED` API の橋渡しは、API 仕様を変えずに内部実装を差し替えられる形で維持する（`enabledFlag`/`mutex`/状態リセットの依存は継続）。

## 次段階で解消すべきチェックポイント
- 以上で定義したストリーミング経路に 4ch FIR 実行ルート（Issue #886）を挿入し、**クロスフィード有効時は 4ch FIR 出力のみを enqueue**する（未処理音声との混在禁止）。
- 切替時のソフトミュートと underflow/overflow 対応（Issue #888）では `streaming_cache` を介した再試行と `PlaybackBuffer` しきい値の調整を使って実装。
- テスト・観測（切替ログ/ready threshold/`runtime_stats` 等）は各Issueの Done に内包する（独立Issueは作らない）。
- 旧 `crossfeed_engine` 経路（Issue #890）は十分に検証されたら段階的に対象外にする（ただし HRTFデータと生成スクリプトは維持）。
