# 4ch FIR統合最小アーキテクチャ設計

## 背景
EPIC #884 で掲げた既存畳み込みエンジンへの Crossfeed/HRTF 4ch FIR統合は、まず最小限の正しさと安全性を担保しながら次段階を着手できるかたちでまとめる必要があります。本 Issue #885 では実装可能なデータフローと切替ポリシー、API境界を明示して後続タスクにブロッキングが出ない状態を目指します。

## データフロー（既存エンジンとの接続点）
1. `src/daemon/audio_pipeline/audio_pipeline.cpp` の `AudioPipeline::process` が ALSA からの平行ステレオ入力を受け取り、アップサンプラ（`deps_.upsampler.process`）で Hi-Rate（705.6kHz / 768kHz）ブロックを生成します。ここまでが既存の主系統です。
2. Crossfeed を ON にすると、4ch FIR 用のストリーミングバッファ（`deps_.cfStreamInput*`, `deps_.cfStreamAccumulated*`）にアップサンプラ出力を蓄積し、`CrossfeedEngine::HRTFProcessor::processStreamBlock` が `outputL/R` を生成して初めて `playback_buffer` にエンキューします。これにより未処理のアップサンプル音声と処理済み音声の混在を防ぎます。
3. 4ch FIR は `CrossfeedEngine` 側の `LL/LR/RL/RR` フィルタセットを用い、`Out_L = In_L*LL + In_R*RL`, `Out_R = In_L*LR + In_R*RR` の合成を GPU 上で行います（`include/crossfeed_engine.h` 参照）。
4. 出力後は `PlaybackBufferManager` のリングバッファに戻り、`renderOutput` 側で SoftMute / リミッタを通して ALSA に渡されます。必要であれば `streaming_cache::StreamingCacheManager` がバッファリセットや soft mute を差し込みます。

## ブロック境界とストリーミングしきい値
- `include/io/playback_buffer.h` の `PlaybackBuffer::computeReadyThreshold` は、クロスフィード有効時に HRTF の `getStreamValidInputPerBlock()`（`g_hrtf_processor` が返す入力ブロック）をしきい値として使います。これにより GPU が一定量の 4ch FIR を処理し終えた段階で ALSA 側の待ち合わせを解除し、過剰な蓄積を防ぎます。
- `g_upsampler->getStreamValidInputPerBlock()` とアップサンプル比率から、実際に `playback_buffer` に溜まるフレーム数（producer block）が算出され、`computeReadyThreshold` の `producerBlockSize` として渡されます。クロスフィード OFF 時は periodSize の 3倍前後を目安に ready threshold が決まり、ON 時は `max(periodSize, crossfeedBlock)` に制限されます。
- Crossfeed で生成される出力も `enqueueOutputFramesLocked` を通じて `PlaybackBufferManager::enqueue` されるため、バッファはアップサンプル出力と同一のキャパシティ管理ロジックを共有します。

## ON/OFF 切替と状態リセット
1. `alsa_daemon.cpp` の `reset_crossfeed_stream_state_locked` はクロスフィードのストリーミングバッファと `HRTFProcessor::resetStreaming()` をクリアし、切替時の残留データから生じる軋み音やバッファ膨張を防ぎます。
2. `control_plane`（`src/daemon/control/control_plane.cpp`）では `deps_.crossfeed.enabledFlag`/`mutex` を介して `CrossfeedEngine::HRTFProcessor` へのアクセスを保護しており、`CROSSFEED_ENABLE/DISABLE` API は `resetStreamingState` を呼ぶことで状態を同期させます。
3. Crossfeed を有効化するタイミングでは `StreamingCacheManager::onCrossfeedReset` を通じて playback buffer をリセットし、ソフトミュートと `softMute::Controller` が `renderOutput` 側で連動してポップ音を吸収します。
4. 無効化後は `deps_.cfStreamInput*` を即時クリアして新しいアップサンプルブロックを待ち、次回有効化時にも `processStreamBlock` が新しいブロックから再スタートするよう整合性をとります。

## 内部インターフェースと CROSSFEED_* API の保持
- 既存の ZeroMQ API である `CROSSFEED_ENABLE`, `CROSSFEED_DISABLE`, `CROSSFEED_GET_STATUS`, `CROSSFEED_*` のエンドポイント（`include/network/zeromq_interface.h` / `src/network/zeromq_interface.cpp`）はそのまま維持し、`control_plane` から `deps_.crossfeed` を通じて `HRTFProcessor` の状態を切り替えます。
- `CrossfeedEngine::HRTFProcessor` は `setCombined` 系コマンドで `LL/LR/RL/RR` の複合フィルタを受け取れる設計を維持する（`include/crossfeed_engine.h` の `setCombinedFilter` 参照）。これにより制御面で Web UI や Python スクリプトが既存 API を呼び出せる。
- GPU 側の実行経路（`audio_pipeline` での `processStreamBlock` 呼び出し）と `CROSSFEED` API の橋渡しは、`CrossfeedEngine::HRTFProcessor` を `deps_.crossfeedProcessor` として渡す形で保持し、新しい 4ch FIR 実行パスに置き換えても API 仕様を変えない。

## 次段階で解消すべきチェックポイント
- 以上で定義したストリーミング経路に 4ch FIR 実行ルート（Issue #886）が挿入され、`processStreamBlock` の戻り値を `upsampler` に渡す。
- 切替時のソフトミュートと underflow/overflow 対応（Issue #888）では `streaming_cache` を介した再試行と `PlaybackBuffer` しきい値の調整を使って実装。
- テスト・メトリクス（Issue #889）としてクロスフィード ON/OFF の切替ログと ready threshold 変化、`runtime_stats` のドロップ/ピークを観測する。
- 旧 `crossfeed_engine` 経路（Issue #890）は十分に検証されたら段階的に対象外にする。
