# 低遅延パーティション畳み込み: GPUコア実装

## 背景
- 現行は 2M タップを単一 FFT で処理するため、1 ブロックあたり 0.5s 超のレイテンシが発生。
- RTP/リアルタイム利用では 10〜50ms 程度まで遅延を抑える必要がある。
- `partition_plan` の土台は存在するが、GPU パイプラインに組み込まれていない。

## やること
1. `GPUUpsampler` に PartitionPlan を結線し、パーティションごとに FFT サイズ・有効サンプル数・メモリを保持する `PartitionState` を実装。
2. ストリーミング処理を fast/tail パーティション aware に分岐し、fast パーティションからの即時出力＋tail の遅延加算を行う。
3. GPU 上に遅延ライン／リングバッファを用意し、tail の畳み込み結果を正しいサンプル位置へ加算できるようにする。
4. 入力停止・再開時のリセット、エラー時のフォールバック（従来モードへ戻る）を整備。
5. CUDA ストリーム/バッファ確保・解放をパーティション数に応じて正しく管理。

## 受け入れ基準
- 低遅延モード有効時、指定した fast パーティションサイズに応じてブロック遅延が短縮される。
- tail が追いついた後の周波数応答が従来フィルタと一致する（手動検証で可）。
- ストリーミング中に XRUN や未処理データが発生せず、停止→再生で状態がリセットされる。
- 従来モード（単一 FFT）も変更なしで動作する。

## メモ (2025-11-28 実装状況)
- `AppConfig::partitionedConvolution` で low-latency パスを切り替え可能（Issue #352 までは `config.json` での指定は未対応、暫定的にコード側で設定）。
- GPU 側では `PartitionPlan`/`PartitionState` を導入し、fast パーティション FFT + tail パーティション FFT をステレオ毎に逐次実行する構成。
- fast パーティション出力は既存ストリーミングと互換の `StreamFloatVector` へ即時書き出し。tail パーティションは同一ブロック内で加算する実装（初期バージョンでは遅延加算を行わず、ブロック長の短縮にフォーカス）。
- multi-rate / quad-phase モードとは排他。設定時は自動的に partition 機能を無効化し、ログへ理由を出力。
  このドキュメントは新方針に移行中で、今後は low-latency モードでも quad-phase（linear/hybrid） を選択できるようにし、排他制御を解除する予定である（Issue #467 参照）。
- Crossfeed(HRTF) と EQ は低遅延モードでは現在サポート外。該当設定が有効な場合は起動時に強制的に無効化される（Issue #353 で拡張予定）。
- ZeroMQ コマンド経由での CROSSFEED_ENABLE もガード済み。low-latency 中は ERR を返す。
- 既存ストリーミング API との整合性: `initializeStreaming` / `resetStreaming` / `freeStreamingBuffers` などは partition 有効時に専用コードへフォールバックする。失敗時は従来モードへ自動退避。

## 手動確認メモ
1. `AppConfig::partitionedConvolution.enabled = true` を直接設定して ALSA/PipeWire daemon を起動。
2. `g_upsampler->getStreamValidInputPerBlock()` が fast パーティションの valid 出力量へ縮小されていることをログで確認。
3. `scripts/run_tests.sh` では GPU ストリーミングが走らないため未検証。実機で `CROSSFEED_ENABLE` が ERR を返すこと、EQ 設定が強制 OFF になることをチェック。
4. フォールバック動作: `partitionPlan` の構築失敗や `initializePartitionedStreaming` 失敗時は警告ログと共に従来モードで継続することを確認。
