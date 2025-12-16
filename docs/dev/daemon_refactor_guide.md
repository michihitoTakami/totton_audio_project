# Daemon リファクタリング方針（モジュール境界とスケルトン）

## 目的
- `alsa_daemon.cpp` のモノリス化を解消し、各責務の置き場所とイベント契約を明文化する。
- 後続タスク (#553/#580/#554/#552) が新規ファイルを迷わず追加できるよう最小スケルトンを提供する。

## ディレクトリごとの責務早見表
| ディレクトリ | 役割 | 代表ファイル |
| --- | --- | --- |
| `src/daemon/control/` | ZMQコマンド受付・ディスパッチ、統計出力、終了処理 | `control_plane.cpp`, `shutdown_manager.cpp` |
| `src/daemon/control/handlers/` | 制御イベントのハンドラ登録ポイント（イベント→モジュールの橋渡し） | `handler_registry.{h,cpp}` |
| `src/daemon/audio_pipeline/` | 入力→GPUアップサンプル→CF→出力バッファ管理、ソフトミュート補助 | `audio_pipeline.cpp`, `rate_switcher.cpp`, `filter_manager.cpp`, `soft_mute_runner.cpp` |
| `src/daemon/output/` | ALSA 出力のライフサイクル管理、デバイス切替対応 | `alsa_output.{h,cpp}` |
| `src/daemon/pcm/` | DAC能力の判定/選択 | `dac_manager.cpp` |
| `src/daemon/metrics/` | ランタイム統計の集約/永続化 | `runtime_stats.cpp` |

## イベント契約（共通ヘッダ）
- `include/daemon/api/events.h`
  - `RateChangeRequested { detectedInputRate, rateFamily }`
  - `DeviceChangeRequested { preferredDevice, mode }`
  - `FilterSwitchRequested { filterPath, phaseType, reloadHeadroom }`
  - `EventDispatcher` : Rate/Device/Filter 用に subscribe/publish を提供
- `include/daemon/api/dependencies.h`
  - `DaemonDependencies` : Config, SoftMute、各モジュール/Atomic 状態への参照を一元化
  - `DaemonContext` : Dispatcher と依存関係の束ね

## スケルトン概要
- パイプライン: `RateSwitcher` がレートイベントを受けて atomic 状態を更新。`FilterManager`/`SoftMuteRunner` はフィルタ切替イベントを購読し、ヘッドルーム更新やソフトミュートをトリガ。
- 出力: `AlsaOutput` がデバイス変更イベントを購読し、出力準備フラグを管理。
- 制御: `HandlerRegistry` がコントロールプレーンの登録窓口となり、イベントディスパッチャに購読を事前登録する。
- テスト: `tests/cpp/daemon/test_daemon_skeleton.cpp` でイベント→ハンドラ配線がコンパイル/実行できることを確認。

## 追加の開発ガイド
- 新規ロジックは極力グローバル変数に触れず、`DaemonDependencies` で依存を明示して注入する。
- コマンド/イベント起点のフローは「Control handler → EventDispatcher → 各モジュール」の順に配線する。
- レート変更・デバイス変更・フィルタ変更は上記イベント型を再利用し、イベント数を増やす場合も `events.h` に追加する。
- `alsa_daemon.cpp` からは共通ヘッダをインクルードし、将来的な分離先が分かるよう参照を残す（本タスクではスタブ運用）。

## 確認手順（本タスク）
1. `cmake --build build`（テスト込みビルド時は `cpu_tests` に新規ファイルが含まれる）
2. `ctest -R DaemonSkeleton` でイベント配線テストを実行（将来の CI 組み込み用）
3. 新規モジュール追加時は上記ディレクトリ責務表に追記する。
