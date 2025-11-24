# Rate Negotiation Handshake Design

> **Related Issues:**
> - EPIC: [#37 レート自動交渉](https://github.com/michihitoTakami/michy_os/issues/37)
> - Task: [#218 入力サンプルレート検出とハンドシェイク](https://github.com/michihitoTakami/michy_os/issues/218)

## 1. 概要

本ドキュメントは、入力サンプルレートの動的検出とレート切り替えハンドシェイクの設計を定義する。

### 1.1 目的

- PipeWire/ALSA入力のサンプルレートを検出し、44.1k/48kファミリと2/4/8/16倍から最適な出力レートを決定
- ZMQコマンド/内部APIでレート切替要求を伝搬（成功/失敗の応答含む）
- レート変更時のバッファサイズ/period調整

### 1.2 サポートレート

| 入力レート | ファミリ | 出力レート候補 (16x/8x/4x/2x) |
|-----------|---------|------------------------------|
| 44100 Hz | 44.1k | 705600 / 352800 / 176400 / 88200 |
| 88200 Hz | 44.1k | 705600 / 352800 / 176400 |
| 176400 Hz | 44.1k | 705600 / 352800 |
| 48000 Hz | 48k | 768000 / 384000 / 192000 / 96000 |
| 96000 Hz | 48k | 768000 / 384000 / 192000 |
| 192000 Hz | 48k | 768000 / 384000 |

## 2. シーケンス図

### 2.1 PipeWireによる入力レート検出フロー

```mermaid
sequenceDiagram
    participant PW as PipeWire
    participant Daemon as C++ Daemon
    participant AutoNeg as AutoNegotiation
    participant GPU as ConvolutionEngine
    participant ALSA as ALSA Output
    participant ZMQ as ZMQ Server (PUB)
    participant Ctrl as Python Controller

    Note over PW: 入力ストリームのレート変更検出
    PW->>Daemon: param_changed (SPA_PARAM_Format)
    Daemon->>Daemon: parse spa_audio_info_raw.rate

    alt レート変更あり
        Daemon->>AutoNeg: negotiate(inputRate, dacCap, currentOutputRate)
        AutoNeg-->>Daemon: NegotiatedConfig

        alt negotiation成功 (isValid=true)
            alt requiresReconfiguration=true (ファミリ変更)
                Daemon->>GPU: softMute(enable=true)
                Note over GPU: Fade-out (50ms)
                Daemon->>ALSA: reconfigure(newOutputRate)
                Daemon->>GPU: switchRateFamily(newFamily)
                Daemon->>GPU: softMute(enable=false)
                Note over GPU: Fade-in (50ms)
            else requiresReconfiguration=false (同一ファミリ)
                Daemon->>GPU: switchRateFamily(newFamily)
                Note over GPU: グリッチなし切り替え
            end

            Daemon->>ZMQ: publishStatus(RATE_CHANGED)
            ZMQ-->>Ctrl: {"type": "rate_changed", ...}
        else negotiation失敗 (isValid=false)
            Daemon->>ZMQ: publishStatus(RATE_ERROR)
            ZMQ-->>Ctrl: {"type": "error", "error_code": "..."}
        end
    end
```

### 2.2 ZMQコマンドによるレート切り替えフロー

```mermaid
sequenceDiagram
    participant Ctrl as Python Controller
    participant ZMQ as ZMQ Client/Server
    participant Daemon as C++ Daemon
    participant AutoNeg as AutoNegotiation
    participant GPU as ConvolutionEngine
    participant ALSA as ALSA Output

    Ctrl->>ZMQ: SWITCH_RATE {"rate_family": "48k"}
    ZMQ->>Daemon: handleSwitchRate(params)

    Daemon->>AutoNeg: negotiate(baseRate, dacCap, currentOutputRate)
    AutoNeg-->>Daemon: NegotiatedConfig

    alt negotiation成功
        alt requiresReconfiguration=true
            Daemon->>GPU: softMute(enable=true)
            Daemon->>ALSA: reconfigure(newOutputRate)
            Daemon->>GPU: switchRateFamily(newFamily)
            Daemon->>GPU: softMute(enable=false)
        else
            Daemon->>GPU: switchRateFamily(newFamily)
        end

        Daemon-->>ZMQ: {"status": "ok", "data": {...}}
        ZMQ-->>Ctrl: Success Response
    else negotiation失敗
        Daemon-->>ZMQ: {"status": "error", "error_code": "DAC_RATE_NOT_SUPPORTED", ...}
        ZMQ-->>Ctrl: Error Response
    end
```

## 3. ZMQ API仕様

### 3.1 SWITCH_RATE コマンド

**Request:**
```json
{
  "cmd": "SWITCH_RATE",
  "params": {
    "rate_family": "44k" | "48k",
    "input_rate": 44100 | 48000 | 88200 | 96000 | 176400 | 192000  // optional
  }
}
```

**Success Response:**
```json
{
  "status": "ok",
  "message": "Rate switched to 48k family",
  "data": {
    "input_rate": 48000,
    "output_rate": 768000,
    "upsample_ratio": 16,
    "rate_family": "48k",
    "reconfigured": true
  }
}
```

**Error Response:**
```json
{
  "status": "error",
  "error_code": "DAC_RATE_NOT_SUPPORTED",
  "message": "DAC does not support 768000 Hz",
  "inner_error": {
    "cpp_code": "0x2004",
    "cpp_message": "Rate 768000 not in DAC supported rates",
    "alsa_errno": -22,
    "alsa_func": "snd_pcm_hw_params_set_rate_near"
  }
}
```

### 3.2 RATE_CHANGED PUB通知

入力レートが変更された場合、PUB/SUBチャネルで通知。

```json
{
  "type": "rate_changed",
  "timestamp": 1700000000000,
  "data": {
    "previous_input_rate": 44100,
    "previous_output_rate": 705600,
    "new_input_rate": 48000,
    "new_output_rate": 768000,
    "upsample_ratio": 16,
    "rate_family": "48k",
    "reconfigured": true
  }
}
```

### 3.3 エラーコード

| ErrorCode | 値 | 説明 |
|-----------|-----|------|
| `AUDIO_INVALID_INPUT_RATE` | 0x1001 | サポートされていない入力レート |
| `DAC_RATE_NOT_SUPPORTED` | 0x2004 | DACが出力レートをサポートしていない |
| `GPU_FILTER_LOAD_FAILED` | 0x4005 | 係数ファイルのロード失敗 |

## 4. 内部API仕様

### 4.1 AutoNegotiation::negotiate()

**既存実装** (`include/auto_negotiation.h`):
```cpp
namespace AutoNegotiation {
    NegotiatedConfig negotiate(int inputRate,
                               const DacCapability::Capability& dacCap,
                               int currentOutputRate = 0);
}
```

**NegotiatedConfig構造体:**
```cpp
struct NegotiatedConfig {
    int inputRate;                              // 入力サンプルレート
    ConvolutionEngine::RateFamily inputFamily;  // RATE_44K or RATE_48K
    int outputRate;                             // ネゴシエート済み出力レート
    int upsampleRatio;                          // アップサンプル比率
    bool isValid;                               // 成功フラグ
    bool requiresReconfiguration;               // ALSA再構成が必要か
    std::string errorMessage;                   // エラーメッセージ
};
```

### 4.2 handle_rate_change()

**現在の状態:** `on_param_changed()` イベントから接続済み

**接続先:**
1. PipeWire `param_changed` イベント
2. ZMQ `SWITCH_RATE` コマンドハンドラ

**実装予定:**
```cpp
// pipewire_daemon.cpp

// PipeWireイベントから呼び出される
static bool handle_rate_change(int detected_sample_rate) {
    if (!g_upsampler || !g_upsampler->isDualRateEnabled()) {
        return false;
    }

    auto dacCap = DacCapability::scan(g_alsa_device);
    auto config = AutoNegotiation::negotiate(
        detected_sample_rate, dacCap, g_current_output_rate.load());

    if (!config.isValid) {
        // PUB通知: エラー
        publishRateError(config.errorMessage);
        return false;
    }

    if (config.requiresReconfiguration) {
        // ファミリ変更: Soft Mute + ALSA再構成
        g_upsampler->softMute(true);
        reconfigureAlsa(config.outputRate);
    }

    // 係数切り替え
    g_upsampler->switchRateFamily(config.inputFamily);
    g_set_rate_family(config.inputFamily);
    g_current_input_rate.store(config.inputRate);
    g_current_output_rate.store(config.outputRate);

    if (config.requiresReconfiguration) {
        g_upsampler->softMute(false);
    }

    // PUB通知: 成功
    publishRateChanged(config);
    return true;
}
```

## 5. バッファ/Period調整

### 5.1 レート変更時の考慮事項

| パラメータ | 44.1k系 (705.6kHz出力) | 48k系 (768kHz出力) | 備考 |
|-----------|------------------------|-------------------|------|
| リングバッファ容量 | 768000 × 2 samples | 768000 × 2 samples | 両方対応のため最大値で統一 |
| ALSA period_size | 256-1024 | 256-1024 | DACに依存 |
| ALSA buffer_size | period × 4 | period × 4 | 安全マージン |

### 5.2 ALSA再構成手順

```cpp
bool reconfigureAlsa(int newRate) {
    // 1. 現在のストリームを停止
    snd_pcm_drop(pcm_handle);

    // 2. 新しいレートでhw_params設定
    snd_pcm_hw_params_t* hw_params;
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(pcm_handle, hw_params);

    // フォーマット設定 (S32_LE)
    snd_pcm_hw_params_set_format(pcm_handle, hw_params, SND_PCM_FORMAT_S32_LE);

    // レート設定
    unsigned int rate = newRate;
    int dir = 0;
    int err = snd_pcm_hw_params_set_rate_near(pcm_handle, hw_params, &rate, &dir);
    if (err < 0 || rate != newRate) {
        return false;  // DAC_RATE_NOT_SUPPORTED
    }

    // 適用
    snd_pcm_hw_params(pcm_handle, hw_params);

    // 3. ストリーム再開
    snd_pcm_prepare(pcm_handle);
    return true;
}
```

### 5.3 エラー時の挙動

| エラー | 原因 | 復旧アクション |
|--------|------|----------------|
| `DAC_RATE_NOT_SUPPORTED` | DACが新レートをサポートしない | 前のレートを維持、エラー通知 |
| `AUDIO_BUFFER_OVERFLOW` | バッファ調整失敗 | Soft Reset実行 |
| `AUDIO_XRUN_DETECTED` | アンダーラン発生 | バッファ再充填待ち |

## 6. PipeWire param_changed イベント

### 6.1 イベントハンドラ登録

```cpp
static const struct pw_stream_events input_stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
    .param_changed = on_param_changed,  // 追加
    .process = on_input_process,
};

static void on_param_changed(void* userdata, uint32_t id, const struct spa_pod* param) {
    if (id != SPA_PARAM_Format || param == nullptr) {
        return;
    }

    struct spa_audio_info_raw info;
    if (spa_format_audio_raw_parse(param, &info) < 0) {
        return;
    }

    int detected_rate = info.rate;
    if (detected_rate != g_current_input_rate.load()) {
        // 別スレッドで処理（リアルタイムスレッドをブロックしない）
        // または、フラグを立てて main loop で処理
        g_pending_rate_change.store(detected_rate);
    }
}
```

### 6.2 非リアルタイム処理

レート変更はALSA再構成を伴うため、オーディオコールバック内で直接実行しない。

```cpp
// Main loop内で処理
while (g_running) {
    int pending_rate = g_pending_rate_change.exchange(0);
    if (pending_rate > 0) {
        handle_rate_change(pending_rate);
    }

    // pw_loop_iterate を使用（pw_main_loop_runの代わり）
    pw_loop_iterate(pw_main_loop_get_loop(data.loop), 10);  // 10ms timeout
}
```

## 7. テスト計画

### 7.1 ユニットテスト

| テストケース | 入力 | 期待出力 |
|-------------|------|---------|
| 44.1kHz入力 → 705.6kHz出力 | `negotiate(44100, dacCap)` | `{outputRate: 705600, ratio: 16}` |
| 48kHz入力 → 768kHz出力 | `negotiate(48000, dacCap)` | `{outputRate: 768000, ratio: 16}` |
| 96kHz入力 → 768kHz出力 | `negotiate(96000, dacCap)` | `{outputRate: 768000, ratio: 8}` |
| 44.1k→48k切り替え | `negotiate(48000, ...)` (from 44.1k) | `{requiresReconfiguration: true}` |
| 同一ファミリ切り替え | `negotiate(88200, ...)` (from 44.1k) | `{requiresReconfiguration: false}` |
| 未サポートレート | `negotiate(22050, dacCap)` | `{isValid: false}` |
| DACレート不足 | `negotiate(44100, dacCap_max384k)` | `{outputRate: 352800}` |

### 7.2 統合テスト (手動)

1. **PipeWireレート検出:**
   - 44.1kHzソース再生 → ログで705.6kHz出力確認
   - 48kHzソースに切り替え → ログで768kHz出力確認

2. **ZMQコマンド:**
   - `SWITCH_RATE {"rate_family": "48k"}` 送信
   - 成功レスポンス確認
   - ステータスで`currentRateFamily: "48k"`確認

## 8. 実装チェックリスト

- [x] `on_param_changed()` イベントハンドラ追加
- [x] `g_pending_rate_change` atomic変数追加
- [x] `handle_rate_change()` の `[[maybe_unused]]` 削除、接続
- [ ] `SWITCH_RATE` ZMQコマンドハンドラ実装 (後続Issue)
- [ ] `publishRateChanged()` PUB通知実装 (後続Issue)
- [x] ユニットテスト追加 (`test_auto_negotiation.cpp`)
- [ ] 統合テスト実行

## 9. 現在の実装制限

> **重要**: 以下の制限は設計上の制約であり、後続Issueで対応予定。

### 9.1 サポートされる入力レート

| 入力レート | サポート状態 | 備考 |
|-----------|-------------|------|
| 44100 Hz | ✅ 完全対応 | 16x upsampling → 705.6kHz |
| 48000 Hz | ✅ 完全対応 | 16x upsampling → 768kHz |
| 88200 Hz | ⚠️ 検出のみ | 8x係数未実装、WARNING出力 |
| 96000 Hz | ⚠️ 検出のみ | 8x係数未実装、WARNING出力 |
| 176400 Hz | ⚠️ 検出のみ | 4x係数未実装、WARNING出力 |
| 192000 Hz | ⚠️ 検出のみ | 4x係数未実装、WARNING出力 |
| 352800 Hz | ⚠️ 検出のみ | 2x係数未実装、WARNING出力 |
| 384000 Hz | ⚠️ 検出のみ | 2x係数未実装、WARNING出力 |

### 9.2 制限の根本原因

現在のシステムには以下の制約がある：

1. **係数ファイルが16x専用**: `filter_44k_16x_*.bin` / `filter_48k_16x_*.bin` のみ存在
2. **GPUUpsamplerの固定比率**: `loadCoefficients()` は16xアップサンプリングを前提に実装
3. **PipeWire/ALSAストリーム再構成未実装**: `reconfigureAlsa()` はスケルトンのみ

### 9.3 ハイレゾ入力時の挙動

88.2kHz以上の入力が検出された場合：

1. `handle_rate_change()` は正しいファミリを判定
2. **警告ログを出力** (`[Rate] WARNING: Hi-res input detected...`)
3. 実際の入力レートを `g_current_input_rate` に記録
4. 係数切り替えは実行されるが、**アップサンプル比率は16xのまま**

**結果**:
- 音声出力は動作するが、ピッチ・再生速度が正しくない可能性あり
- ユーザーはログを確認して、44.1kHz/48kHzソースを使用すべき

### 9.4 後続Issueで対応予定

| 機能 | 関連Issue | 優先度 |
|------|----------|--------|
| 8x/4x/2x係数ファイル生成 | TBD | 高 |
| GPUUpsampler動的比率切り替え | TBD | 高 |
| ALSA再構成実装 | TBD | 中 |
| PipeWireストリーム再構成 | TBD | 中 |
