# オーディオデータフロー

## 概要

Totton Audioのオーディオ処理パイプラインは、USB入力からDAC出力まで一貫したデータフローで構成されています。

---

## 全体フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Audio Data Flow                                    │
│                                                                              │
│  Raspberry Pi (Sender)                                                        │
│      │                                                                       │
│      │ ALSA Capture → GStreamer RTP (rtpbin, RTCP, audioresample q=10)       │
│      ▼                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     Jetson Input (RTP → Loopback)                      │ │
│  │  ┌──────────────┐    ┌──────────────┐                                  │ │
│  │  │ RTP Receiver │───>│ ALSA Playback│ (hw:Loopback,0,0)                │ │
│  │  └──────────────┘    └──────────────┘                                  │ │
│  └─────────────────────────────────────────────────────────┼──────────────┘ │
│                                                             │  (capture)     │
│                                                             ▼                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        GPU Processing Stage                             │ │
│  │                                                                         │ │
│  │  ┌────────────────────────────────────────────────────────────────┐    │ │
│  │  │                    Overlap-Save Convolution                     │    │ │
│  │  │                                                                  │    │ │
│  │  │   Input Block     FFT        Multiply      IFFT      Output     │    │ │
│  │  │  ┌─────────┐   ┌─────┐    ┌─────────┐   ┌─────┐   ┌─────────┐  │    │ │
│  │  │  │ 4096    │──>│cuFFT│───>│  H(f)   │──>│cuFFT│──>│ 4096    │  │    │ │
│  │  │  │ samples │   │ R2C │    │  ×      │   │ C2R │   │ samples │  │    │ │
│  │  │  └─────────┘   └─────┘    └────┬────┘   └─────┘   └─────────┘  │    │ │
│  │  │                                 │                               │    │ │
│  │  │                    ┌────────────┴────────────┐                  │    │ │
│  │  │                    │   Filter Coefficients   │                  │    │ │
│  │  │                    │   640k-tap (pre-FFT'd)  │                  │    │ │
│  │  │                    │   44k/48k × min/linear  │                  │    │ │
│  │  │                    └─────────────────────────┘                  │    │ │
│  │  └────────────────────────────────────────────────────────────────┘    │ │
│  │                                                                         │ │
│  │  ┌──────────────┐    ┌──────────────┐                                  │ │
│  │  │ EQ Apply     │───>│ Soft Mute    │ (レート切替時クロスフェード)      │ │
│  │  │ (Parametric) │    │ Fade In/Out  │                                  │ │
│  │  └──────────────┘    └──────────────┘                                  │ │
│  └─────────────────────────────────────────────────────────┼──────────────┘ │
│                                                             │                │
│                                                             ▼                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        Output Stage                                     │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │ │
│  │  │ Ring Buffer  │───>│ ALSA Playback│───>│ External DAC │              │ │
│  │  │ (Output)     │    │ (hw:AUDIO)   │    │ USB Type-A   │              │ │
│  │  └──────────────┘    └──────────────┘    └──────────────┘              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                              │               │
│                                                              ▼               │
│                                                         Headphones          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 処理ステージ詳細

### 1. 入力ステージ

#### RTP受信（Jetson）

- `web/services/rtp_input.py` が `rtpbin` ベースのパイプラインを生成し、ALSA Loopback playback (`hw:Loopback,0,0`) へ出力。
- デフォルトポート: RTP 46000 / RTCP in 46001 / RTCP back 46002。
- `audioresample quality=10` でクロックドリフトを吸収し、RTCP で送信側クロックに同期。
- 例: `gst-launch-1.0 -e rtpbin name=rtpbin ntp-sync=true buffer-mode=sync latency=100 udpsrc port=46000 caps=... ! rtpbin.recv_rtp_sink_0 rtpbin.recv_rtp_src_0 ! rtpL24depay ! audioconvert ! audioresample quality=10 ! audio/x-raw,format=S24LE,rate=44100,channels=2 ! alsasink device=hw:Loopback,0,0 sync=true provide-clock=true udpsrc port=46001 ! rtpbin.recv_rtcp_sink_0 rtpbin.send_rtcp_src_0 ! udpsink host=<pi-host> port=46002 sync=false async=false`

#### ALSA Capture（GPU入口）

```cpp
// RTPで再生した Loopback をキャプチャし GPU パイプラインへ投入
snd_pcm_open(&capture_handle, "hw:Loopback,1,0", SND_PCM_STREAM_CAPTURE, 0);
snd_pcm_set_params(capture_handle,
    SND_PCM_FORMAT_S32_LE,
    SND_PCM_ACCESS_RW_INTERLEAVED,
    2,                          // Stereo
    sample_rate,                // 44100 or 48000
    1,                          // Allow resampling (soft)
    latency_us);
```

#### レート判定

RTP ペイロードと GStreamer caps から受信レート/フォーマットを決定し、GPU 側フィルタを選択：

| 入力レート | 系譜 | 使用フィルタ |
|-----------|------|-------------|
| 44100 Hz | 44.1k系 | filter_44k_*.bin |
| 48000 Hz | 48k系 | filter_48k_*.bin |
| 88200 Hz | 44.1k系 | filter_44k_*.bin (ダウンサンプル後) |
| 96000 Hz | 48k系 | filter_48k_*.bin (ダウンサンプル後) |

#### Ring Buffer (入力)

```cpp
// Lock-free キュー (moodycamel::ReaderWriterQueue)
struct AudioBlock {
    float samples[BLOCK_SIZE * 2];  // Interleaved L/R
    uint32_t sample_rate;
    uint32_t frame_count;
};

ReaderWriterQueue<AudioBlock> input_queue(16);  // 16ブロック分バッファ
```

---

### 2. GPU処理ステージ

#### Overlap-Save畳み込み

2Mタップフィルタを効率的に処理するための分割畳み込み：

```
Input Block Size:  N = 4096 samples
Filter Length:     L = 2,000,000 taps
Partitions:        L / N = 488 partitions
FFT Size:          2N = 8192 points
Overlap:           N samples (50%)
```

**処理フロー**:

```
1. 入力ブロック (N samples) を取得
2. 前回のN samplesと連結 (2N samples)
3. FFT (cuFFT R2C)
4. 各パーティションの周波数応答 H_k(f) と乗算
5. 累積加算
6. IFFT (cuFFT C2R)
7. 後半N samplesを出力
```

#### フィルタ係数

```cpp
// GPU メモリに常駐
struct FilterCoefficients {
    cufftComplex* freq_domain;  // Pre-computed FFT
    size_t num_partitions;       // 488
    size_t fft_size;             // 8192
};

// 4種類プリロード
FilterCoefficients filters[4] = {
    { ... },  // 44k minimum phase
    { ... },  // 44k linear phase
    { ... },  // 48k minimum phase
    { ... },  // 48k linear phase
};
```

#### EQ適用

```cpp
// Parametric EQ (時間領域)
// または周波数領域での乗算（将来最適化）
void apply_eq(float* samples, size_t count, const EQProfile& eq) {
    for (const auto& band : eq.bands) {
        apply_biquad(samples, count, band);
    }
}
```

#### Soft Mute

レート切り替え時のポップノイズ防止：

```cpp
// クロスフェード（5ms）
constexpr float FADE_TIME_SEC = 0.005f;

void soft_mute_transition(float* output, size_t count, float fade_ratio) {
    for (size_t i = 0; i < count; i++) {
        float env = calculate_envelope(i, count, fade_ratio);
        output[i * 2]     *= env;  // L
        output[i * 2 + 1] *= env;  // R
    }
}
```

---

### 3. 出力ステージ

#### Ring Buffer (出力)

```cpp
// 出力レートに合わせたバッファリング
// 705.6kHz出力時: 16倍のサンプル数
ReaderWriterQueue<AudioBlock> output_queue(32);  // 大きめのバッファ
```

#### ALSA Playback

```cpp
// 外部DAC への出力
snd_pcm_open(&playback_handle, "hw:AUDIO", SND_PCM_STREAM_PLAYBACK, 0);
snd_pcm_set_params(playback_handle,
    SND_PCM_FORMAT_S32_LE,
    SND_PCM_ACCESS_RW_INTERLEAVED,
    2,                          // Stereo
    output_rate,                // 705600 or 768000
    0,                          // No resampling
    latency_us);
```

---

## レイテンシ分析

### 最小位相フィルタ使用時

| ステージ | レイテンシ |
|---------|-----------|
| USB入力バッファ | ~5 ms |
| 入力Ring Buffer | ~10 ms |
| GPU処理 (1ブロック) | ~1 ms |
| 出力Ring Buffer | ~10 ms |
| ALSA出力バッファ | ~5 ms |
| **合計** | **~31 ms** |

### ハイブリッドフィルタ使用時

| ステージ | 追加レイテンシ |
|---------|---------------|
| ハイブリッド遅延整列 | ~10 ms（全帯域をクロスオーバー周波数100Hzの1周期位置に揃える） |
| **合計** | **~41 ms** |

> 参考: 旧線形位相フィルタは 2Mタップ @705.6kHz で約1.4秒の遅延が発生していました。

---

## エラー処理

### XRUN発生時

```cpp
void handle_xrun() {
    // 1. Soft Mute 開始
    soft_mute.fade_out();

    // 2. バッファクリア
    input_queue.clear();
    output_queue.clear();

    // 3. ALSA デバイス再準備
    snd_pcm_prepare(capture_handle);
    snd_pcm_prepare(playback_handle);

    // 4. Soft Mute 解除
    soft_mute.fade_in();
}
```

### GPU負荷超過時

```cpp
// Fallback Manager による自動切り替え
if (gpu_utilization > 80.0f && consecutive_high_count >= 3) {
    // フォールバックモードへ移行
    // (タップ数削減 or バイパス)
    fallback_manager.enter_fallback_mode();
}
```

---

## 性能指標

### RTX 2070S (PC開発環境)

| 指標 | 値 |
|------|-----|
| 処理速度 | ~28x realtime |
| GPU使用率 | ~15% |
| メモリ帯域使用率 | ~20% |

### Jetson Orin Nano (本番環境・推定)

| 指標 | 推定値 | 備考 |
|------|--------|------|
| 処理速度 | ~3-5x realtime | メモリ帯域制約 |
| GPU使用率 | ~50-70% | 要実機検証 |
| メモリ帯域使用率 | ~80% | ボトルネック |

**注意**: Jetsonでの性能は実機検証が必要です。メモリ帯域がボトルネックになる可能性があります。

---

## 関連ドキュメント

- [system-overview.md](./system-overview.md) - システム全体構成
- [../quality/performance-benchmark.md](../quality/performance-benchmark.md) - 性能ベンチマーク
- [../reliability/error-recovery.md](../reliability/error-recovery.md) - エラー回復
