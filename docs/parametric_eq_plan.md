# パラメトリックEQ機能 実装計画

## 概要

GPU Audio Upsamplerにパラメトリックイコライザー機能を追加する。
AutoEq/Equalizer APO形式のEQプロファイルを読み込み、オーバーサンプリングフィルタと合成して適用する。

## 設計方針

### EQ適用順序: オーバーサンプリング前 (Before)

**理由（音質優先）:**
1. **エイリアシング回避**: アンチエイリアスフィルタが最終段で確実に高周波を抑制
2. **計算効率**: 44.1kHzで処理（705.6kHzの1/16の計算量）
3. **実装シンプル**: EQ応答とFIRフィルタを周波数領域で合成 `H_combined = H_eq × H_fir`

### 処理フロー
```
[Audio Input @ 44.1kHz]
       ↓
[Parametric EQ (周波数領域合成)]  ← EQ応答を131k-tap FIRのFFTと乗算
       ↓
[Zero-padding (16x)]
       ↓
[FFT Convolution with H_combined]
       ↓
[Audio Output @ 705.6kHz]
```

## EQファイルフォーマット

**対応形式:** AutoEq/Equalizer APO形式

```
Preamp: -10.5db
Filter 1: ON PK Fc 140.3 Hz Gain -2 dB Q 0.81
Filter 2: ON PK Fc 556.4 Hz Gain 0.9 dB Q 3.26
...
Filter 17: ON LS Fc 100 Hz Gain 3 dB Q 1.2
Filter 18: ON HS Fc 8190.5 Hz Gain 0.1 dB Q 2.93
```

| フィールド | 説明 | 例 |
|------------|------|-----|
| `Preamp` | 全体ゲイン調整 (dB) | `-10.5db` |
| `ON/OFF` | フィルタ有効/無効 | `ON` |
| フィルタタイプ | `PK`=ピーキング, `HS`=ハイシェルフ, `LS`=ローシェルフ | `PK` |
| `Fc` | 中心周波数 (Hz) | `140.3 Hz` |
| `Gain` | ゲイン (dB) | `-2 dB` |
| `Q` | Q値 (帯域幅) | `0.81` |

## 実装タスク

### A. EQパーサー
- **ファイル:** `src/eq_parser.cpp`, `include/eq_parser.h`
- AutoEq/Equalizer APO形式のパース
- 対応フィルタ: PK(ピーキング), LS(ローシェルフ), HS(ハイシェルフ)
- Preamp値の抽出

### B. EQ→FIR変換
- **ファイル:** `src/eq_to_fir.cpp`, `include/eq_to_fir.h`
- パラメトリックEQからIIRバイクアッド係数を計算
- IIR応答を周波数領域で評価 → FIRインパルス応答へ変換
- 既存131k-tap FIRとの周波数領域合成

### C. Convolution Engine拡張
- **ファイル:** `src/convolution_engine.cu`, `include/convolution_engine.h`
- `setEqResponse()` メソッド追加
- `d_filterFFT_` をEQ適用済みバージョンに更新する機能
- EQ無効時のオリジナルフィルタ復元

### D. Config拡張
- **ファイル:** `include/config_loader.h`, `src/config_loader.cpp`
```cpp
struct AppConfig {
    // 既存フィールド...
    std::string eqProfilePath = "";  // 空 = EQ無効
    bool eqEnabled = false;
};
```

### E. Web API
- **ファイル:** `web/main.py`

| エンドポイント | メソッド | 機能 |
|---------------|---------|------|
| `/eq/profiles` | GET | プロファイル一覧取得 |
| `/eq/import` | POST | ファイルアップロード |
| `/eq/activate/{name}` | POST | プロファイル適用 |
| `/eq/profiles/{name}` | DELETE | プロファイル削除 |
| `/eq/active` | GET | 現在のアクティブ状態 |

### F. Web UI
- **ファイル:** `web/static/index.html` または埋め込みHTML
- EQプロファイル選択ドロップダウン
- ファイルアップロードボタン
- ON/OFFトグル

## ファイル構成

```
data/EQ/
├── Sample_EQ.txt          # 既存サンプル
├── profiles.json          # プロファイル管理メタデータ
└── *.txt                  # ユーザーアップロード済みプロファイル

include/
├── eq_parser.h            # 新規
└── eq_to_fir.h            # 新規

src/
├── eq_parser.cpp          # 新規
├── eq_to_fir.cpp          # 新規
└── convolution_engine.cu  # 拡張
```

## システムフロー

```
[Web UI] → EQファイルアップロード
    ↓
[API] → data/EQ/ に保存 + profiles.json 更新
    ↓
[Activate] → config.json 更新 + SIGHUP送信
    ↓
[Daemon再起動] → EQパース → FIR合成 → GPU転送
    ↓
[リアルタイム処理] 合成済みフィルタで畳み込み
```

## 技術詳細

### バイクアッド係数計算

ピーキングフィルタ (PK):
```
A  = sqrt(10^(gain/20))
w0 = 2 * pi * Fc / Fs
alpha = sin(w0) / (2 * Q)

b0 = 1 + alpha * A
b1 = -2 * cos(w0)
b2 = 1 - alpha * A
a0 = 1 + alpha / A
a1 = -2 * cos(w0)
a2 = 1 - alpha / A
```

### 周波数応答の合成

1. 各バイクアッドフィルタの周波数応答を計算: `H_biquad(f)`
2. 全バンドを乗算: `H_eq(f) = H_band1(f) × H_band2(f) × ... × H_bandN(f) × preamp`
3. FIRフィルタと合成: `H_combined(f) = H_eq(f) × H_fir(f)`
4. GPU転送して畳み込みに使用

## ブランチ

- 作業ブランチ: `feature/parametric-eq`
- Worktree: `../gpu_os-eq`
