# Filter Generation Batch

複数フィルター係数の一括生成とメタデータ検証を自動化します。

## Description

このSkillは、最小位相または線形位相のFIRフィルター係数を一括生成し、メタデータ検証と周波数応答プロットを自動作成します。44.1kHz系と48kHz系の8つのマルチレート設定を効率的に生成します。

## Trigger Words

- `generate filters`
- `batch filters`
- `create coefficients`
- `フィルター生成`
- `係数作成`

## Requirements

- `uv` がインストールされていること
- Python dependencies: scipy, numpy, matplotlib
- 十分なディスクスペース（約20MB per 640k-tap filter）
- 実行ディレクトリ: プロジェクトルート

## Parameters

### 必須パラメータ

- **phase_type**: 位相タイプ
  - `minimum`: 最小位相（推奨、リスニング用途）
  - `linear`: 線形位相（ミキシング・マスタリング用途）
  - 例: `generate filters minimum`

### オプションパラメータ

- **rate_family**: サンプルレートファミリー
  - `44k`: 44.1kHz系のみ（1x, 2x, 4x, 16x）
  - `48k`: 48kHz系のみ（1x, 2x, 4x, 16x）
  - `all`: 両方（デフォルト、合計8個）
  - 例: `generate filters minimum 44k`

- **taps**: タップ数
  - デフォルト: `640000` (640k)
  - 例: `generate filters minimum all 640000` (640k-tap)

## Execution Steps

```bash
# 1. 環境確認
uv sync  # Python依存関係インストール

# 2. 生成対象決定
# rate_family=all の場合、8個のフィルター生成:
# - 44.1kHz系: 1x, 2x, 4x, 16x
# - 48kHz系: 1x, 2x, 4x, 16x

# 3. 並列生成（CPU許す限り）
# phase_type=minimum の場合:
uv run python scripts/filters/generate_minimum_phase.py --generate-all

# phase_type=linear の場合:
uv run python scripts/filters/generate_linear_phase.py --generate-all

# 4. メタデータ検証
# data/coefficients/*.json を解析:
# - DC gain 正規化確認
# - Stopband attenuation チェック
# - Peak coefficient ≤1.0 確認

# 5. プロット生成
# plots/analysis/ に以下を生成:
# - 周波数応答プロット
# - インパルス応答プロット
# - 位相応答プロット

# 6. サマリーレポート生成
```

## Expected Output

### 成功時（8個生成）:
```markdown
# フィルター係数生成完了

## 生成されたフィルター（8個）

### 44.1kHz系（Minimum Phase, 640k-tap）
- ✅ filter_44k_1x_640k_min_phase.bin (2.44 MB)
  - DC Gain: 1.0 (normalized)
  - Stopband: -160.3 dB
  - Peak Coeff: 0.998

- ✅ filter_44k_2x_640k_min_phase.bin (2.44 MB)
  - DC Gain: 2.0 (normalized)
  - Stopband: -160.1 dB
  - Peak Coeff: 1.000

- ✅ filter_44k_4x_640k_min_phase.bin (2.44 MB)
  - DC Gain: 4.0 (normalized)
  - Stopband: -160.4 dB
  - Peak Coeff: 0.999

- ✅ filter_44k_16x_640k_min_phase.bin (2.44 MB)
  - DC Gain: 16.0 (normalized)
  - Stopband: -160.2 dB
  - Peak Coeff: 1.000

### 48kHz系（Minimum Phase, 640k-tap）
- ✅ filter_48k_1x_640k_min_phase.bin (2.44 MB)
  - DC Gain: 1.0 (normalized)
  - Stopband: -160.2 dB
  - Peak Coeff: 0.997

- ✅ filter_48k_2x_640k_min_phase.bin (2.44 MB)
  - DC Gain: 2.0 (normalized)
  - Stopband: -160.3 dB
  - Peak Coeff: 0.999

- ✅ filter_48k_4x_640k_min_phase.bin (2.44 MB)
  - DC Gain: 4.0 (normalized)
  - Stopband: -160.1 dB
  - Peak Coeff: 1.000

- ✅ filter_48k_16x_640k_min_phase.bin (2.44 MB)
  - DC Gain: 16.0 (normalized)
  - Stopband: -160.4 dB
  - Peak Coeff: 0.998

## 検証結果
- ✅ DC Gain: 全て正規化済み
- ✅ Stopband Attenuation: -160.2 dB（平均）
- ✅ Peak Coefficient: 全て ≤1.0（クリッピング防止）
- ✅ メタデータ整合性: OK

## 生成されたプロット
- plots/analysis/44k_16x_frequency_response.png
- plots/analysis/44k_16x_impulse_response.png
- plots/analysis/48k_16x_frequency_response.png
- plots/analysis/48k_16x_impulse_response.png

## ファイルサイズ
- 合計: 19.5 MB（8個 × 2.44 MB）

## 実行時間
- 44.1kHz系: 9分12秒
- 48kHz系: 9分22秒
- 合計: 18分34秒
```

### 一部失敗時:
```markdown
# フィルター係数生成完了（一部失敗）

## 生成成功（7個）
[上記と同様のリスト]

## 生成失敗（1個）
- ❌ filter_48k_16x_640k_min_phase.bin
  - エラー: MemoryError: Unable to allocate array
  - 原因: メモリ不足（必要: ~8GB、利用可能: 4GB）
  - 推奨: タップ数を削減（--taps 320000）または RAM増設

## 推奨アクション
- [ ] メモリを増設
- [ ] または `generate filters minimum 48k --taps 320000` で再試行
- [ ] swap領域を確認: `free -h`
```

## Error Handling

このSkillはベストエフォート戦略を採用しています：

1. **1個失敗時**:
   - 残りのフィルター生成は継続
   - 失敗したフィルターのみ再試行
   - エラー原因を特定（メモリ、CPU、ディスク）

2. **メモリ不足時**:
   - タップ数削減を提案
   - スワップ領域の確認方法を提示
   - システムリソース状況を表示

3. **検証失敗時**:
   - 警告表示（処理は継続）
   - 手動確認を促す
   - 期待値との差分を表示

4. **長時間実行時**:
   - 進捗状況を定期的に表示
   - 推定残り時間を計算
   - 中断可能（Ctrl+C）

## Best Practices

### 位相タイプの選択

| 位相タイプ | 用途 | 特性 |
|-----------|------|------|
| **Minimum Phase** | 音楽再生、リスニング | プリリンギングなし、レイテンシゼロ |
| **Linear Phase** | ミキシング、マスタリング | 対称インパルス応答、高レイテンシ |

### タップ数の選択

| タップ数 | Stopband | ファイルサイズ | 生成時間（概算） |
|---------|---------|--------------|----------------|
| 320k | -140 dB | 1.2 MB | ~5分 |
| 640k | -160 dB | 2.4 MB | ~18分（推奨） |
| 2M | -180 dB | 7.6 MB | ~60分 |

### Kaiser β値

- デフォルト: β≈28（32bit Float実装の量子ノイズ限界に最適）
- 手動調整は非推奨

## Related Skills

- `build-and-test`: フィルター生成後、C++側でテスト推奨
- `worktree-pr-workflow`: フィルター追加後のPR作成

## Implementation Notes

このSkillは以下の既存スクリプトを活用します：

### スクリプト
- `scripts/filters/generate_minimum_phase.py`: 最小位相フィルター生成
- `scripts/filters/generate_linear_phase.py`: 線形位相フィルター生成
- `scripts/analysis/verify_frequency_response.py`: 周波数応答検証（オプション）

### 生成ファイル構造
```
data/coefficients/
├── filter_44k_1x_640k_min_phase.bin
├── filter_44k_1x_640k_min_phase.json  # メタデータ
├── filter_44k_2x_640k_min_phase.bin
├── filter_44k_2x_640k_min_phase.json
...
```

### メタデータJSON形式
```json
{
  "taps": 640000,
  "phase_type": "minimum",
  "input_rate": 44100,
  "upsample_ratio": 16,
  "output_rate": 705600,
  "dc_gain": 16.0,
  "stopband_attenuation_db": -160.2,
  "peak_coefficient": 1.000,
  "kaiser_beta": 28.0,
  "generated_at": "2025-12-06T10:30:45Z"
}
```

## Technical Constraints

### Minimum Phase要件
- プリリンギングなし（t≥0 に集中）
- 因果的フィルタ（リアルタイム処理可能）
- **音楽再生に最適**: トランジェント保持

### Stopband Attenuation目標
- 160dB: 24-bit audio（144dB dynamic range）に十分
- Kaiser β≈28: 32-bit Float実装の最適値

### DC Gain正規化
- Zero-stuffing upsamplingはDCを1/L に減少
- Filter DC gain = L で補償
- Peak coefficient ≤1.0 でクリッピング防止

## Automation Level

**完全自動実行**: 全フィルター生成、検証、プロット作成を自動化します。
ユーザー入力は phase_type のみ必要です。
