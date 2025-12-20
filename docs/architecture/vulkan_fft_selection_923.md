## Vulkan: FFTライブラリ選定（VkFFT候補）と要件/ライセンス確認（Issue #923）

### 結論（採用方針）
- **第一候補: VkFFT を採用する**
  - 目的（EPIC #921）で必要な **1D FFT（R2C/C2R）** を Vulkan compute 上で提供できる
  - 「CUDA実装の構造に近い形」での単純置換に向く（plan/execute のモデルが近い）
  - 多バックエンド対応（Vulkan以外も視野）で、将来の検証/移植リスクを下げられる

参照: `https://github.com/DTolm/VkFFT`

### ライセンス（配布/商用/同梱の可否）
- **MIT License**
  - 商用利用・同梱配布は基本的に可能
  - **必要条件**: 著作権表示とライセンス文を同梱（NOTICE/THIRD_PARTY_NOTICES等でも可）

※実装取り込み時の運用案:
- `third_party/vkfft/` にソースを配置し、`LICENSE` を必ず含める
- もしくは submodule で管理し、配布物にライセンス同梱を行う

### 要件（Pi5 / Docker / 将来Androidを見据えた整理）
Issue #923 の受け入れ条件に合わせ、最小限の要件を列挙する。

#### 必須（EPIC #921 の「まず動く」段階）
- **Vulkan compute が動作すること**
  - compute queue が取得できる
  - storage buffer を使える（FFTの入出力・作業領域）
- **メモリ要件**
  - device local のバッファ確保（FFT作業領域）
  - host-visible の staging（Docker/ファイルI/O からの入力/出力）
- **実装要件**
  - 1D R2C / C2R を実行できる
  - バッチ実行（将来のパーティション/複数ch）に拡張できる

#### 推奨（性能/拡張の余地）
- **非同期実行 + 明示同期**
  - fence / timeline semaphore で in-flight を管理できる（ストリーミング化で必要）
- **サブグループ（subgroup）最適化**
  - 利用できれば性能面の伸びが期待できる（ただし必須にはしない）
- **FP16/INT8 系**
  - 初期は FP32 前提で進め、後から必要なら検討

### Vulkan機能（箇条書き）
「必須」ではなく「期待する前提」として整理（実装段階で `vkGetPhysicalDeviceFeatures*` により確認する）。

- **必須（初期）**
  - compute pipeline / descriptor set / push constants（基本機能）
  - storage buffer（`VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`）
  - host-visible memory（staging）
  - device-local memory（作業領域）
- **任意（将来の最適化）**
  - timeline semaphore（`VK_KHR_timeline_semaphore` または Vulkan 1.2 相当）
  - subgroup（`VkPhysicalDeviceSubgroupProperties`）
  - buffer device address（最適化で必要になれば）

### 次のアクション（EPIC #921）
- Pi5 上で **VkFFTの最小サンプル（1D R2C/C2R）** を Docker 環境で起動し、FFT実行が通ることを確認する。
- その結果を踏まえ、Issue #926 の backend 抽象（alloc/copy/fft/mul+scale/sync）に沿って VulkanBackend 実装へ進む。
