#!/usr/bin/env python3
"""
最小位相フィルタのインパルス応答詳細調査
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# バイナリファイルから係数を読み込む
coeffs_path = Path('data/coefficients/filter_131k_min_phase.bin')
h = np.fromfile(coeffs_path, dtype=np.float32)

print(f"フィルタ係数: {len(h)} taps")
print(f"\n先頭100サンプルの値:")
print("=" * 60)

# 先頭100サンプルの統計
first_100 = h[:100]
peak_idx = np.argmax(np.abs(h[:1000]))  # 最初の1000サンプル内でピーク検索
peak_value = h[peak_idx]

print(f"ピーク位置: サンプル {peak_idx}")
print(f"ピーク値: {peak_value:.6f}")
print(f"\nサンプル0の値: {h[0]:.6f}")
print(f"サンプル1の値: {h[1]:.6f}")
print(f"サンプル35の値: {h[35]:.6f}")

# 先頭10サンプルの詳細
print(f"\n先頭10サンプルの詳細:")
for i in range(10):
    print(f"  h[{i}] = {h[i]:.8f}")

# 絶対値の最大値を持つサンプルを探す
abs_max_idx = np.argmax(np.abs(first_100))
print(f"\n先頭100サンプル内の絶対値最大: サンプル{abs_max_idx}, 値={h[abs_max_idx]:.6f}")

# 先頭100サンプルのエネルギー
energy_100 = np.sum(first_100**2)
energy_total = np.sum(h**2)
energy_ratio = (energy_100 / energy_total) * 100

print(f"\n先頭100サンプルのエネルギー比率: {energy_ratio:.2f}%")

# プロット
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 先頭500サンプルのプロット
axes[0].plot(h[:500], linewidth=0.8)
axes[0].axvline(peak_idx, color='r', linestyle='--', alpha=0.7, label=f'Peak at {peak_idx}')
axes[0].set_title('Minimum Phase Impulse Response (First 500 samples)', fontsize=12)
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 先頭100サンプルの拡大
axes[1].plot(h[:100], linewidth=1.2, marker='o', markersize=3)
axes[1].axvline(peak_idx, color='r', linestyle='--', alpha=0.7, label=f'Peak at {peak_idx}')
axes[1].set_title('Minimum Phase Impulse Response (First 100 samples - Zoom)', fontsize=12)
axes[1].set_xlabel('Sample')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
output_path = Path('plots/analysis/impulse_detail.png')
plt.savefig(output_path, dpi=150)
print(f"\nプロット保存: {output_path}")
plt.close()
