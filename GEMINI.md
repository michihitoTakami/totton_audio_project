# GEMINI.md

## Language
Think in English and answer in Japanese.

## Project Overview

**Magic Box Project - 魔法の箱**

全てのヘッドホンユーザーに最高の音を届けるスタンドアロンDDC/DSPデバイス

### Vision
- 箱をつなぐ → 管理画面でポチポチ → 最高の音
- ユーザーに余計なことを考えさせない

### Core Features
- **2M-tap FIR upsampling:** 2,000,000タップ最小位相FIRフィルタ（197dB stopband）
- **Headphone EQ:** oratory1990データ + KB5000_7ターゲットカーブ
- **Auto-negotiation:** 入力レート自動検知、DAC性能に応じた最適化

## Tech Stack

| Layer | Technology |
|-------|------------|
| Control Plane | Python, FastAPI, scipy |
| Data Plane | C++17, CUDA, cuFFT |
| Communication | ZeroMQ |
| Audio I/O | PipeWire, ALSA |

## Hardware Targets

| Environment | Hardware |
|-------------|----------|
| Development | NVIDIA RTX 2070S (8GB, SM 7.5) |
| Production | Jetson Orin Nano Super (8GB, SM 8.7) |

## Project Structure

```
gpu_os/
├── src/               # C++/CUDA (convolution_engine.cu, alsa_daemon.cpp)
├── scripts/           # Python (generate_filter.py, analysis tools)
├── data/coefficients/ # FIR filter binaries (2M-tap)
├── web/               # FastAPI Web UI
└── docs/              # Documentation
```

## Key Commands

```bash
# Filter generation (2M taps)
uv sync
uv run python scripts/generate_filter.py --taps 2000000 --kaiser-beta 55

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run
./scripts/daemon.sh start
```

## Development Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core Engine & Middleware | In Progress |
| 2 | Control Plane & Web UI | Planned |
| 3 | Jetson Integration | Planned |

## Technical Constraints

1. **Minimum Phase** - プリリンギングなし（必須）
2. **2M taps** - 197dB stopband attenuation達成に必要
3. **Kaiser β=55** - 最適なストップバンド特性
4. **DC gain = 1.0** - クリッピング防止

## Git Workflow

**main直接コミット禁止。** Git Worktreeを使用:

### Mandatory Rules

1. **GitHub CLI (`gh`) Required:** GitHub操作（Issue、PR、ラベル等）は必ず `gh` コマンドを使用
2. **Issue Number Required:** ブランチ名・PR名には必ずIssue番号を含める
   - ブランチ名: `feature/#123-feature-name` または `fix/#456-bug-description`
   - PR名: `#123 機能の説明` または `Fix #456: バグの説明`

### Workflow

```bash
git worktree add worktrees/123-feature -b feature/#123-feature
cd worktrees/123-feature
# ... work ...
git push -u origin feature/#123-feature
gh pr create --title "#123 機能の説明" --body "..."
```

## Reference

- `CLAUDE.md` - 詳細な技術仕様
- `docs/roadmap.md` - 開発計画
