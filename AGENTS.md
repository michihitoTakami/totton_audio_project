# Magic Box Project - AI Collaboration Guidelines

## Language
- Think in English and output in Japanese.

## Project Vision

**Magic Box Project - 魔法の箱**

全てのヘッドホンユーザーに最高の音を届ける箱

**Ultimate Simplicity:**
1. 箱をつなぐ
2. 管理画面でポチポチ
3. 最高の音

## What This Project Does

- **GPU-accelerated audio upsampling** with 640k-tap minimum phase FIR filter (Kaiser β≈28, ~160dB stopband)
- **Headphone EQ correction** using OPRA data (CC BY-SA 4.0) + KB5000_7 target curve
- **Standalone DDC/DSP device** running on Jetson Orin Nano (production) or PC (development)

## Architecture Overview

```
Control Plane (Python/FastAPI)     Data Plane (C++ Audio Engine)
├── IR Generator (scipy)           ├── GPU FFT Convolution (CUDA)
├── OPRA Integration               ├── libsoxr Resampling
└── ZeroMQ Command Interface   <-> └── ALSA Output
```

## Development Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core Engine & Middleware | In Progress |
| 2 | Control Plane & Web UI | Planned |
| 3 | Jetson Hardware Integration | Planned |

## Project Structure

```
gpu_os/
├── src/                   # C++/CUDA source (convolution_engine.cu, alsa_daemon.cpp, etc.)
├── include/               # C++ headers
├── scripts/               # Python tools (filter generation, analysis)
├── data/coefficients/     # FIR filter binaries
├── data/EQ/               # EQ profiles
├── docs/                  # Documentation
├── web/                   # FastAPI Web UI
└── build/                 # Build output
```

## Build & Run Commands

```bash
# Filter generation
uv sync
uv run python scripts/filters/generate_minimum_phase.py --taps 640000

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run daemon
./scripts/daemon.sh start
```

## Coding Standards

- **C++17/CUDA**, 4-space indent, braces on control line
- **RAII** and `std::vector` over raw pointers
- **Naming:** `PascalCase` (types), `camelCase` (functions), `UPPER_SNAKE` (constants)
- GPU arch: SM 7.5 (PC) or SM 8.7 (Jetson)

### Web UI Component Reuse (Mandatory)

以下のJinja2マクロ（またはHTMLパーツ）が `web/templates/components/` に既に存在します。
**新しいUIを作る際は、必ずこれらをimportして再利用してください。ベタ書き禁止。**

- `{% macro btn_primary(text, icon) %}` - プライマリボタン
- `{% macro card_panel(title) %}` - カードパネル
- `{% macro slider_input(value) %}` - スライダー入力

**重複実装を避け、DRY原則を徹底すること。**

## Git Workflow

**Never commit directly to main.** Use Git Worktree:

### Mandatory Rules

1. **GitHub CLI (`gh`) Required:** GitHub操作（Issue、PR、ラベル等）は必ず `gh` コマンドを使用
2. **Issue Number Required:** ブランチ名・PR名には必ずIssue番号を含める
   - ブランチ名: `feature/#123-feature-name` または `fix/#456-bug-description`
   - PR名: `#123 機能の説明` または `Fix #456: バグの説明`
3. **No Auto Merge Without Explicit User Request (必須)**:
   - **ユーザーが明示的に「マージして」と依頼した場合のみ** `gh pr merge` / merge API 等の **マージ操作を実行**すること。
   - ユーザーが「マージ判断をせよ」「レビューして問題なければ判断」と言った場合は、**マージ可否の結論と根拠を提示するだけ**で、マージ操作は行わない。
   - マージが必要そうでも、**明示依頼が無い限りは “マージ可能です。マージしますか？” 相当の確認を文章で返す**（マージ操作はしない）。

### Workflow

```bash
git worktree add worktrees/123-feature -b feature/#123-feature
cd worktrees/123-feature
# ... work ...
git push -u origin feature/#123-feature
gh pr create --title "#123 機能の説明" --body "..."
```

**必須:** ワークツリーを作成する前に必ず `git fetch origin main` を実行し、最新の`origin/main`を取り込んでから作業を開始すること。過去のmainで作業を始めるとコンフリクト発生率が高くなる。

## Testing
- Validate with sample WAVs in `test_data/`
- Run `scripts/analysis/verify_frequency_response.py` for filter changes
- Always generate `compile_commands.json` (e.g., `cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`) before lint/test runs so `clang-tidy`/`diff-based-tests` have the data they need.
- Do **not** skip the `clang-tidy`/`diff-based-tests` pre-push hooks; run `pre-commit run --hook-stage pre-push` (or `git push`) and resolve any failures before submitting changes.

## RTP Audio Streaming Notes

- **RTP L16/L24/L32 はネットワークバイトオーダー（BE）**前提（送信・受信で `S24BE` 等に統一すること）
- **payload type は 96 を使用**（送受で不一致だとcapsが噛み合わない）
- 送信側（Pi）は `alsasrc` の `buffer-time/latency-time` を明示し、変換/リサンプル遅延を吸収するため **`queue` を必ず挟む**

## Key Technical Constraints

1. **Minimum Phase FIR** - No pre-ringing allowed
2. **640k taps** - ~160dB stopband attenuation (sufficient for 24-bit audio)
3. **Kaiser window** - β≈28 (optimal for 32-bit Float GPU implementation)
4. **DC gain = 1.0** - Normalized to prevent clipping

## Reference

- See `CLAUDE.md` for detailed technical specifications
- See `docs/roadmap.md` for full development plan
