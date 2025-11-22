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

- **GPU-accelerated audio upsampling** with 2M-tap minimum phase FIR filter (197dB stopband)
- **Headphone EQ correction** using oratory1990 data + KB5000_7 target curve
- **Standalone DDC/DSP device** running on Jetson Orin Nano (production) or PC (development)

## Architecture Overview

```
Control Plane (Python/FastAPI)     Data Plane (C++ Audio Engine)
├── Web UI                         ├── PipeWire/ALSA Input
├── IR Generator (scipy)           ├── GPU FFT Convolution (CUDA)
├── oratory1990 Integration        ├── libsoxr Resampling
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
uv run python scripts/generate_filter.py --taps 2000000

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

## Git Workflow

**Never commit directly to main.** Use Git Worktree:

### Mandatory Rules

1. **GitHub CLI (`gh`) Required:** GitHub操作（Issue、PR、ラベル等）は必ず `gh` コマンドを使用
2. **Issue Number Required:** ブランチ名・PR名には必ずIssue番号を含める
   - ブランチ名: `feature/#123-feature-name` または `fix/#456-bug-description`
   - PR名: `#123 機能の説明` または `Fix #456: バグの説明`

### Workflow

```bash
git worktree add ../gpu_os_#123-feature -b feature/#123-feature
cd ../gpu_os_#123-feature
# ... work ...
git push -u origin feature/#123-feature
gh pr create --title "#123 機能の説明" --body "..."
```

## Testing

- Validate with sample WAVs in `test_data/`
- Run `scripts/verify_frequency_response.py` for filter changes
- Test realtime via `gpu_upsampler_alsa` with PipeWire null sink

## Key Technical Constraints

1. **Minimum Phase FIR** - No pre-ringing allowed
2. **2M taps** - Required for 197dB stopband attenuation
3. **Kaiser window** - β=55 for optimal stopband performance
4. **DC gain = 1.0** - Normalized to prevent clipping

## Reference

- See `CLAUDE.md` for detailed technical specifications
- See `docs/roadmap.md` for full development plan
