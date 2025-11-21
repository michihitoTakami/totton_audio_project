# Repository Guidelines

## Language
- Think in English and output in Japanese.

## Project Structure & Module Organization
- `src/`: CLI entry (`main.cpp`), GPU core (`convolution_engine.cu`, `audio_io.cpp`), daemons (`pipewire_daemon.cpp`, `alsa_daemon.cpp`).
- `include/`: Public headers for the C++/CUDA targets.
- `data/coefficients/`: FIR tap binaries and metadata; keep any regenerated filters aligned with `filter_coefficients.h`.
- `scripts/`: Python tools for filter generation/verification and waveform analysis.
- `docs/` and `test_data/`: Setup notes, reports, and reference WAVs. Avoid rewriting large binaries unless strictly needed.

## Build, Test, and Development Commands
- Configure & build: `cmake -B build -DCMAKE_BUILD_TYPE=Release` then `cmake --build build -j$(nproc)` (builds CLIとデーモン)。
- Run CLI: `./build/gpu_upsampler input.wav output.wav --ratio 16 --block 4096`.
- Daemons: `./build/gpu_upsampler_alsa` (ALSA direct), `./build/gpu_upsampler_daemon` (PipeWire capture).
- Python helpers (>=3.11): e.g., `python scripts/verify_frequency_response.py data/coefficients/filter_1m_min_phase.bin`.

## Coding Style & Naming Conventions
- C++17/CUDA, 4-space indent, braces on the control line, prefer RAII and `std::vector` over raw pointers.
- Types use `PascalCase`, functions/methods `camelCase`, constants/macros `UPPER_SNAKE`.
- Keep GPU settings explicit (arch set to 75 in `CMakeLists.txt`); update when targeting different hardware.
- Preserve warning flags (`-Wall -Wextra`) and log errors before returning.

## Testing Guidelines
- No automated unit suite yet; validate changes with the sample WAVs in `test_data/` and capture before/after stats.
- For filter or DSP changes, run `scripts/verify_frequency_response.py` and attach plots/metrics.
- For realtime paths, test end-to-end via `gpu_upsampler_alsa` with the PipeWire null sink (see `docs/setup_guide.md`), confirming stable streaming and correct sample rate.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative summaries similar to existing history (`Fix overlap-save blocking`, `Add PipeWire daemon logging`).
- PRs should describe what changed, why, how to verify (commands + expected logs/plots), and any performance impact.
- Link related issues or phase docs, note CUDA/PipeWire/ALSA requirements, and call out any new binaries or config files added to `data/` or `docs/`.
