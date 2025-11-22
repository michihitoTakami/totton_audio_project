# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Most important rule about language
Think in English and answer in Japanese.

## Project Overview

**GPU-Driven High-Precision Audio Oversampling Plugin**

This project implements GPU-accelerated ultra-high-precision audio upsampling for Linux audio environments (PipeWire/Easy Effects). The goal is to achieve upsampling quality impossible with traditional DAC chip filters or CPU-based processing by leveraging GPU compute power for massive FIR filter convolution.

**Key Specifications:**
- Input: 44.1kHz/48kHz (16-32bit float)
- Output: 705.6kHz/768kHz (16x oversampling)
- Filter: 131,072-tap minimum phase FIR
- Target Hardware: NVIDIA GeForce RTX 2070 Super (8GB VRAM) or better
- Plugin Format: LV2 for Easy Effects integration

## Three-Phase Development Architecture

The project follows a structured three-phase approach. **Work must proceed sequentially - do not skip ahead to later phases.**

### Phase 1: Algorithm Verification & Coefficient Generation (Python)

**Purpose:** Generate and validate the 131k-tap minimum phase FIR filter coefficients.

**Directory Structure:**
```
scripts/           - Filter generation Python scripts
data/coefficients/ - Generated filter coefficient files
plots/analysis/    - Validation plots (impulse, frequency response)
```

**Key Requirements:**
- Use `scipy.signal` for filter design
- Start with linear phase design, convert to minimum phase via homomorphic processing
- **Critical constraint:** Minimum phase is MANDATORY to eliminate pre-ringing (pre-ringing degrades transient response)
- Frequency specs:
  - Passband: 0-20kHz (flat response)
  - Stopband start: 22.05kHz (for 44.1kHz input)
  - Stopband attenuation: ≤-180dB
  - Window: Kaiser (β ≈ 18)

**Validation Checklist:**
- Impulse response shows NO pre-ringing (energy concentrated at time=0 and forward)
- Frequency response meets -180dB stopband requirement
- Passband ripple is minimal within audible range

**Export Formats:**
- Binary (.bin): float32 array for direct loading
- C++ header (.h): for Phase 2 integration
- Include metadata: tap count, sample rate, achieved specs

### Phase 2: GPU Convolution Engine (C++ Standalone)

**Purpose:** Build and benchmark GPU-accelerated FFT convolution engine without real-time constraints.

**Tech Stack Decision:**
- **Primary (recommended):** Vulkan Compute + VkFFT
  - Rationale: Vendor-agnostic, excellent Linux compatibility, VkFFT is extremely fast
- **Alternative:** CUDA + cuFFT
  - Rationale: Lower implementation complexity, guaranteed RTX 2070S performance, suitable for prototyping

**Implementation Approach:**
- Create console application (no audio I/O dependency yet)
- Input: WAV file (44.1/48kHz)
- Processing: GPU FFT convolution using Partitioned FFT (Overlap-Save or Overlap-Add method)
- Output: WAV file (705.6/768kHz)

**Why Partitioned FFT:**
With 131k taps, direct time-domain convolution is computationally prohibitive. FFT convolution transforms the problem into frequency-domain multiplication, dramatically reducing complexity from O(N×M) to O(N×log(N)).

**Performance Target:**
- Real-time processing with <20% GPU utilization on RTX 2070S
- This ensures sufficient headroom for Phase 3's real-time streaming constraints

**Critical Implementation Details:**
- Filter coefficients pre-loaded into GPU memory (one-time transfer)
- Block size: 4096-8192 samples recommended (balances latency vs. efficiency)
- Overlap-Save/Add handles block boundaries to prevent artifacts

### Phase 3: LV2 Plugin Integration (Real-Time)

**Purpose:** Integrate Phase 2 engine into LV2 plugin framework for Easy Effects.

**New Challenges:**
- **Ring buffer management:** CPU-GPU async processing to prevent audio dropouts
- **Sample rate negotiation:** Handle PipeWire's rate constraints (most plugins assume input rate = output rate)
- **Latency reporting:** Use LV2's latency reporting feature (FFT blocking introduces ~tens of ms latency)

**Latency Philosophy:**
This is optimized for **listening/music playback quality**, not low-latency monitoring. Accept 50-100ms latency as acceptable trade-off for extreme filter quality. Report latency to host for proper video sync (lip-sync) compensation.

**PipeWire Integration Strategy:**
- Option A: Set PipeWire system-wide to max rate (768kHz), plugin operates at fixed rate
- Option B: Implement as dedicated upsampling Sink rather than in-chain effect
- Requires investigation of PipeWire's rate-change capabilities during Phase 3

## Algorithm Core: FFT Convolution Pipeline

**Data Flow (Phase 2 & 3):**
1. **Input Buffer (CPU):** Receive audio block from PipeWire/file
2. **Ring Buffer Accumulation:** Aggregate to 4096-8192 samples for efficient GPU transfer
3. **H2D Transfer:** CPU → GPU VRAM (PCIe bandwidth consideration)
4. **Partitioned FFT Convolution (GPU):**
   - Pre-loaded 131k-tap filter coefficients (already in GPU memory)
   - Perform FFT on input block
   - Complex multiply with filter's frequency-domain representation
   - IFFT to time domain
   - Overlap-Save/Add to handle block boundaries
5. **D2H Transfer:** GPU → CPU processed audio
6. **Output Buffer (CPU):** Send to host application

**Memory Footprint Estimate:**
- Filter coefficients: 131k taps × 4 bytes (float32) ≈ 512KB
- Working buffers for FFT: ~tens of MB depending on block size
- Well within 8GB VRAM budget of RTX 2070S

## Key Technical Constraints

**Minimum Phase Requirement:**
- Linear phase filters cause pre-ringing (artifacts BEFORE transients)
- Minimum phase concentrates impulse energy at t≥0, preserving transient attack
- This is non-negotiable for high-fidelity audio reproduction

**Stopband Attenuation (-180dB):**
- Ensures aliasing components are below quantization noise floor
- Requires large tap count + careful windowing (Kaiser β≈18)

**GPU Memory Management:**
- Filter coefficients loaded once at initialization
- Streaming data uses ring buffers to minimize transfer overhead
- Vulkan/CUDA device memory allocation should be persistent, not per-block

## Development Commands

**Phase 1 (Python - uv):**
```bash
# Setup environment (first time only)
uv sync

# Generate filter coefficients
uv run python scripts/generate_filter.py

# Output:
# - data/coefficients/filter_131k_min_phase.bin (512 KB binary)
# - data/coefficients/filter_coefficients.h (C++ header)
# - data/coefficients/metadata.json (filter specifications)
# - plots/analysis/*.png (validation plots)
```

**Phase 2 (C++):**
*TBD - Add CMake build commands, test WAV processing*

**Phase 3 (LV2 Plugin):**
*TBD - Add LV2 build/install commands, Easy Effects loading instructions*

## Git Workflow

**Always use Git Worktree for feature development and bug fixes.**

Do NOT commit directly to main. Instead:

```bash
# Create a new worktree for the feature branch
git worktree add ../gpu_os_<feature-name> -b feature/<feature-name>

# Work in the worktree directory
cd ../gpu_os_<feature-name>

# After completion, push and create PR
git push -u origin feature/<feature-name>
gh pr create --title "..." --body "..."

# Clean up after PR is merged
git worktree remove ../gpu_os_<feature-name>
```

**Rationale:**
- Keeps main branch clean and stable
- Enables parallel work on multiple features
- Facilitates proper code review via PRs
- Avoids accidental pushes to main

## Reference Projects

- **HQPlayer:** Commercial benchmark for target audio quality
- **VkFFT:** High-performance Vulkan FFT library (GitHub: DTolm/VkFFT)
- **CamillaDSP:** Linux FIR filter engine (CPU-based, architectural reference)

## Project Status

See `docs/first.txt` for complete Japanese specification document.

Current phase: **Phase 1** (not yet started)
