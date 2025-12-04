# Clip Headroom Investigation

## Symptoms

- Admin UI reported `clip_rate ≈ 0.2%` even when EQ was disabled.
- `clip_count` comes directly from the ALSA output thread, i.e. the last stage before samples
  are converted to `int32` and written to the DAC, so these are true output clips.

## Instrumentation Updates

- `alsa_daemon.cpp` now records per-stage peak levels (PipeWire input, GPU/FIR output,
  crossfeed/buffer, post gain + soft-mute). The values are exported via
  `/tmp/gpu_upsampler_stats.json` and surfaced in the Admin UI.
- The HTML dashboard renders the new `Peak Monitor` card so we can spot which stage
  pushes the signal beyond 0 dBFS without re-attaching a debugger.

## Root Cause

- The minimum-phase FIR is correctly normalized for unity DC gain **after** the 16× upsampler,
  which means the impulse response itself contains coefficients larger than 1.0.
- `data/coefficients/filter_44k_16x_2m_linear_phase.json` lists
  `max_coefficient_amplitude: 1.031726...` (≈ +0.26 dBFS). Any transient that excites that
  coefficient—e.g. a 0 dBFS snare or impulse—will exceed the ±1.0 digital full scale before
  we reach the ALSA driver.
- Because the global `gain` default is `1.0`, there is no reserved headroom and the safety
  clamp in `alsa_output_thread()` has to engage, producing the audible clicks that were
  observed.

## Recommended Workflow / Tests

1. **Filter audit**
   ```bash
   uv run python scripts/check_headroom.py \
     --filter data/coefficients/filter_44k_16x_2m_linear_phase.bin
   ```
   Reports the theoretical maximum coefficient and required headroom.

2. **Max-level stimulus**
   ```bash
   uv run python scripts/generate_test_audio.py \
     --amplitude 0.999 --duration 10 --sample-rate 44100
   ```
   Feed the generated WAV through the daemon and capture the output (PipeWire or loopback).

3. **PCM histogram & clip check**
   ```bash
   uv run python scripts/check_headroom.py \
     --output-wav test_output/captured.wav \
     --fail-on-clip --fail-threshold 0.995
   ```
   Fails the pipeline if even a single sample hits the clamp.

4. **Long duration stress**
   ```bash
   uv run python scripts/watch_clip_rate.py \
     --duration 900 --headroom-threshold 0.995
   ```
   Monitors the live stats file and aborts if the clip counter increments or if the
   post-gain peaks approach the threshold.

## Next Steps

- Decide whether to bake a fixed safety headroom (e.g. `gain = 0.94`), or implement an
  automatic limiter that trims `g_peak_post_gain` back under a target threshold.
- Extend the headroom script to run the same inspection across both 44.1 kHz and 48 kHz
  filter families to guarantee coverage before shipping new coefficients.
