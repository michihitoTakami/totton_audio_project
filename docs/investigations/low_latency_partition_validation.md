# Low-Latency Partition Validation

## Goals

- Verify that the fast partition delivers the promised latency (<10–12 ms for the default 32k taps).
- Confirm that the tail partitions catch up without introducing ripples or stopband regressions.
- Provide a repeatable loopback workflow (ALSA/PipeWire) that records impulse and sweep data while monitoring XRUN and GPU load.
- Document how to feed the captured files into the updated analysis scripts.

---

## Prerequisites

- PipeWire 1.0+ with `pw-loopback` and `pw-record`.
- ALSA loopback module (`snd_aloop`) for systems without PipeWire.
- `uv` environment (or `python -m venv`) with project dependencies installed: `uv sync`.
- GPU daemon built from `main` after PR #371 (partition switches on via `/partitioned-convolution` API or `config.json`).

---

## Step 1 – Provision Test Stimuli

```bash
uv run python scripts/generate_test_audio.py \
  --output-dir test_data/low_latency \
  --duration 5.0 \
  --sample-rate 44100 \
  --amplitude 0.9
```

Artifacts:

- `test_impulse_44100hz.wav` – single impulse for IR reconstruction.
- `test_sweep_44100hz.wav` – 20 Hz→20 kHz log sweep (used for frequency verification).

---

## Step 2 – PipeWire Loopback Capture

1. **Connect daemon output to a dedicated loopback sink**

   ```bash
   pw-loopback --playback-props='media.class=Audio/Sink node.name=magicbox-lowlat-sink' \
               --capture-props='node.name=magicbox-loop-capture'
   ```

2. **Route test stimulus into the daemon**

   - Play `test_impulse_44100hz.wav` via the normal playback stack (e.g. `pw-play`).
   - Ensure the daemon is running with `partitionedConvolution.enabled = true` and the desired taps.

3. **Record the upsampled output**

   ```bash
   pw-record --target=magicbox-loop-capture --channels=2 \
     --rate 705600 test_output/lowlat_impulse.wav
   pw-record --target=magicbox-loop-capture --channels=2 \
     --rate 705600 test_output/lowlat_sweep.wav
   ```

4. **Monitor health**

   ```bash
   watch -n0.5 "pw-top | head -n20"
   watch -n0.5 "cat /proc/asound/card*/pcm*/sub0/status | grep -E 'XRUNs|appl_ptr'"
   watch -n1 "nvidia-smi dmon -s u"   # GPU utilization
   ```

5. **ALSA fallback**: load the loopback module (`sudo modprobe snd_aloop`) and replace the `pw-*` commands with `aplay/arecord`.

---

## Step 3 – Inspect the Filter / Partition Plan

```bash
uv run python scripts/inspect_impulse.py \
  --coeff data/coefficients/filter_44k_16x_2m_hybrid_phase.bin \
  --metadata data/coefficients/filter_44k_16x_2m_hybrid_phase.json \
  --config config.json \
  --enable-partition \
  --summary-json plots/analysis/partition_summary.json
```

Outputs:

- PNG plot highlighting the fast/tail regions.
- Printed partition table (taps, FFT size, energy share) plus:
  - **Fastパーティション累積エネルギー**（fastPartition taps内に収まるエネルギー%）
  - **プリリンギング比**（ピークより前のエネルギー%）
  - **100Hz帯域の群遅延**（平均と変動幅、グラフ表示）
- JSON summary (`fast_partition_energy_pct`, `pre_ringing_pct`, `group_delay_100hz`) used by QA spreadsheets.

---

## Step 4 – Verify Frequency Response (Fast vs Combined)

```bash
uv run python scripts/verify_frequency_response.py \
  test_data/low_latency/test_sweep_44100hz.wav \
  test_output/lowlat_sweep.wav \
  --metadata data/coefficients/filter_44k_16x_2m_hybrid_phase.json \
  --config config.json \
  --analysis-window-seconds 1.5 \
  --compare-fast-tail \
  --reference-output test_output/lowlat_sweep_minphase.wav \
  --reference-label "Min-phase ref"
```

- The script skips the automatically calculated settling time before computing the primary FFT.
- `--compare-fast-tail` prints the spectral delta between the initial fast-only block and the steady-state response (should be <0.5 dB across the band).
- When `--reference-output` is provided, the output spectrum plot overlays the baseline (旧最小位相) and a third panel renders the **Hybrid−Reference** dB delta. The console also reports max/RMS deltas (`reference_delta_max_db`, `reference_delta_rms_db`).
- Baseline capture tip: run the same sweep with `partitionedConvolution.enabled=false` (or fast taps=full) and save it as `test_output/lowlat_sweep_minphase.wav`.

For impulse captures:

```bash
uv run python scripts/verify_frequency_response.py \
  test_data/low_latency/test_impulse_44100hz.wav \
  test_output/lowlat_impulse.wav \
  --metadata data/coefficients/filter_44k_16x_2m_hybrid_phase.json \
  --config config.json \
  --settling-seconds 0.015 \
  --plot plots/analysis/lowlat_impulse_spectrum.png
```

---

## Step 5 – Checklist / Acceptance Criteria

- **Latency**: `inspect_impulse.py` reports `fast_window ≈ 32k samples` → `<11 ms @ 705.6 kHz`.
- **Energy split**: `fast_partition_energy_pct ≥ 97%` forハイブリッド係数（32k fast taps基準）。
- **Pre-ringing**: `pre_ringing_pct ≤ 0.10%`.
- **Group delay continuity**: `group_delay_100hz.span_ms ≤ 0.40 ms`.
- **Stopband**: `verify_frequency_response.py` shows stopband attenuation within 3 dB of the legacy (non-partitioned) pipeline.
- **Hybrid vs legacy delta**: `reference_delta_max_db ≤ 0.4 dB` and `reference_delta_rms_db ≤ 0.15 dB`.
- **XRUN**: `/proc/asound/.../status` stays at `XRUNs: 0` for at least 10 minutes during sweep playback.
- **GPU load**: `nvidia-smi dmon` stays under 20% on RTX 2070S when low-latency mode is active.

Record results in `docs/jetson/quality/test-checklist.md` (new section “低遅延パーティション検証”).

---

## Troubleshooting

- **Tail never catches up**: verify `maxPartitions` ≥ 3 and `fastPartitionTaps < total taps`. Use the JSON summary to confirm.
- **FFT artifacts**: ensure the capture file is at the upsampled output rate (705.6k/768k). Down-sampled captures hide stopband leakage.
- **XRUN bursts**: increase PipeWire buffer (`pw-metadata -n settings 0 clock.min.quantum 256`) and confirm `periodSize` in `config.json` matches.
