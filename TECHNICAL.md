# Technical Details

In-depth documentation of the algorithms, design decisions, and architecture behind Beep Cleaner.

## Architecture Overview

The project has two processing modes sharing the same core ideas but with different implementations optimized for their constraints:

| | File Mode (`beep_cleaner.py`) | Real-Time Mode (`beep_cleaner_realtime.py`) |
|--|--|--|
| Input | Complete audio file | Audio stream, block by block |
| Detection | High-res STFT over entire file | Per-block windowed FFT |
| Removal | `filtfilt` (zero-phase) + STFT spectral subtraction | Stateful `lfilter` (causal) + FFT spectral zeroing |
| Latency | N/A (offline) | ~43 ms + 3-block lookback delay |
| Measured suppression | 70+ dB | ~47 dB |

### Class Hierarchy

**File mode:**
- `CleanerConfig` — all tunable parameters with sensitivity presets
- `BeepDetector` — STFT-based multi-criteria detection
- `BeepRemover` — multi-stage removal with crossfade blending
- `visualize()` — spectrogram comparison plots

**Real-time mode:**
- `RealtimeConfig` — streaming-specific parameters with sensitivity presets
- `RealtimeBeepProcessor` — per-block detection + stateful IIR suppression
- `RealtimeBeepCleaner` — `sounddevice` stream manager (input→process→output loop)

---

## Detection

### Why These Metrics?

A beep is a short, constant-frequency sinusoidal tone. This gives it three distinguishing properties compared to speech, music, or noise:

1. **Extreme spectral concentration** — Nearly all energy sits at one frequency. Speech spreads energy across formants, harmonics, and noise bands.
2. **Constant pitch** — The frequency doesn't drift over the beep's duration. Speech formants shift continuously.
3. **Abrupt onset** — Beeps start and stop suddenly. Speech transitions are gradual (10–50 ms ramps).

We exploit all three with separate metrics, then combine them into a confidence score to prevent false positives.

### File Mode: STFT-Based Detection

**STFT parameters:**
- 4096-point FFT → ~5.9 Hz resolution at 48 kHz
- 128-sample hop → ~2.7 ms time resolution at 48 kHz
- This gives dense time-frequency coverage at the cost of processing time (acceptable in offline mode)

**Per-frame metrics:**

1. **Peakiness** = `peak_magnitude / median_magnitude` in the target frequency band.
   - A pure 715 Hz beep in our test audio produced peakiness values of ~58,500,000. Typical speech peaks: ~10,000–40,000.
   - Threshold is set adaptively at the 95th percentile with a floor of 50.

2. **Spectral purity** = `energy_in_±3_bins / total_energy`.
   - A perfect sinusoid windowed with a Hann window has purity ≈ 1.0. Speech typically: 0.1–0.5.
   - The ±3 bin window (≈ ±17 Hz at 4096-point FFT) accommodates the spectral leakage from windowing.

3. **Energy ratio** = `peak_bin_magnitude / median_spectrum_magnitude`.
   - Calibrated from real audio: a 715 Hz beep at normal volume produces energy ratios >150,000,000. Loudest speech frame: ~41,000.
   - Default threshold: 50,000 (medium), providing a 1200× safety margin above speech.

**Candidate filtering pipeline:**

```
STFT frames
  → peakiness + purity thresholds → candidate frames
  → group consecutive frames into segments
  → duration filter (10–2000 ms)
  → frequency stability filter (σ < 30 Hz)
  → energy ratio filter (> 50,000)
  → onset sharpness check
  → weighted confidence score
  → harmonic scan
  → final beep list
```

**Confidence scoring weights:**

| Factor | Weight | Reasoning |
|--------|--------|-----------|
| Energy ratio | 30% | Strongest discriminator between beeps and speech |
| Spectral purity | 25% | Captures the "pure tone" nature |
| Peakiness | 15% | Redundant with energy ratio but catches edge cases |
| Onset sharpness | 15% | Rejects gradual speech transitions |
| Frequency stability | 10% | Rejects speech formant drift |
| Duration | 5% | Longer beeps are more likely intentional tones |

### Real-Time Mode: Per-Block FFT Detection

Real-time mode cannot look at the whole file. Instead, each 2048-sample block (~43 ms) is analyzed independently:

1. **Hann window + zero-padded FFT** — The block is windowed, then FFT'd with 4× zero-padding (8192-point FFT for a 2048-sample block). This gives ~5.9 Hz bin spacing, matching the file mode's resolution.

2. **Parabolic interpolation** — The FFT bin spacing (even with zero-padding) may not align with the exact beep frequency. We refine the peak estimate using the three bins around the maximum:
   $$\hat{f} = (k + p) \cdot \frac{f_s}{N_{\text{FFT}}}$$
   where $p = \frac{1}{2} \cdot \frac{\alpha - \gamma}{\alpha - 2\beta + \gamma}$, and $\alpha, \beta, \gamma$ are the magnitudes of bins $k-1, k, k+1$.

   This typically reduces frequency error from ~12 Hz (raw bin center) to <2 Hz.

3. **Purity and energy ratio** — Computed identically to file mode, but on the zero-padded spectrum. The purity radius is scaled to ±12 bins (equivalent to ±3 at 1× resolution).

4. **Consecutive-block confirmation** — A single block might contain a transient that looks tonal. Requiring `min_beep_blocks` consecutive detections (default: 2, ~86 ms) eliminates virtually all false positives.

**Why these thresholds are so different from file mode:**

File mode's STFT uses a 128-sample hop, meaning a 100 ms beep spans ~18 frames. Each frame's metrics are averaged across the segment. Real-time mode sees only one 43 ms block at a time, so the metrics are noisier. The thresholds were calibrated empirically:

| Metric | File Mode (medium) | Real-Time (medium) | Why different |
|--------|--------------------|--------------------|---------------|
| Purity | 0.3 | 0.7 | Single-block purity is higher variance; raising it prevents noisy speech frames from triggering |
| Energy ratio | 50,000 | 50,000 | Same — this metric is surprisingly stable per-block |
| Confidence | 0.5 | N/A | Not used in real-time; replaced by consecutive-block confirmation |

---

## Removal

### Why Cascaded Notch Filters?

A single IIR notch filter (`scipy.signal.iirnotch`) has a limited stop-band depth per Q factor:

- **Low Q (wide notch):** removes a broad band but only ~15–20 dB deep. Good for catching the beep even if detection frequency is slightly off, but insufficient suppression.
- **High Q (narrow notch):** very deep (~40+ dB) but only within a few Hz. Misses energy outside the exact center.

Cascading multiple Q values gives both coverage and depth:

```
Q=5  (wide):    ██████████████████████████  ← catches ±100 Hz spread
Q=10 (medium):    ████████████████████      ← deepens the center
Q=20 (narrow):        ████████████          ← surgical precision at exact freq
```

Each Q is applied across `n_passes` passes (default: 3 in high sensitivity, 2 in medium). With 3 Q values × 3 passes = 9 filter applications per frequency. This achieves >70 dB suppression at the target frequency.

### File Mode: `filtfilt` (Zero-Phase)

File mode uses `scipy.signal.filtfilt`, which applies the filter forward and backward. This doubles the effective order and produces zero phase distortion — the filtered signal is perfectly time-aligned with the original.

**Why `filtfilt` is unavailable in real-time:** It requires the entire signal (or at least a full buffer) since it processes backwards. Real-time mode can only process causally (forward-only).

### File Mode: STFT Spectral Suppression (Stage 2)

After notch filtering, residual energy may remain in side-lobes. The STFT stage directly multiplies the complex STFT coefficients in a ±40 Hz band around each target frequency by a suppression factor (default: $10^{-40/20} = 0.01$ = −40 dB).

```
STFT(filtered_region) → zero out ±40 Hz bands → ISTFT → blended output
```

This catches anything the notch filters missed. The combination of time-domain (notch) and frequency-domain (STFT) suppression is what achieves the 70+ dB reduction.

### File Mode: Crossfade Blending

The filtered region is blended with the original audio at the boundaries using a linear crossfade envelope (default: 5 ms):

```
|---original---|==crossfade==|---filtered---|==crossfade==|---original---|
               ↑ 5ms ramp    ↑ beep region  ↑ 5ms ramp
```

Without crossfade, the sudden switch between original and filtered signal would produce audible clicks (due to waveform discontinuity at the splice point). The 5 ms ramp is short enough to be imperceptible while long enough to suppress discontinuity transients below the audibility threshold.

### Real-Time Mode: Stateful `lfilter`

Real-time mode uses `scipy.signal.lfilter` with explicit filter state (`zi`) that persists across blocks. This is the key to seamless block-to-block filtering:

```python
# Each block continues from where the previous one left off
out, self._filter_states[key] = signal.lfilter(b, a, block, zi=self._filter_states[key])
```

On the first block of a beep, the filter state is initialized with `lfilter_zi`:

```python
self._filter_states[key] = signal.lfilter_zi(b, a) * block[0]
```

This produces a state that assumes the filter has been running on a constant signal equal to the first sample, minimizing startup transients.

**Multi-pass in real-time:** Each Q value is applied 3 times (passes), giving 3 Q × 3 passes = 9 cascaded filters per frequency. Each filter instance has its own state keyed by `(freq, q_idx, pass_idx)`. States persist as long as the beep is active and are cleared on beep release.

### Real-Time Mode: Spectral Suppression for Forward Blocks

In addition to the IIR notch cascade, each confirmed beep block also gets FFT-domain suppression:

```python
output = self._suppress_tone(block, freq)        # IIR cascade
output = self._suppress_tone_spectral(output, freq)  # FFT zeroing
```

This provides additional depth without requiring filter state, and fills in any notch filter gaps.

### Real-Time Mode: Lookback Buffer for Onset Suppression

**The problem:** When a beep starts, the first block contains beep energy but hasn't yet been confirmed (we need `min_beep_blocks` consecutive detections). By the time we confirm at block $N$, blocks $N - 2$ through $N - 1$ have already been output unsuppressed.

**The solution:** A 3-block lookback buffer delays output. When a beep is confirmed at block $N$:

1. Blocks $N - 3$ through $N - 1$ are still in the buffer
2. They are retroactively processed with stateless FFT spectral suppression
3. Then output in order

```
Block timeline:    ... | N-3 | N-2 | N-1 |  N  | N+1 | ...
                               ↑ onset    ↑ confirmed
                                          (retroactively suppress N-3..N-1)
```

**Why stateless (FFT) for lookback:** IIR filters require continuous state. Retroactively filtering already-passed blocks with IIR would create state discontinuities and audible artifacts. FFT-domain suppression is stateless — each block is independently transformed, filtered, and inverse-transformed. The trade-off is less precise suppression (broader frequency zeroing), but since the onset blocks typically have lower beep energy (the tone is ramping up), this is sufficient.

**Added latency:** 3 blocks × 43 ms = ~129 ms total pipeline delay (lookback buffer fill + processing). This is acceptable for broadcast monitoring and call recording, though may be noticeable in live conversation.

### Release Hold

After the last detected beep block, suppression continues for `release_blocks` (default: 3) additional blocks. This handles:

- **Beep tail energy** — the tone may decay over 1–2 blocks after the detection threshold is no longer met
- **FFT windowing artifacts** — the Hann window spreads energy across block boundaries
- **Detection jitter** — energy ratio may dip below threshold momentarily during the beep

When the release period ends and the beep is confirmed gone, all filter states are cleared (`_reset_filter_states()`). This prevents stale state from affecting future unrelated audio.

---

## Harmonic Handling

Real-world beeps often contain harmonics due to:
- Non-linear electronics (square-wave generators produce odd harmonics)
- Clipping during recording
- Codec artifacts

**File mode** scans for harmonics 2 through 8 by checking if $\text{spectrum}[n \cdot f_0] > 3 \times \text{median}$, and adds confirmed harmonics to the removal targets.

**Real-time mode** unconditionally filters harmonics 2–5 of the detected fundamental. This is more aggressive but avoids the cost of per-block harmonic detection.

---

## Sensitivity Presets: Design Rationale

### `low` — Conservative

- Only catches unambiguous beeps (purity > 0.5, energy ratio > 80K)
- Single pass, gentler suppression (−20 dB)
- Use case: preserving audio quality when false positives are unacceptable

### `medium` — Balanced (default)

- Moderate thresholds calibrated on real recordings
- 2 passes with 3 Q values
- The energy ratio threshold (50,000) provides >1200× margin above typical speech

### `high` — Aggressive

- Lower thresholds to catch faint or brief beeps
- 3 passes, 4 Q values, −40 dB spectral suppression
- May produce slight narrowband artifacts on speech with strong harmonic content near the beep frequency

---

## Performance Characteristics

### File Mode

Processing is dominated by:
1. STFT computation (numpy/librosa FFT) — $O(N \log N)$ per frame
2. Per-frame metric computation — $O(N_{\text{freq}})$ per frame
3. Notch filtering — $O(N \cdot P \cdot Q)$ where $P$ = passes, $Q$ = Q values

Typical throughput: ~2–5 seconds per minute of audio at 48 kHz on a modern laptop.

### Real-Time Mode

Per-block budget at 48 kHz with 2048-sample blocks: **42.7 ms**.

Actual computation per block:
- FFT detection (8192-point with zero-pad): ~0.5 ms
- 9× IIR notch filters (fundamental): ~0.2 ms
- 9× IIR notch filters (4 harmonics): ~0.8 ms
- FFT spectral suppression: ~0.1 ms
- Total: ~1.6 ms → **~4% CPU** at 48 kHz

This leaves ample headroom. The actual bottleneck is the `sounddevice` I/O latency, not processing.

---

## Key Design Decisions

### Why IIR notch filters instead of FIR?

- IIR notch filters are extremely efficient (2nd-order, 5 coefficients per filter) and provide infinite stop-band depth in theory
- FIR would require hundreds of taps to achieve equivalent narrowband rejection
- IIR's `lfilter` with state propagation gives seamless block-to-block processing

### Why not just use FFT zeroing for everything?

- FFT zeroing on short blocks (2048 samples) has poor frequency resolution and produces audible "musical noise" artifacts at block boundaries
- IIR notch filters operate in the time domain and produce smooth, continuous output
- The hybrid approach (IIR for forward blocks + FFT for lookback) gives the best of both worlds

### Why not use overlap-add for real-time?

- Overlap-add would provide better frequency-domain processing quality but at the cost of 2× latency (50% overlap) and more complex buffer management
- The IIR approach achieves sufficient suppression (~47 dB) at lower latency and complexity
- For use cases requiring >50 dB suppression, the file mode should be used

### Why `filtfilt` in file mode but `lfilter` in real-time?

- `filtfilt` (forward-backward filtering) doubles the effective filter order and produces zero-phase output. It requires the complete signal.
- `lfilter` (forward-only) is causal and supports streaming with state propagation. The trade-off is half the effective order (hence multi-pass to compensate) and a small phase shift at the notch frequency (inaudible since we're suppressing that frequency anyway).

### Why parabolic interpolation for real-time frequency detection?

With a 2048-sample block at 48 kHz and 4× zero-padding, FFT bin spacing is ~5.9 Hz. The beep in our test audio is at ~715 Hz. Without interpolation, the nearest bin is at 703 Hz — a 12 Hz error. The notch filter centered at 703 Hz partially misses the 715 Hz energy, reducing suppression by ~10 dB. Parabolic interpolation narrows this error to <2 Hz, which is well within the narrowest notch filter bandwidth.
