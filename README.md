# Beep Cleaner

A command-line tool for **automatic detection and removal of short pure-tone beeps** from audio files. Designed for cleaning up recorded calls, broadcasts, or any audio containing unwanted tonal artifacts (censorship beeps, equipment tones, alert sounds, etc.).

## Features

- **Automatic detection** — finds beeps by analyzing spectral purity, peakiness, and energy ratios; no need to manually specify frequency or timing
- **Harmonic-aware** — detects and removes beep harmonics (up to 8th) for thorough cleaning
- **Multi-stage removal** — cascaded IIR notch filters + STFT spectral suppression for deep attenuation (tested up to 70+ dB reduction)
- **Artifact-free** — crossfade blending at beep boundaries preserves surrounding audio
- **Sensitivity presets** — `low`, `medium`, `high` for different use cases
- **Visualization** — generates before/after spectrogram comparison plots
- **Format support** — WAV, OGG, FLAC, and other formats supported by libsndfile

## Installation

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy scipy librosa soundfile matplotlib
```

### Dependencies

| Package      | Version | Purpose                          |
|-------------|---------|----------------------------------|
| numpy       | ≥1.24   | Array operations                 |
| scipy       | ≥1.10   | IIR notch filter design          |
| librosa     | ≥0.10   | STFT, frequency analysis         |
| soundfile   | ≥0.12   | Audio I/O (WAV, OGG, FLAC)      |
| matplotlib  | ≥3.7    | Visualization (optional)         |

## Quick Start

```bash
# Basic usage — detect and remove beeps automatically
python beep_cleaner.py recording.ogg

# Specify output file
python beep_cleaner.py recording.ogg -o recording_clean.ogg

# High sensitivity + generate spectrum plot
python beep_cleaner.py recording.wav --sensitivity high --visualize

# Dry run — detect without modifying
python beep_cleaner.py recording.wav --dry-run
```

## Usage

```
usage: beep_cleaner [-h] [-o OUTPUT] [-s {low,medium,high}]
                    [--min-freq MIN_FREQ] [--max-freq MAX_FREQ]
                    [--min-duration MIN_DURATION] [--max-duration MAX_DURATION]
                    [--min-purity MIN_PURITY] [--min-confidence MIN_CONFIDENCE]
                    [--n-passes N_PASSES] [--visualize] [--viz-output VIZ_OUTPUT]
                    [--dry-run] [-v]
                    input

Detect and remove short pure-tone beeps from audio files.

positional arguments:
  input                 Input audio file path

options:
  -o, --output          Output file path (default: <input>_clean.<ext>)
  -s, --sensitivity     Detection sensitivity preset (default: medium)
  --min-freq            Minimum beep frequency in Hz (default: 200)
  --max-freq            Maximum beep frequency in Hz (default: 8000)
  --min-duration        Minimum beep duration in ms (default: 10)
  --max-duration        Maximum beep duration in ms (default: 2000)
  --min-purity          Minimum spectral purity 0–1 (default: 0.3)
  --min-confidence      Minimum confidence 0–1 (default: 0.5)
  --n-passes            Number of filter passes (default: 3)
  --visualize           Generate spectrum analysis plot
  --viz-output          Visualization output path (default: beep_analysis.png)
  --dry-run             Detect beeps but don't remove them
  -v, --verbose         Verbose output
```

## Sensitivity Presets

| Preset   | Use Case                          | Purity | Confidence | Energy Ratio | Passes | Suppression |
|----------|-----------------------------------|--------|------------|-------------|--------|-------------|
| `low`    | Only obvious, strong beeps        | 0.5    | 0.7        | 80,000      | 1      | −20 dB      |
| `medium` | General purpose (default)         | 0.3    | 0.5        | 50,000      | 2      | −30 dB      |
| `high`   | Catch faint or short beeps        | 0.2    | 0.3        | 30,000      | 3      | −40 dB      |

## Examples

### Clean a recorded phone call

```bash
python beep_cleaner.py call_recording.wav -o call_clean.wav
```

### Remove censorship beeps from a broadcast (typically ~1000 Hz)

```bash
python beep_cleaner.py broadcast.ogg --min-freq 800 --max-freq 1200 --sensitivity high
```

### Detect beeps without modifying the file

```bash
python beep_cleaner.py audio.flac --dry-run -v
```

### Generate before/after comparison

```bash
python beep_cleaner.py audio.ogg --visualize --viz-output comparison.png
```

## How It Works

### Detection Pipeline

1. **STFT Analysis** — Computes a high-resolution Short-Time Fourier Transform (4096-point FFT, 128-sample hop) for precise time-frequency localization.

2. **Frame Scoring** — For each time frame, computes:
   - **Peakiness**: ratio of peak magnitude to median magnitude in the target frequency band. Pure tones produce extremely high peakiness (>10,000×).
   - **Spectral purity**: fraction of total energy concentrated in ±3 frequency bins around the peak. A perfect sine wave has purity ≈ 1.0.

3. **Candidate Grouping** — Frames exceeding both peakiness and purity thresholds are grouped into contiguous segments. Segments outside the configured duration range are discarded.

4. **Beep vs Speech Discrimination** — Each candidate segment is filtered by multiple criteria to reject speech and musical content:
   - **Energy ratio**: beeps have extreme peak-to-median energy ratios (>50,000× for typical beeps vs <25,000× for speech). Configurable via `--min-energy-ratio`.
   - **Frequency stability**: beeps are constant-frequency; speech formants drift. Candidates with frequency variation exceeding `max_freq_variation_hz` are rejected.
   - **Onset sharpness**: beeps have abrupt onset (energy jumps >10× in a few frames) vs gradual speech transitions.

5. **Confidence Scoring** — Each candidate is scored based on weighted combination of purity (25%), energy ratio (30%), peakiness (15%), onset sharpness (15%), frequency stability (10%), and duration (5%).

6. **Harmonic Detection** — For each confirmed beep, scans for harmonics (2nd through 8th) by checking energy at integer multiples of the fundamental frequency.

### Removal Pipeline

1. **Cascaded Notch Filters** — IIR notch filters (scipy `iirnotch`) at the beep frequency and each harmonic, applied with multiple Q factors (bandwidth settings) across multiple passes. This provides deep, narrow-band suppression.

2. **STFT Spectral Suppression** — Additional frequency-domain processing zeros out ±40 Hz bands around each target frequency, catching any residual energy the notch filters miss.

3. **Crossfade Blending** — The filtered region is blended with the original audio using a smooth crossfade envelope at the boundaries (default 5 ms), preventing click artifacts.

### Visualization

When `--visualize` is used, generates a multi-panel PNG showing:
- Full spectrograms (original vs. cleaned)
- Zoomed spectrograms around each detected beep
- Frequency spectrum overlay at beep timestamps
- Difference plot showing removed energy

## API Usage

The tool can also be used as a Python library:

```python
from beep_cleaner import BeepDetector, BeepRemover, CleanerConfig, visualize
import soundfile as sf
import numpy as np

# Load audio
y, sr = sf.read("input.wav")
y_mono = np.mean(y, axis=1) if y.ndim > 1 else y

# Configure
config = CleanerConfig.from_sensitivity("high")

# Detect
detector = BeepDetector(sr, config)
beeps = detector.detect(y_mono)
for b in beeps:
    print(b)

# Remove
remover = BeepRemover(sr, config)
y_clean = remover.remove(y, beeps)

# Save
sf.write("output.wav", y_clean, sr)

# Optional: visualize
visualize(y, y_clean, sr, beeps, "analysis.png", config)
```

## Limitations

- Designed for **pure-tone beeps** (sinusoidal). Complex sounds like music or speech-like tones may not be detected.
- Very short beeps (<10 ms) may fall below the time resolution of the STFT analysis.
- Beeps overlapping in frequency with dominant speech content may cause slight artifacts in the surrounding audio at the beep frequency.
- Processing time scales with audio duration (~2-5 seconds per minute of audio on a modern machine).

## License

MIT
