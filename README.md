# Beep Cleaner

A command-line tool for **automatic detection and removal of short pure-tone beeps** from audio files, with support for **real-time streaming** processing. Designed for cleaning up recorded calls, broadcasts, or any audio containing unwanted tonal artifacts (censorship beeps, equipment tones, alert sounds, etc.).

## Features

- **Automatic detection** — finds beeps by analyzing spectral purity, peakiness, and energy ratios; no need to manually specify frequency or timing
- **Harmonic-aware** — detects and removes beep harmonics (up to 8th) for thorough cleaning
- **Multi-stage removal** — cascaded IIR notch filters + STFT spectral suppression for deep attenuation (tested up to 70+ dB reduction)
- **Artifact-free** — crossfade blending at beep boundaries preserves surrounding audio
- **Real-time streaming** — live audio processing via virtual audio cable or loopback device (~43ms latency)
- **Sensitivity presets** — `low`, `medium`, `high` for different use cases
- **Visualization** — generates before/after spectrogram comparison plots
- **Format support** — WAV, OGG, FLAC, and other formats supported by libsndfile

## Installation

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy scipy librosa soundfile matplotlib sounddevice
```

### Dependencies

| Package      | Version | Purpose                          |
|-------------|---------|----------------------------------|
| numpy       | ≥1.24   | Array operations                 |
| scipy       | ≥1.10   | IIR notch filter design          |
| librosa     | ≥0.10   | STFT, frequency analysis         |
| soundfile   | ≥0.12   | Audio I/O (WAV, OGG, FLAC)      |
| sounddevice | ≥0.4    | Real-time audio streaming        |
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

## API Usage

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

## Real-Time Streaming Mode

The real-time mode captures audio from one device, removes beeps on-the-fly, and outputs to another device. This requires a virtual audio cable to route application audio through the cleaner.

### Setup

#### Windows (VB-Audio Virtual Cable)

1. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (free)
2. In your application (e.g., browser, media player), set audio output to **"CABLE Input (VB-Audio Virtual Cable)"**
3. Run the cleaner:

```bash
# List available devices to find the right indices
python beep_cleaner_realtime.py --list-devices

# Route virtual cable → your speakers
python beep_cleaner_realtime.py -i "CABLE Output" -o "Speakers"
```

#### macOS (BlackHole)

1. Install [BlackHole](https://existential.audio/blackhole/): `brew install blackhole-2ch`
2. Open **Audio MIDI Setup** → create a **Multi-Output Device** combining BlackHole + your speakers
3. Set the Multi-Output Device as system output
4. Run the cleaner:

```bash
python beep_cleaner_realtime.py -i "BlackHole 2ch" -o "MacBook Pro Speakers"
```

### Real-Time Options

```
python beep_cleaner_realtime.py --list-devices          # show audio devices
python beep_cleaner_realtime.py -i 3 -o 5               # by device index
python beep_cleaner_realtime.py -i "CABLE" -o "Speaker"  # by name substring
python beep_cleaner_realtime.py -i 3 -o 5 -s high        # high sensitivity
python beep_cleaner_realtime.py -i 3 -o 5 --block-size 1024  # lower latency (~21ms)
```

| Option          | Default | Description                       |
|-----------------|---------|-----------------------------------|
| `--block-size`  | 2048    | Samples per block (~43ms at 48kHz)|
| `--sample-rate` | auto    | Force a specific sample rate      |
| `--sensitivity` | medium  | `low` / `medium` / `high`        |
| `--min-freq`    | 200     | Minimum detection frequency (Hz)  |
| `--max-freq`    | 8000    | Maximum detection frequency (Hz)  |

### Latency

| Block Size | Latency at 48kHz | Latency at 44.1kHz |
|------------|-------------------|--------------------|
| 512        | ~11ms             | ~12ms              |
| 1024       | ~21ms             | ~23ms              |
| 2048       | ~43ms             | ~46ms              |
| 4096       | ~85ms             | ~93ms              |

Smaller blocks = lower latency but higher CPU usage and less frequency resolution.

## Limitations

- Designed for **pure-tone beeps** (sinusoidal). Complex sounds like music or speech-like tones may not be detected.
- Very short beeps (<10 ms) may fall below the time resolution of the STFT analysis.
- Beeps overlapping in frequency with dominant speech content may cause slight artifacts in the surrounding audio at the beep frequency.
- File mode: processing time scales with audio duration (~2-5 seconds per minute of audio).
- Real-time mode: adds ~43ms latency (default block size). Detection requires ~86ms before suppression starts (2-block confirmation).

For technical details on the algorithms, design decisions, and architecture, see [TECHNICAL.md](TECHNICAL.md).

## License

MIT
