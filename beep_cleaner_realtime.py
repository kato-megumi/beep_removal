#!/usr/bin/env python3
"""
beep_cleaner_realtime.py — Real-time audio beep detection and removal.

Captures audio from an input device, detects and removes pure-tone beeps
in real-time, and plays the cleaned audio through an output device.

Typical setup (Windows with VB-Audio Virtual Cable):
  1. Set your application's audio output to "CABLE Input (VB-Audio)"
  2. Run this script with the virtual cable as input and your speakers as output
  3. Beeps are removed in real-time from the audio stream

Usage:
    python beep_cleaner_realtime.py --list-devices
    python beep_cleaner_realtime.py -i 3 -o 5
    python beep_cleaner_realtime.py -i "CABLE Output" -o "Speakers"
    python beep_cleaner_realtime.py -i 3 -o 5 --sensitivity high
"""

from __future__ import annotations

import argparse
import signal as signal_mod
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
from scipy import signal


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RealtimeConfig:
    """Configuration for real-time beep cleaning."""
    # Detection
    min_freq: float = 200.0
    max_freq: float = 8000.0
    purity_threshold: float = 0.7       # spectral purity to trigger suppression
    energy_ratio_threshold: float = 50000.0  # peak-to-median ratio in FFT

    # Suppression
    notch_q_values: list[float] = field(default_factory=lambda: [5, 10, 20])
    suppression_gain: float = 0.01      # multiply beep band by this (= -40 dB)

    # Streaming
    block_size: int = 2048              # samples per processing block
    n_fft: int = 2048                   # FFT size (= block_size for zero latency)
    lookahead_blocks: int = 2           # blocks to buffer for detection confirmation
    sample_rate: int | None = None      # None = use device default
    channels: int = 1                   # mono processing (mixed down internally if stereo)

    # Beep tracking
    min_beep_blocks: int = 2            # minimum consecutive blocks to confirm beep
    release_blocks: int = 3             # blocks to keep suppressing after beep ends

    @classmethod
    def from_sensitivity(cls, level: str) -> "RealtimeConfig":
        presets = {
            "low": cls(
                purity_threshold=0.9,
                energy_ratio_threshold=500000.0,
                notch_q_values=[10, 20],
                suppression_gain=0.1,
                min_beep_blocks=3,
            ),
            "medium": cls(
                purity_threshold=0.7,
                energy_ratio_threshold=50000.0,
                notch_q_values=[5, 10, 20],
                suppression_gain=0.01,
                min_beep_blocks=2,
            ),
            "high": cls(
                purity_threshold=0.5,
                energy_ratio_threshold=20000.0,
                notch_q_values=[3, 5, 10, 20],
                suppression_gain=0.001,
                min_beep_blocks=1,
            ),
        }
        if level not in presets:
            raise ValueError(f"Sensitivity must be one of: {list(presets.keys())}")
        return presets[level]


# ---------------------------------------------------------------------------
# Real-time beep detector + suppressor
# ---------------------------------------------------------------------------

class RealtimeBeepProcessor:
    """
    Streaming beep detector and suppressor.

    Processes audio block-by-block with a small lookahead buffer.
    Uses per-block FFT for detection and IIR notch filters for removal.
    """

    def __init__(self, sr: int, config: RealtimeConfig | None = None):
        self.sr = sr
        self.cfg = config or RealtimeConfig()
        self.n_fft = self.cfg.n_fft

        # Pre-compute frequency bins
        self.freqs = np.fft.rfftfreq(self.n_fft, 1 / sr)
        self.freq_mask = (self.freqs >= self.cfg.min_freq) & (self.freqs <= self.cfg.max_freq)
        self.freq_indices = np.where(self.freq_mask)[0]

        # Beep tracking state
        self._beep_active = False
        self._beep_freq: float = 0.0
        self._consecutive_beep_blocks = 0
        self._release_counter = 0

        # Pre-built notch filter cache: {freq_hz: [(b, a), ...]}
        self._filter_cache: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

        # IIR filter states per frequency per Q — for stateful filtering
        # {(freq_hz, q_idx): zi} or {(freq_hz, q_idx, pass_i): zi}
        self._filter_states: dict[tuple[int, ...], np.ndarray] = {}

        # Lookahead buffer
        self._buffer: deque[np.ndarray] = deque(
            maxlen=self.cfg.lookahead_blocks + 1
        )

        # Lookback buffer for retroactive onset suppression
        # Stores (block, was_candidate) for the last N blocks
        self._lookback: deque[tuple[np.ndarray, bool, float]] = deque(
            maxlen=max(self.cfg.min_beep_blocks + 1, 3)
        )
        self._output_buffer: deque[np.ndarray] = deque()

        # Statistics
        self.stats = {
            "blocks_processed": 0,
            "beeps_detected": 0,
            "blocks_suppressed": 0,
        }

        # Hann window for analysis
        self._window = np.hanning(self.n_fft)

    def _get_notch_filters(self, freq: float) -> list[tuple[np.ndarray, np.ndarray]]:
        """Get or create cached notch filters for a frequency."""
        freq_key = round(freq)
        if freq_key not in self._filter_cache:
            filters = []
            for q in self.cfg.notch_q_values:
                b, a = signal.iirnotch(freq, q, self.sr)
                filters.append((b, a))
            self._filter_cache[freq_key] = filters
        return self._filter_cache[freq_key]

    def _detect_tone(self, block: np.ndarray) -> tuple[bool, float, float, float]:
        """
        Analyze a single block for pure-tone content.

        Returns (is_tone, frequency, purity, energy_ratio).
        """
        # Apply window and compute FFT (zero-pad 4x for finer freq resolution)
        n = len(block)
        zp_nfft = self.n_fft * 4
        if n < self.n_fft:
            windowed = np.zeros(self.n_fft)
            windowed[:n] = block[:n] * self._window[:n]
        else:
            windowed = block[:self.n_fft] * self._window
        spectrum = np.abs(np.fft.rfft(windowed, n=zp_nfft))

        # Frequency bins for zero-padded FFT
        freqs_zp = np.fft.rfftfreq(zp_nfft, 1.0 / self.sr)

        # Find peak in target frequency band (using zero-padded bins)
        band_mask = (freqs_zp >= self.cfg.min_freq) & (freqs_zp <= self.cfg.max_freq)
        band_indices = np.where(band_mask)[0]
        band_spectrum = spectrum[band_indices]
        if len(band_spectrum) == 0 or np.max(band_spectrum) == 0:
            return False, 0.0, 0.0, 0.0

        peak_local_idx = np.argmax(band_spectrum)
        peak_global_idx = band_indices[peak_local_idx]
        peak_mag = spectrum[peak_global_idx]

        # Parabolic interpolation for sub-bin frequency accuracy
        if 0 < peak_global_idx < len(spectrum) - 1:
            alpha = float(spectrum[peak_global_idx - 1])
            beta = float(spectrum[peak_global_idx])
            gamma = float(spectrum[peak_global_idx + 1])
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-10:
                p = 0.5 * (alpha - gamma) / denom
            else:
                p = 0.0
            peak_freq = (peak_global_idx + p) * self.sr / zp_nfft
        else:
            peak_freq = freqs_zp[peak_global_idx]

        # Spectral purity: energy in ±12 bins (equivalent to ±3 at 1x) / total
        purity_radius = 12  # 4x zero-pad means 4x more bins per Hz
        lo = max(0, peak_global_idx - purity_radius)
        hi = min(len(spectrum), peak_global_idx + purity_radius + 1)
        total_energy = np.sum(spectrum ** 2) + 1e-10
        peak_energy = np.sum(spectrum[lo:hi] ** 2)
        purity = peak_energy / total_energy

        # Energy ratio: peak to median
        median_mag = np.median(spectrum) + 1e-10
        energy_ratio = peak_mag / median_mag

        is_tone = bool(
            purity >= self.cfg.purity_threshold
            and energy_ratio >= self.cfg.energy_ratio_threshold
        )
        return is_tone, float(peak_freq), float(purity), float(energy_ratio)

    def _suppress_tone(self, block: np.ndarray, freq: float) -> np.ndarray:
        """Apply cascaded notch filters (multiple passes) to remove a tone."""
        out = block.copy()
        filters = self._get_notch_filters(freq)
        num_passes = 3

        for pass_i in range(num_passes):
            for q_idx, (b, a) in enumerate(filters):
                key = (round(freq), q_idx, pass_i)
                if key not in self._filter_states:
                    self._filter_states[key] = signal.lfilter_zi(b, a) * out[0]
                out, self._filter_states[key] = signal.lfilter(b, a, out, zi=self._filter_states[key])

        # Also suppress harmonics
        for harmonic in range(2, 6):
            h_freq = freq * harmonic
            if h_freq >= self.sr / 2:
                break
            h_filters = self._get_notch_filters(h_freq)
            for pass_i in range(num_passes):
                for q_idx, (b, a) in enumerate(h_filters):
                    key = (round(h_freq), q_idx, pass_i)
                    if key not in self._filter_states:
                        self._filter_states[key] = signal.lfilter_zi(b, a) * out[0]
                    out, self._filter_states[key] = signal.lfilter(b, a, out, zi=self._filter_states[key])

        return out

    def _suppress_tone_spectral(self, block: np.ndarray, freq: float) -> np.ndarray:
        """
        Stateless FFT-domain suppression for retroactive lookback blocks.

        Zeros out the beep frequency and its harmonics in the spectrum,
        then inverse-FFTs. No filter state needed.
        """
        n = len(block)
        spectrum = np.fft.rfft(block)
        fft_freqs = np.fft.rfftfreq(n, 1.0 / self.sr)
        suppression = self.cfg.suppression_gain

        # Suppress fundamental and harmonics
        for harmonic in range(1, 6):
            h_freq = freq * harmonic
            if h_freq >= self.sr / 2:
                break
            # Zero out bins within ±bandwidth of the target frequency
            bandwidth = max(30.0, h_freq / 10)  # Hz
            mask = np.abs(fft_freqs - h_freq) < bandwidth
            spectrum[mask] *= suppression

        return np.fft.irfft(spectrum, n=n)

    def _reset_filter_states(self) -> None:
        """Clear filter states when beep ends."""
        self._filter_states.clear()

    def process_block(self, block: np.ndarray) -> np.ndarray:
        """
        Process a single audio block.

        Uses a 1-block delay to allow retroactive suppression of the beep
        onset block when the beep is confirmed in the following block.

        Parameters
        ----------
        block : np.ndarray
            1-D mono audio block.

        Returns
        -------
        np.ndarray
            Processed audio block (same length). On the very first call,
            returns silence while the lookback buffer fills.
        """
        self.stats["blocks_processed"] += 1

        # Detect tone in current block
        is_tone, freq, purity, energy_ratio = self._detect_tone(block)

        if is_tone:
            self._consecutive_beep_blocks += 1
            self._beep_freq = freq
            self._release_counter = self.cfg.release_blocks
        else:
            self._consecutive_beep_blocks = 0
            if self._release_counter > 0:
                self._release_counter -= 1
            else:
                if self._beep_active:
                    self._beep_active = False
                    self._reset_filter_states()

        # Confirm beep after minimum consecutive blocks
        just_confirmed = False
        if self._consecutive_beep_blocks >= self.cfg.min_beep_blocks:
            if not self._beep_active:
                self._beep_active = True
                self.stats["beeps_detected"] += 1
                just_confirmed = True

        # If we just confirmed a beep, retroactively suppress lookback blocks
        if just_confirmed:
            suppressed_lookback: deque[tuple[np.ndarray, bool, float]] = deque(
                maxlen=self._lookback.maxlen
            )
            for lb_block, lb_was_candidate, lb_freq in self._lookback:
                # Use stateless spectral suppression for lookback (no state issues)
                suppressed = self._suppress_tone_spectral(lb_block, self._beep_freq)
                suppressed_lookback.append((suppressed, True, self._beep_freq))
            self._lookback = suppressed_lookback

        # Process current block: IIR notch + spectral for deeper suppression
        if self._beep_active or self._release_counter > 0:
            self.stats["blocks_suppressed"] += 1
            output = self._suppress_tone(block, self._beep_freq)
            output = self._suppress_tone_spectral(output, self._beep_freq)
        else:
            output = block.copy()

        # Push to lookback and return the oldest block (1-block delay)
        # Save the oldest block BEFORE appending (deque auto-evicts at maxlen)
        if len(self._lookback) == self._lookback.maxlen:
            oldest, _, _ = self._lookback[0]
            self._lookback.append((output, is_tone, freq))
            return oldest
        else:
            self._lookback.append((output, is_tone, freq))
            # Buffer not full yet — return silence while it fills
            return np.zeros_like(block)

    def flush(self) -> list[np.ndarray]:
        """Return all remaining buffered blocks after stream ends."""
        remaining = [block for block, _, _ in self._lookback]
        self._lookback.clear()
        return remaining


# ---------------------------------------------------------------------------
# Stream manager
# ---------------------------------------------------------------------------

class RealtimeBeepCleaner:
    """
    Manages the audio stream: captures from input device, processes,
    and plays to output device.
    """

    def __init__(
        self,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        config: RealtimeConfig | None = None,
    ):
        self.cfg = config or RealtimeConfig()
        self.input_device = input_device
        self.output_device = output_device

        # Resolve sample rate
        if self.cfg.sample_rate is None:
            info = sd.query_devices(input_device, 'input')
            self.sr = int(info['default_samplerate'])
        else:
            self.sr = self.cfg.sample_rate

        self.processor = RealtimeBeepProcessor(self.sr, self.cfg)
        self._running = False
        self._stream: sd.Stream | None = None
        self._lock = threading.Lock()

        # Status display
        self._last_status = ""
        self._status_interval = 0.5  # seconds
        self._last_status_time = 0.0

    def _audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice for each audio block."""
        if status:
            print(f"  [stream] {status}", file=sys.stderr)

        # Mix to mono for processing
        if indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata[:, 0]

        # Process
        with self._lock:
            cleaned = self.processor.process_block(mono)

        # Output (duplicate to all output channels)
        for ch in range(outdata.shape[1]):
            outdata[:, ch] = cleaned

    def start(self) -> None:
        """Start the real-time audio stream."""
        in_info = sd.query_devices(self.input_device, 'input')
        out_info = sd.query_devices(self.output_device, 'output')
        in_channels = min(int(in_info['max_input_channels']), 2)
        out_channels = min(int(out_info['max_output_channels']), 2)

        print(f"Input:  [{self.input_device}] {in_info['name']}")
        print(f"Output: [{self.output_device}] {out_info['name']}")
        print(f"Sample rate: {self.sr} Hz | Block size: {self.cfg.block_size}")
        latency_ms = self.cfg.block_size / self.sr * 1000
        print(f"Latency: ~{latency_ms:.0f}ms per block")
        print()

        self._stream = sd.Stream(
            samplerate=self.sr,
            blocksize=self.cfg.block_size,
            device=(self.input_device, self.output_device),
            channels=(in_channels, out_channels),
            dtype='float32',
            callback=self._audio_callback,
            latency='low',
        )
        self._running = True
        self._stream.start()
        print("Streaming... Press Ctrl+C to stop.\n")

    def stop(self) -> None:
        """Stop the audio stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def print_status(self) -> None:
        """Print current status line."""
        now = time.time()
        if now - self._last_status_time < self._status_interval:
            return
        self._last_status_time = now

        stats = self.processor.stats
        active = self.processor._beep_active
        freq = self.processor._beep_freq

        status = (
            f"\r  Blocks: {stats['blocks_processed']:>6d} | "
            f"Beeps found: {stats['beeps_detected']:>3d} | "
            f"Suppressed: {stats['blocks_suppressed']:>6d}"
        )
        if active:
            status += f" | ACTIVE: {freq:.0f}Hz"
        else:
            status += " |               "

        if status != self._last_status:
            print(status, end="", flush=True)
            self._last_status = status

    def run(self) -> None:
        """Run the stream until interrupted."""
        self.start()
        try:
            while self._running:
                self.print_status()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            print("\n\nStopping...")
            self.stop()
            stats = self.processor.stats
            print(f"Total blocks: {stats['blocks_processed']}")
            print(f"Beeps detected: {stats['beeps_detected']}")
            print(f"Blocks suppressed: {stats['blocks_suppressed']}")


# ---------------------------------------------------------------------------
# Device listing
# ---------------------------------------------------------------------------

def list_devices() -> None:
    """Print all available audio devices."""
    print(sd.query_devices())
    print()

    default_in = sd.query_devices(kind='input')
    default_out = sd.query_devices(kind='output')
    print(f"Default input:  {default_in['name']}")
    print(f"Default output: {default_out['name']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="beep_cleaner_realtime",
        description="Real-time audio beep detection and removal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
setup (Windows with VB-Audio Virtual Cable):
  1. Install VB-Audio Virtual Cable: https://vb-audio.com/Cable/
  2. Set your app's audio output to "CABLE Input (VB-Audio Virtual Cable)"
  3. Run: %(prog)s -i "CABLE Output" -o "Speakers"

setup (macOS with BlackHole):
  1. Install BlackHole: brew install blackhole-2ch
  2. Create a Multi-Output Device in Audio MIDI Setup
  3. Run: %(prog)s -i "BlackHole 2ch" -o "MacBook Pro Speakers"

examples:
  %(prog)s --list-devices
  %(prog)s -i 3 -o 5
  %(prog)s -i "CABLE Output" -o "Speakers" --sensitivity high
  %(prog)s -i 3 -o 5 --block-size 1024
""",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available audio devices and exit",
    )
    p.add_argument(
        "-i", "--input-device",
        help="Input device index (int) or name substring (str)",
    )
    p.add_argument(
        "-o", "--output-device",
        help="Output device index (int) or name substring (str)",
    )
    p.add_argument(
        "-s", "--sensitivity", choices=["low", "medium", "high"], default="medium",
        help="Detection sensitivity (default: medium)",
    )
    p.add_argument(
        "--block-size", type=int, default=2048,
        help="Audio block size in samples (default: 2048, ~43ms at 48kHz)",
    )
    p.add_argument(
        "--sample-rate", type=int,
        help="Sample rate in Hz (default: device default)",
    )
    p.add_argument(
        "--min-freq", type=float,
        help="Minimum beep frequency in Hz (default: 200)",
    )
    p.add_argument(
        "--max-freq", type=float,
        help="Maximum beep frequency in Hz (default: 8000)",
    )
    return p


def _parse_device(value: str | None) -> int | str | None:
    """Parse device argument as int index or string name."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    # Build config
    config = RealtimeConfig.from_sensitivity(args.sensitivity)
    config.block_size = args.block_size
    config.n_fft = args.block_size
    if args.sample_rate is not None:
        config.sample_rate = args.sample_rate
    if args.min_freq is not None:
        config.min_freq = args.min_freq
    if args.max_freq is not None:
        config.max_freq = args.max_freq

    input_dev = _parse_device(args.input_device)
    output_dev = _parse_device(args.output_device)

    cleaner = RealtimeBeepCleaner(
        input_device=input_dev,
        output_device=output_dev,
        config=config,
    )

    # Handle SIGINT gracefully
    def _sigint_handler(sig: int, frame: object) -> None:
        cleaner.stop()
        sys.exit(0)

    signal_mod.signal(signal_mod.SIGINT, _sigint_handler)

    try:
        cleaner.run()
    except sd.PortAudioError as e:
        print(f"Audio error: {e}", file=sys.stderr)
        print("\nRun with --list-devices to see available devices.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
