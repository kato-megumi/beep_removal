#!/usr/bin/env python3
"""
beep_cleaner.py — Generic audio beep detection and removal tool.

Automatically detects short pure-tone beeps in audio files and removes them
while preserving the surrounding audio content. Supports any sample rate and
common audio formats (WAV, OGG, FLAC, MP3 via soundfile/librosa).

Usage:
    python beep_cleaner.py input.ogg
    python beep_cleaner.py input.wav -o cleaned.wav
    python beep_cleaner.py input.ogg --min-freq 500 --max-freq 2000
    python beep_cleaner.py input.ogg --sensitivity high --visualize
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Beep:
    """A detected beep event."""
    start_time: float        # seconds
    end_time: float          # seconds
    frequency: float         # dominant frequency in Hz
    harmonics: list[float] = field(default_factory=list)  # harmonic frequencies
    confidence: float = 0.0  # 0-1 detection confidence
    purity: float = 0.0      # spectral purity (1 = pure tone)
    energy_ratio: float = 0.0  # peak energy vs median

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def __str__(self) -> str:
        harmonics_str = ""
        if self.harmonics:
            harmonics_str = f", harmonics: {', '.join(f'{h:.0f}Hz' for h in self.harmonics)}"
        return (
            f"Beep @ {self.start_time:.3f}s–{self.end_time:.3f}s "
            f"({self.duration_ms:.0f}ms) | "
            f"{self.frequency:.0f}Hz | "
            f"confidence={self.confidence:.2f}, purity={self.purity:.2f}"
            f"{harmonics_str}"
        )


@dataclass
class CleanerConfig:
    """Configuration for the beep cleaner."""
    # Detection parameters
    min_freq: float = 200.0   # minimum beep frequency (Hz)
    max_freq: float = 8000.0  # maximum beep frequency (Hz)
    min_duration_ms: float = 10.0   # minimum beep duration (ms)
    max_duration_ms: float = 2000.0  # maximum beep duration (ms)
    min_purity: float = 0.3   # minimum spectral purity (0–1)
    min_confidence: float = 0.5  # minimum detection confidence
    min_energy_ratio: float = 50000.0  # minimum peak-to-median energy ratio
    max_freq_variation_hz: float = 30.0  # max frequency drift within a beep

    # Removal parameters
    notch_q_values: list[float] = field(default_factory=lambda: [5, 10, 20])
    n_passes: int = 3
    spectral_suppression_db: float = -40.0  # dB suppression in spectral domain
    spectral_band_hz: float = 40.0          # bandwidth for spectral suppression
    crossfade_ms: float = 5.0               # crossfade at edges
    margin_ms: float = 10.0                 # extra margin around detected beep

    # Analysis parameters
    n_fft: int = 4096
    hop_length_detect: int = 128   # fine resolution for detection
    hop_length_stft: int = 512     # for spectrograms

    @classmethod
    def from_sensitivity(cls, level: str) -> "CleanerConfig":
        """Create config from a sensitivity preset."""
        presets = {
            "low": cls(
                min_purity=0.5,
                min_confidence=0.7,
                min_energy_ratio=80000.0,
                max_freq_variation_hz=20.0,
                notch_q_values=[10, 20],
                n_passes=1,
                spectral_suppression_db=-20.0,
            ),
            "medium": cls(
                min_purity=0.3,
                min_confidence=0.5,
                min_energy_ratio=50000.0,
                max_freq_variation_hz=30.0,
                notch_q_values=[5, 10, 20],
                n_passes=2,
                spectral_suppression_db=-30.0,
            ),
            "high": cls(
                min_purity=0.2,
                min_confidence=0.3,
                min_energy_ratio=30000.0,
                max_freq_variation_hz=50.0,
                notch_q_values=[3, 5, 10, 20],
                n_passes=3,
                spectral_suppression_db=-40.0,
            ),
        }
        if level not in presets:
            raise ValueError(f"Sensitivity must be one of: {list(presets.keys())}")
        return presets[level]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class BeepDetector:
    """Detect pure-tone beeps in audio signals."""

    def __init__(self, sr: int, config: CleanerConfig | None = None):
        self.sr = sr
        self.config = config or CleanerConfig()

    def detect(self, y: np.ndarray) -> list[Beep]:
        """
        Detect beeps in a mono audio signal.

        Parameters
        ----------
        y : np.ndarray
            1-D mono audio signal.

        Returns
        -------
        list[Beep]
            Detected beep events sorted by time.
        """
        cfg = self.config

        # Compute high-resolution STFT
        S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length_detect))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=cfg.n_fft)
        times = librosa.frames_to_time(
            np.arange(S.shape[1]), sr=self.sr, hop_length=cfg.hop_length_detect
        )

        # Restrict frequency range
        freq_mask = (freqs >= cfg.min_freq) & (freqs <= cfg.max_freq)
        freq_indices = np.where(freq_mask)[0]
        if len(freq_indices) == 0:
            return []

        # Per-frame metrics
        S_band = S[freq_indices, :]
        peak_in_band = np.max(S_band, axis=0)
        median_full = np.median(S, axis=0) + 1e-10
        peakiness = peak_in_band / median_full

        # Dominant frequency per frame (within our band)
        peak_bin_local = np.argmax(S_band, axis=0)
        peak_freqs = freqs[freq_indices[peak_bin_local]]

        # Spectral purity per frame: energy in ±3 bins around peak / total
        purity = np.zeros(S.shape[1])
        for i in range(S.shape[1]):
            global_peak_idx = freq_indices[peak_bin_local[i]]
            lo = max(0, global_peak_idx - 3)
            hi = min(S.shape[0], global_peak_idx + 4)
            total = np.sum(S[:, i] ** 2) + 1e-10
            purity[i] = np.sum(S[lo:hi, i] ** 2) / total

        # Adaptive threshold for peakiness
        peakiness_threshold = np.percentile(peakiness, 95)
        peakiness_threshold = max(peakiness_threshold, 50.0)  # absolute minimum

        # Candidate frames: high peakiness AND reasonable purity
        candidate_mask = (
            (peakiness > peakiness_threshold)
            & (purity > cfg.min_purity)
        )
        candidate_frames = np.where(candidate_mask)[0]

        if len(candidate_frames) == 0:
            return []

        # Group consecutive candidate frames into segments
        segments = self._group_frames(candidate_frames, gap_frames=5)

        # Evaluate each segment
        beeps: list[Beep] = []
        for seg_start_idx, seg_end_idx in segments:
            seg_frames = candidate_frames[seg_start_idx:seg_end_idx + 1]
            t_start = times[seg_frames[0]]
            t_end = times[seg_frames[-1]]
            duration_ms = (t_end - t_start) * 1000

            # Duration filter
            if duration_ms < cfg.min_duration_ms or duration_ms > cfg.max_duration_ms:
                continue

            # Dominant frequency (median across segment for stability)
            seg_peak_freqs = peak_freqs[seg_frames]
            dom_freq = float(np.median(seg_peak_freqs))

            # Frequency stability — beeps have near-constant frequency
            freq_std = float(np.std(seg_peak_freqs))
            if freq_std > cfg.max_freq_variation_hz:
                continue  # too much frequency drift = likely speech

            # Average purity and peakiness in segment
            avg_purity = float(np.mean(purity[seg_frames]))
            avg_peakiness = float(np.mean(peakiness[seg_frames]))

            # Find harmonics and energy ratio early (used in confidence)
            seg_spectrum = np.mean(S[:, seg_frames], axis=1)
            dom_idx = np.argmin(np.abs(freqs - dom_freq))
            energy_ratio = float(seg_spectrum[dom_idx] / (np.median(seg_spectrum) + 1e-10))

            # Energy ratio filter — beeps have extreme peak-to-median ratio
            if energy_ratio < cfg.min_energy_ratio:
                continue

            # Onset sharpness — beeps have abrupt onset vs gradual speech
            # Compare energy in beep band just before and during the segment
            onset_sharpness = 0.0
            if seg_frames[0] > 5:
                pre_energy = float(np.mean(peakiness[seg_frames[0]-5:seg_frames[0]]))
                seg_energy = float(np.mean(peakiness[seg_frames[:5]]))
                onset_sharpness = seg_energy / (pre_energy + 1e-10)

            # Confidence score based on multiple factors
            norm_peakiness = min(avg_peakiness / (peakiness_threshold * 10), 1.0)
            norm_energy = min(energy_ratio / 100000.0, 1.0)
            norm_onset = min(onset_sharpness / 10.0, 1.0)
            freq_stability = max(0, 1.0 - freq_std / cfg.max_freq_variation_hz)

            confidence = (
                avg_purity * 0.25
                + norm_peakiness * 0.15
                + norm_energy * 0.30
                + norm_onset * 0.15
                + freq_stability * 0.10
                + min(duration_ms / 50, 1.0) * 0.05
            )

            if confidence < cfg.min_confidence:
                continue

            harmonics = self._find_harmonics(seg_spectrum, freqs, dom_freq)

            beeps.append(Beep(
                start_time=t_start,
                end_time=t_end,
                frequency=dom_freq,
                harmonics=harmonics,
                confidence=confidence,
                purity=avg_purity,
                energy_ratio=energy_ratio,
            ))

        return sorted(beeps, key=lambda b: b.start_time)

    def _group_frames(
        self, frames: np.ndarray, gap_frames: int = 5
    ) -> list[tuple[int, int]]:
        """Group frame indices into contiguous segments."""
        if len(frames) == 0:
            return []
        segments = []
        seg_start = 0
        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] > gap_frames:
                segments.append((seg_start, i - 1))
                seg_start = i
        segments.append((seg_start, len(frames) - 1))
        return segments

    def _find_harmonics(
        self, spectrum: np.ndarray, freqs: np.ndarray, fundamental: float,
        max_harmonic: int = 8,
    ) -> list[float]:
        """Find harmonics of the fundamental in the spectrum."""
        median_energy = np.median(spectrum)
        harmonics = []
        for n in range(2, max_harmonic + 1):
            h_freq = fundamental * n
            if h_freq >= self.sr / 2:
                break
            h_idx = np.argmin(np.abs(freqs - h_freq))
            if spectrum[h_idx] > median_energy * 3:
                harmonics.append(float(freqs[h_idx]))
        return harmonics


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

class BeepRemover:
    """Remove detected beeps from audio signals."""

    def __init__(self, sr: int, config: CleanerConfig | None = None):
        self.sr = sr
        self.config = config or CleanerConfig()

    def remove(self, y: np.ndarray, beeps: list[Beep]) -> np.ndarray:
        """
        Remove beeps from an audio signal.

        Parameters
        ----------
        y : np.ndarray
            Audio signal (1-D mono or 2-D multi-channel, shape: samples or samples×channels).
        beeps : list[Beep]
            Beeps to remove.

        Returns
        -------
        np.ndarray
            Cleaned audio signal (same shape as input).
        """
        if not beeps:
            return y.copy()

        cfg = self.config
        y_clean = y.copy()
        is_multichannel = y.ndim > 1

        for beep in beeps:
            y_clean = self._remove_single(y_clean, beep, is_multichannel, cfg)

        return y_clean

    def _remove_single(
        self,
        y: np.ndarray,
        beep: Beep,
        is_multichannel: bool,
        cfg: CleanerConfig,
    ) -> np.ndarray:
        """Remove a single beep from the signal."""
        margin_s = cfg.margin_ms / 1000.0
        crossfade_samples = int(cfg.crossfade_ms / 1000.0 * self.sr)

        start_sample = int(max(0, (beep.start_time - margin_s) * self.sr))
        end_sample = int(min(len(y), (beep.end_time + margin_s) * self.sr))

        # Extend for crossfade
        ext_start = max(0, start_sample - crossfade_samples)
        ext_end = min(len(y), end_sample + crossfade_samples)

        # Extract region
        if is_multichannel:
            region = y[ext_start:ext_end, :].copy()
        else:
            region = y[ext_start:ext_end].copy()

        # Notch frequencies: fundamental + harmonics
        notch_freqs = [beep.frequency] + beep.harmonics

        # Stage 1: Cascaded IIR notch filters
        filtered = region.copy()
        for _ in range(cfg.n_passes):
            for nf in notch_freqs:
                if nf >= self.sr / 2:
                    continue
                for q in cfg.notch_q_values:
                    b, a = signal.iirnotch(nf, q, self.sr)
                    if is_multichannel:
                        for ch in range(region.shape[1]):
                            filtered[:, ch] = signal.filtfilt(b, a, filtered[:, ch])
                    else:
                        filtered = signal.filtfilt(b, a, filtered)

        # Stage 2: STFT spectral suppression
        suppression_linear = 10 ** (cfg.spectral_suppression_db / 20.0)
        process_channels = (
            [filtered[:, ch] for ch in range(filtered.shape[1])]
            if is_multichannel
            else [filtered]
        )
        result_channels = []
        stft_n_fft = min(cfg.n_fft, len(process_channels[0]))
        # Ensure even n_fft
        stft_n_fft = max(stft_n_fft - (stft_n_fft % 2), 16)
        for ch_data in process_channels:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                stft = librosa.stft(ch_data, n_fft=stft_n_fft, hop_length=128)
            stft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=stft_n_fft)
            for nf in notch_freqs:
                band_mask = (
                    (stft_freqs >= nf - cfg.spectral_band_hz)
                    & (stft_freqs <= nf + cfg.spectral_band_hz)
                )
                stft[band_mask, :] *= suppression_linear
            result_channels.append(
                librosa.istft(stft, hop_length=128, length=len(ch_data))
            )

        if is_multichannel:
            filtered = np.column_stack(result_channels)
        else:
            filtered = result_channels[0]

        # Crossfade blending
        total_len = ext_end - ext_start
        envelope = np.ones(total_len)
        if crossfade_samples > 0 and total_len > 2 * crossfade_samples:
            envelope[:crossfade_samples] = np.linspace(0, 1, crossfade_samples)
            envelope[-crossfade_samples:] = np.linspace(1, 0, crossfade_samples)

        if is_multichannel:
            envelope = envelope[:, np.newaxis]

        inv_envelope = 1 - envelope
        original_region = y[ext_start:ext_end].copy()
        blended = original_region * inv_envelope + filtered * envelope

        y_out = y.copy()
        y_out[ext_start:ext_end] = blended
        return y_out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(
    y_orig: np.ndarray,
    y_clean: np.ndarray,
    sr: int,
    beeps: list[Beep],
    output_path: str = "beep_analysis.png",
    config: CleanerConfig | None = None,
) -> None:
    """Generate a spectrum comparison plot of original vs cleaned audio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = config or CleanerConfig()

    # Mono versions
    y_orig_mono = np.mean(y_orig, axis=1) if y_orig.ndim > 1 else y_orig
    y_clean_mono = np.mean(y_clean, axis=1) if y_clean.ndim > 1 else y_clean

    n_fft = cfg.n_fft
    hop = cfg.hop_length_stft

    S_orig = np.abs(librosa.stft(y_orig_mono, n_fft=n_fft, hop_length=hop))
    S_clean = np.abs(librosa.stft(y_clean_mono, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S_orig.shape[1]), sr=sr, hop_length=hop)

    S_orig_db = librosa.amplitude_to_db(S_orig, ref=np.max)
    S_clean_db = librosa.amplitude_to_db(S_clean, ref=np.max)

    n_beeps = len(beeps)
    fig, axes = plt.subplots(2 + n_beeps, 2, figsize=(18, 6 + 5 * n_beeps))
    fig.suptitle("Beep Cleaner — Spectrum Analysis", fontsize=16, fontweight="bold")

    # Row 0: Full spectrograms
    for col, (S_db, label) in enumerate(
        [(S_orig_db, "Original"), (S_clean_db, "Cleaned")]
    ):
        ax = axes[0, col]
        img = ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma", vmin=-80, vmax=0)
        ax.set_title(f"{label} — Full Spectrogram")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, min(cfg.max_freq * 1.5, sr / 2))
        for beep in beeps:
            ax.axvline(beep.start_time, color="cyan", ls="--", alpha=0.6, lw=0.8)
            ax.axvline(beep.end_time, color="cyan", ls="--", alpha=0.6, lw=0.8)
        plt.colorbar(img, ax=ax, label="dB")

    # Per-beep rows
    for b_idx, beep in enumerate(beeps):
        row = 1 + b_idx
        pad = 0.3  # seconds padding around beep
        t_lo = max(0, beep.start_time - pad)
        t_hi = min(times[-1], beep.end_time + pad)
        zoom_mask = (times >= t_lo) & (times <= t_hi)
        zoom_t = times[zoom_mask]

        # Col 0: Zoomed spectrogram (original vs cleaned overlay region)
        for col, (S_db, label, cmap) in enumerate([
            (S_orig_db, "Original", "magma"),
            (S_clean_db, "Cleaned", "magma"),
        ]):
            ax = axes[row, col]
            ax.pcolormesh(zoom_t, freqs, S_db[:, zoom_mask], shading="auto", cmap=cmap, vmin=-80, vmax=0)
            ax.set_title(f"{label} — Beep {b_idx + 1} ({beep.frequency:.0f}Hz, {beep.duration_ms:.0f}ms)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_ylim(0, min(beep.frequency * 5, sr / 2))
            ax.axvline(beep.start_time, color="cyan", ls="--", alpha=0.8, lw=1.5)
            ax.axvline(beep.end_time, color="cyan", ls="--", alpha=0.8, lw=1.5)
            ax.axhline(beep.frequency, color="lime", ls=":", alpha=0.6, label=f"{beep.frequency:.0f}Hz")
            for h in beep.harmonics:
                ax.axhline(h, color="lime", ls=":", alpha=0.3)
            ax.legend(loc="upper right", fontsize=7)

    # Last row: Frequency spectrum overlay and difference
    # Use the first (or most confident) beep for the spectrum comparison
    ref_beep = max(beeps, key=lambda b: b.confidence)
    beep_mask = (times >= ref_beep.start_time) & (times <= ref_beep.end_time)

    orig_spectrum_db = 20 * np.log10(np.mean(S_orig[:, beep_mask], axis=1) + 1e-10)
    clean_spectrum_db = 20 * np.log10(np.mean(S_clean[:, beep_mask], axis=1) + 1e-10)

    ax_spec = axes[-1, 0]
    ax_spec.plot(freqs, orig_spectrum_db, color="#ff6b6b", lw=1.2, alpha=0.8, label="Original")
    ax_spec.plot(freqs, clean_spectrum_db, color="#4ecdc4", lw=1.2, alpha=0.8, label="Cleaned")
    ax_spec.set_title(f"Frequency Spectrum at Beep ({ref_beep.start_time:.2f}s–{ref_beep.end_time:.2f}s)")
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Magnitude (dB)")
    ax_spec.set_xlim(0, min(cfg.max_freq * 1.5, sr / 2))
    for nf in [ref_beep.frequency] + ref_beep.harmonics:
        ax_spec.axvline(nf, color="orange", ls="--", alpha=0.4)
    ax_spec.legend()
    ax_spec.grid(True, alpha=0.3)

    ax_diff = axes[-1, 1]
    diff = orig_spectrum_db - clean_spectrum_db
    ax_diff.fill_between(freqs, 0, diff, where=(diff > 0), color="#ff6b6b", alpha=0.6, label="Removed")
    ax_diff.fill_between(freqs, 0, diff, where=(diff < 0), color="#4ecdc4", alpha=0.6, label="Added")
    ax_diff.plot(freqs, diff, color="#333", lw=0.8)
    ax_diff.set_title("Difference (Original − Cleaned)")
    ax_diff.set_xlabel("Frequency (Hz)")
    ax_diff.set_ylabel("dB Difference")
    ax_diff.set_xlim(0, min(cfg.max_freq * 1.5, sr / 2))
    for nf in [ref_beep.frequency] + ref_beep.harmonics:
        ax_diff.axvline(nf, color="orange", ls="--", alpha=0.4)
    ax_diff.legend()
    ax_diff.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="beep_cleaner",
        description="Detect and remove short pure-tone beeps from audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s recording.wav
  %(prog)s call.ogg -o call_clean.ogg
  %(prog)s audio.flac --sensitivity high --visualize
  %(prog)s audio.wav --min-freq 800 --max-freq 1200 --min-duration 20
""",
    )
    p.add_argument("input", help="Input audio file path")
    p.add_argument("-o", "--output", help="Output file path (default: <input>_clean.<ext>)")
    p.add_argument(
        "-s", "--sensitivity", choices=["low", "medium", "high"], default="medium",
        help="Detection sensitivity preset (default: medium)",
    )
    p.add_argument("--min-freq", type=float, help="Minimum beep frequency in Hz (default: 200)")
    p.add_argument("--max-freq", type=float, help="Maximum beep frequency in Hz (default: 8000)")
    p.add_argument("--min-duration", type=float, help="Minimum beep duration in ms (default: 10)")
    p.add_argument("--max-duration", type=float, help="Maximum beep duration in ms (default: 2000)")
    p.add_argument("--min-purity", type=float, help="Minimum spectral purity 0–1 (default: 0.3)")
    p.add_argument("--min-confidence", type=float, help="Minimum confidence 0–1 (default: 0.5)")
    p.add_argument("--min-energy-ratio", type=float, help="Minimum peak-to-median energy ratio (default: 50000)")
    p.add_argument("--n-passes", type=int, help="Number of filter passes (default: 3)")
    p.add_argument(
        "--visualize", action="store_true",
        help="Generate spectrum analysis plot",
    )
    p.add_argument(
        "--viz-output", default="beep_analysis.png",
        help="Visualization output path (default: beep_analysis.png)",
    )
    p.add_argument("--dry-run", action="store_true", help="Detect beeps but don't remove them")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    # Build config
    config = CleanerConfig.from_sensitivity(args.sensitivity)
    if args.min_freq is not None:
        config.min_freq = args.min_freq
    if args.max_freq is not None:
        config.max_freq = args.max_freq
    if args.min_duration is not None:
        config.min_duration_ms = args.min_duration
    if args.max_duration is not None:
        config.max_duration_ms = args.max_duration
    if args.min_purity is not None:
        config.min_purity = args.min_purity
    if args.min_confidence is not None:
        config.min_confidence = args.min_confidence
    if args.min_energy_ratio is not None:
        config.min_energy_ratio = args.min_energy_ratio
    if args.n_passes is not None:
        config.n_passes = args.n_passes

    # Load audio
    print(f"Loading: {input_path}")
    y, sr = sf.read(str(input_path))
    duration = len(y) / sr
    is_multichannel = y.ndim > 1
    channels = y.shape[1] if is_multichannel else 1
    print(f"  Sample rate: {sr} Hz | Duration: {duration:.2f}s | Channels: {channels}")

    # Detect
    y_mono = np.mean(y, axis=1) if is_multichannel else y
    detector = BeepDetector(sr, config)
    beeps = detector.detect(y_mono)

    if not beeps:
        print("\nNo beeps detected.")
        return 0

    print(f"\nDetected {len(beeps)} beep(s):")
    for i, beep in enumerate(beeps, 1):
        print(f"  {i}. {beep}")
        if args.verbose:
            print(f"     energy_ratio={beep.energy_ratio:.1f}")

    if args.dry_run:
        print("\n(Dry run — no files modified)")
        return 0

    # Remove
    print("\nRemoving beeps...")
    remover = BeepRemover(sr, config)
    y_clean = remover.remove(y, beeps)

    # Verify
    for beep in beeps:
        start = int(beep.start_time * sr)
        end = int(beep.end_time * sr)
        n = min(config.n_fft, end - start)
        if n < 16:
            continue
        orig_fft = np.abs(np.fft.rfft(y_mono[start:end], n=n))
        clean_mono = np.mean(y_clean, axis=1) if is_multichannel else y_clean
        clean_fft = np.abs(np.fft.rfft(clean_mono[start:end], n=n))
        fft_freqs = np.fft.rfftfreq(n, 1 / sr)
        beep_idx = np.argmin(np.abs(fft_freqs - beep.frequency))
        if orig_fft[beep_idx] > 0:
            reduction = 20 * np.log10(clean_fft[beep_idx] / (orig_fft[beep_idx] + 1e-10) + 1e-10)
            print(f"  {beep.frequency:.0f}Hz: {-reduction:.1f} dB reduction")

    # Save
    output_path = args.output
    if output_path is None:
        output_path = str(input_path.with_stem(input_path.stem + "_clean"))
    print(f"\nSaving: {output_path}")
    sf.write(output_path, y_clean, sr)

    # Visualize
    if args.visualize:
        print("Generating visualization...")
        visualize(y, y_clean, sr, beeps, args.viz_output, config)

    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
