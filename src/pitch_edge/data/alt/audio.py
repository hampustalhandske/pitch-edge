"""Stadium-audio momentum index — feature extraction is real, the data feed is not.

Given any audio clip (WAV/FLAC/MP3 via librosa), this extracts the published
crowd-reaction feature set: RMS loudness envelope, spectral flux "roar"
onsets, third-octave-style band shape (mel bands), tonality (spectral
flatness) and a rolling z-scored roar-spike count that can be aligned to
match time as an in-play momentum feature.

What we do *not* have is a licensed live broadcast-audio feed, and we will
not scrape one. So: `extract_momentum_features` works on any clip you are
allowed to process (tested with synthetic audio); the pipeline marks the
feature "architected, data-stubbed" and no model trains on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

implemented_as_feature = False


@dataclass
class AudioMomentum:
    times_s: np.ndarray
    rms_db: np.ndarray
    spectral_flux: np.ndarray
    roar_spikes: np.ndarray  # 1 where a roar onset is detected
    flatness: np.ndarray
    band_shape: np.ndarray  # (n_bands, n_frames) normalised mel band energies

    def momentum_index(self, window_frames: int = 20) -> np.ndarray:
        """Rolling roar-spike density scaled by loudness change — a leading momentum proxy."""
        kernel = np.ones(window_frames) / window_frames
        density = np.convolve(self.roar_spikes, kernel, mode="same")
        loud = np.convolve(np.gradient(self.rms_db), kernel, mode="same")
        return density * (1.0 + np.clip(loud, -1, 1))


def extract_momentum_features(
    audio: np.ndarray | str | Path,
    sr: int = 22050,
    frame_length: int = 2048,
    hop_length: int = 512,
    spike_z: float = 2.5,
    n_bands: int = 24,
) -> AudioMomentum:
    import librosa

    if isinstance(audio, (str, Path)):
        y, sr_loaded = librosa.load(str(audio), sr=sr, mono=True)
        sr = int(sr_loaded)
    else:
        y = np.asarray(audio, dtype=np.float32)

    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=1.0)
    stft = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    flux = np.concatenate([[0.0], np.sqrt(np.sum(np.diff(stft, axis=1).clip(min=0) ** 2, axis=0))])
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    mel = librosa.feature.melspectrogram(S=stft**2, sr=sr, n_mels=n_bands)
    band_shape = mel / (mel.sum(axis=0, keepdims=True) + 1e-10)

    n = min(len(rms_db), len(flux), len(flatness), band_shape.shape[1])
    rms_db, flux, flatness, band_shape = rms_db[:n], flux[:n], flatness[:n], band_shape[:, :n]

    z = (flux - flux.mean()) / (flux.std() + 1e-10)
    spikes = (z > spike_z).astype(float)
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    return AudioMomentum(times, rms_db, flux, spikes, flatness, band_shape)
