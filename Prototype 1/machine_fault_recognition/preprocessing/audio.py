from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from ..config import AudioConfig
from .augmentation import AugmentationPipeline


def _normalize_waveform(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32_768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2_147_483_648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    sample_rate, audio = wavfile.read(path)
    audio = _normalize_waveform(audio)
    if sample_rate != target_sr:
        gcd = int(np.gcd(sample_rate, target_sr))
        audio = signal.resample_poly(audio, target_sr // gcd, sample_rate // gcd).astype(np.float32)
    return audio


def spectral_noise_subtract(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    nperseg = max(2, min(512, len(audio)))
    noverlap = min(max(0, nperseg // 2), nperseg - 1)
    _, _, zxx = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    noise_bins = max(1, int(0.2 * sample_rate / 256))
    noise_profile = np.mean(np.abs(zxx[:, :noise_bins]), axis=1, keepdims=True)
    gain = np.maximum(np.abs(zxx) - 0.9 * noise_profile, 0.01 * np.abs(zxx))
    denoised = signal.istft(gain * np.exp(1j * np.angle(zxx)), fs=sample_rate, nperseg=nperseg, noverlap=noverlap)[1]
    if len(denoised) >= len(audio):
        return denoised[: len(audio)].astype(np.float32)
    output = np.zeros(len(audio), dtype=np.float32)
    output[: len(denoised)] = denoised
    return output


def rms_normalise(audio: np.ndarray, target_lufs: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio**2) + 1e-9))
    target_rms = 10 ** (target_lufs / 20.0)
    return (audio * (target_rms / (rms + 1e-9))).astype(np.float32)


def trim_silence(audio: np.ndarray, threshold: float) -> np.ndarray:
    mask = np.flatnonzero(np.abs(audio) > threshold)
    return audio[mask[0] : mask[-1] + 1] if len(mask) else audio


def pad_or_crop(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        start = (len(audio) - length) // 2
        return audio[start : start + length].astype(np.float32)
    output = np.zeros(length, dtype=np.float32)
    output[: len(audio)] = audio
    return output


@dataclass
class AudioPreprocessor:
    config: AudioConfig
    augmenter: AugmentationPipeline | None = None

    def preprocess(self, source: Path | np.ndarray, *, training: bool = False) -> np.ndarray:
        if isinstance(source, Path):
            audio = load_audio(source, self.config.sample_rate)
        else:
            audio = source.astype(np.float32)

        audio = spectral_noise_subtract(audio, self.config.sample_rate)
        audio = rms_normalise(audio, self.config.rms_target_lufs)
        audio = trim_silence(audio, self.config.trim_threshold)

        if training and self.augmenter is not None:
            audio = self.augmenter.add_noise(audio)
            audio = self.augmenter.time_stretch(audio)
            audio = self.augmenter.pitch_shift(audio)

        return pad_or_crop(audio, int(self.config.sample_rate * self.config.duration_seconds))
