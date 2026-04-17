from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..config import AudioConfig


def _resize_time_axis(matrix: np.ndarray, frames: int) -> np.ndarray:
    if matrix.shape[1] == frames:
        return matrix.astype(np.float32)
    return signal.resample(matrix, frames, axis=1).astype(np.float32)


@dataclass
class SpectrogramExtractor:
    config: AudioConfig

    def log_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        nperseg = max(2, min(self.config.n_fft, len(audio)))
        noverlap = min(max(0, nperseg - self.config.hop_length), nperseg - 1)
        _, _, zxx = signal.stft(
            audio,
            fs=self.config.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        return np.log1p(np.abs(zxx)).astype(np.float32)

    def delta(self, matrix: np.ndarray) -> np.ndarray:
        return np.diff(matrix, axis=1, prepend=matrix[:, :1]).astype(np.float32)

    def tensor(self, audio: np.ndarray) -> np.ndarray:
        log_spec = self.log_spectrogram(audio)
        if log_spec.shape[0] < self.config.mel_bins:
            pad_rows = self.config.mel_bins - log_spec.shape[0]
            log_spec = np.pad(log_spec, ((0, pad_rows), (0, 0)))
        log_spec = log_spec[: self.config.mel_bins]
        delta = self.delta(log_spec)
        cqt_like = np.abs(signal.resample(log_spec, log_spec.shape[0], axis=0)).astype(np.float32)
        return np.stack(
            [
                _resize_time_axis(log_spec, self.config.spec_frames),
                _resize_time_axis(cqt_like, self.config.spec_frames),
                _resize_time_axis(delta, self.config.spec_frames),
            ],
            axis=0,
        ).astype(np.float32)

    def features(self, audio: np.ndarray) -> np.ndarray:
        tensor = self.tensor(audio)
        channel_means = tensor.mean(axis=2)
        channel_stds = tensor.std(axis=2)
        channel_max = tensor.max(axis=2)
        channel_min = tensor.min(axis=2)
        return np.concatenate([channel_means, channel_stds, channel_max, channel_min], axis=1).reshape(-1).astype(np.float32)
