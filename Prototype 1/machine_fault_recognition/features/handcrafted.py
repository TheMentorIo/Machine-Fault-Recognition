from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

from ..config import AudioConfig


@dataclass
class HandcraftedFeatureExtractor:
    config: AudioConfig

    def extract(self, audio: np.ndarray) -> np.ndarray:
        eps = 1e-9
        nperseg = max(2, min(512, len(audio)))
        noverlap = min(max(0, nperseg // 2), nperseg - 1)
        freqs, _, zxx = signal.stft(audio, fs=self.config.sample_rate, nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(zxx)
        centroid = (freqs[:, None] * magnitude).sum(0) / (magnitude.sum(0) + eps)
        roll_idx = np.argmax(
            np.cumsum(magnitude, axis=0) >= 0.85 * (magnitude.sum(0, keepdims=True) + eps), axis=0
        )
        rolloff = freqs[np.clip(roll_idx, 0, len(freqs) - 1)]
        flux = np.sqrt(np.sum(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]) ** 2, axis=0) + eps)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(audio).astype(np.int8)))))
        rms = float(np.sqrt(np.mean(audio**2) + eps))

        time_domain = np.array(
            [
                float(np.mean(audio)),
                float(np.std(audio)),
                float(np.min(audio)),
                float(np.max(audio)),
                zcr,
                rms,
                float(stats.skew(audio)),
                float(stats.kurtosis(audio)),
            ],
            dtype=np.float32,
        )

        spectral = np.array(
            [
                float(np.mean(centroid)),
                float(np.std(centroid)),
                float(np.mean(rolloff)),
                float(np.std(rolloff)),
                float(np.mean(flux)),
                float(np.std(flux)),
            ],
            dtype=np.float32,
        )

        log_mag = np.log1p(magnitude)
        sub_bands = np.array_split(log_mag, 8, axis=0)
        subband_stats = np.array(
            [value for band in sub_bands for value in (float(np.mean(band)), float(np.std(band)))],
            dtype=np.float32,
        )

        mel_like_bands = np.array_split(log_mag, 20, axis=0)
        mel_like_stats = np.array(
            [
                value
                for band in mel_like_bands
                for value in (float(np.mean(band)), float(np.std(band)), float(stats.skew(band.ravel())))
            ],
            dtype=np.float32,
        )

        return np.nan_to_num(np.concatenate([time_domain, spectral, subband_stats, mel_like_stats]), copy=False).astype(np.float32)
