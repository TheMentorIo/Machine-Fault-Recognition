from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class AugmentationPipeline:
    noise_std: float = 0.0035
    stretch_min: float = 0.90
    stretch_max: float = 1.10
    pitch_steps: int = 2
    freq_mask_param: int = 20
    time_mask_param: int = 40

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()

    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        return (audio + self._rng.normal(0.0, self.noise_std, size=audio.shape)).astype(np.float32)

    def time_stretch(self, audio: np.ndarray, rate: float | None = None) -> np.ndarray:
        rate = rate or float(self._rng.uniform(self.stretch_min, self.stretch_max))
        target_length = max(1, int(len(audio) / rate))
        return signal.resample(audio, target_length).astype(np.float32)

    def pitch_shift(self, audio: np.ndarray, n_steps: int | None = None) -> np.ndarray:
        steps = n_steps if n_steps is not None else int(self._rng.integers(-self.pitch_steps, self.pitch_steps + 1))
        factor = 2 ** (steps / 12.0)
        stretched = signal.resample(audio, max(1, int(len(audio) / factor)))
        return signal.resample(stretched, len(audio)).astype(np.float32)

    def spec_augment(self, spec: np.ndarray, n_freq_masks: int = 2, n_time_masks: int = 2) -> np.ndarray:
        spec = spec.copy()
        _, freq_bins, time_bins = spec.shape
        for _ in range(n_freq_masks):
            width = int(self._rng.integers(0, self.freq_mask_param))
            start = int(self._rng.integers(0, max(1, freq_bins - width)))
            spec[:, start : start + width, :] = 0.0
        for _ in range(n_time_masks):
            width = int(self._rng.integers(0, self.time_mask_param))
            start = int(self._rng.integers(0, max(1, time_bins - width)))
            spec[:, :, start : start + width] = 0.0
        return spec
