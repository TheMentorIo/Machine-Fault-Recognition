from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        probabilities = np.clip(probabilities, 1e-9, 1.0)
        logits = np.log(probabilities)
        best_temperature = 1.0
        best_loss = float("inf")
        for candidate in np.linspace(0.3, 3.0, 40):
            scaled = logits / candidate
            scaled -= scaled.max(axis=1, keepdims=True)
            calibrated = np.exp(scaled)
            calibrated /= calibrated.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(calibrated[np.arange(len(labels)), labels] + 1e-9))
            if loss < best_loss:
                best_loss = loss
                best_temperature = float(candidate)
        self.temperature = best_temperature
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(probabilities, 1e-9, 1.0)
        logits = np.log(probabilities) / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        calibrated = np.exp(logits)
        return calibrated / calibrated.sum(axis=1, keepdims=True)
