from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class StratifiedSplitter:
    test_size: float = 0.15
    validation_size: float = 0.15
    random_state: int = 42

    def split(self, features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        indices = np.arange(len(labels))
        train_idx, temp_idx, y_train, y_temp = train_test_split(
            indices,
            labels,
            test_size=self.test_size + self.validation_size,
            stratify=labels,
            random_state=self.random_state,
        )
        relative_validation = self.validation_size / (self.test_size + self.validation_size)
        validation_idx, test_idx, y_validation, y_test = train_test_split(
            temp_idx,
            y_temp,
            test_size=1 - relative_validation,
            stratify=y_temp,
            random_state=self.random_state,
        )
        return {
            "train_idx": train_idx,
            "validation_idx": validation_idx,
            "test_idx": test_idx,
            "y_train": y_train,
            "y_validation": y_validation,
            "y_test": y_test,
        }
