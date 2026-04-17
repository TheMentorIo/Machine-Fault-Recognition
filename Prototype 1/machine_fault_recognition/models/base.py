from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pickle
import numpy as np


class BaseBranchModel(ABC):
    @abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray) -> "BaseBranchModel":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "BaseBranchModel":
        with path.open("rb") as handle:
            return pickle.load(handle)
