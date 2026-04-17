from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseBranchModel


@dataclass
class BranchBModel(BaseBranchModel):
    random_state: int = 42
    pipeline: Pipeline = field(init=False)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "BranchBModel":
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-4,
                        max_iter=300,
                        early_stopping=True,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.pipeline.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(features)
