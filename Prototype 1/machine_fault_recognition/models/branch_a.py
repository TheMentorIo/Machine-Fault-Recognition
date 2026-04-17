from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseBranchModel


@dataclass
class BranchAModel(BaseBranchModel):
    max_components: int = 64
    random_state: int = 42
    pipeline: Pipeline = field(init=False)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "BranchAModel":
        n_components = max(1, min(self.max_components, features.shape[0] - 1, features.shape[1]))
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=self.random_state)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.pipeline.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(features)
