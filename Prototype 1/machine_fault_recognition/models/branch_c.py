from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base import BaseBranchModel


@dataclass
class BranchCModel(BaseBranchModel):
    random_state: int = 42
    _rf: Pipeline = field(init=False)
    _gb: Pipeline = field(init=False)
    _svm: Pipeline = field(init=False)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "BranchCModel":
        self._rf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_features="sqrt",
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self._gb = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(random_state=self.random_state),
                ),
            ]
        )
        self._svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(C=5.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=self.random_state),
                ),
            ]
        )
        self._rf.fit(features, labels)
        self._gb.fit(features, labels)
        self._svm.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        rf = self._rf.predict_proba(features)
        gb = self._gb.predict_proba(features)
        svm = self._svm.predict_proba(features)
        weights = np.array([0.40, 0.35, 0.25], dtype=np.float32)
        combined = weights[0] * rf + weights[1] * gb + weights[2] * svm
        return combined / combined.sum(axis=1, keepdims=True)
