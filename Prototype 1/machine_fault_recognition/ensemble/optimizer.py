from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score

from .voting import soft_vote

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


def optimise_weights(
    branch_a: np.ndarray,
    branch_b: np.ndarray,
    branch_c: np.ndarray,
    labels: np.ndarray,
    initial: Tuple[float, float, float] = (0.40, 0.40, 0.20),
) -> Tuple[float, float, float]:
    def objective(weights: np.ndarray) -> float:
        weights = np.maximum(weights, 1e-6)
        weights = weights / weights.sum()
        predictions = np.argmax(soft_vote(branch_a, branch_b, branch_c, tuple(weights.tolist())), axis=1)
        return -accuracy_score(labels, predictions)

    if minimize is None:
        return initial

    result = minimize(
        objective,
        x0=np.array(initial, dtype=np.float32),
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-4},
    )
    weights = np.maximum(result.x, 1e-6)
    weights /= weights.sum()
    return tuple(weights.tolist())
