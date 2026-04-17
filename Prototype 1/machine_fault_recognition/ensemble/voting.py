from __future__ import annotations

from typing import List, Tuple

import numpy as np


def soft_vote(
    branch_a: np.ndarray,
    branch_b: np.ndarray,
    branch_c: np.ndarray,
    weights: Tuple[float, float, float],
) -> np.ndarray:
    weights_array = np.array(weights, dtype=np.float32)
    weights_array /= weights_array.sum()
    combined = weights_array[0] * branch_a + weights_array[1] * branch_b + weights_array[2] * branch_c
    return combined / combined.sum(axis=1, keepdims=True)


def confidence_gate(
    ensemble_probs: np.ndarray,
    branch_a: np.ndarray,
    branch_b: np.ndarray,
    branch_c: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, List[str], np.ndarray]:
    predictions: list[int] = []
    sources: list[str] = []
    confidences: list[float] = []
    for index in range(len(ensemble_probs)):
        ensemble_confidence = float(np.max(ensemble_probs[index]))
        if ensemble_confidence >= threshold:
            predictions.append(int(np.argmax(ensemble_probs[index])))
            sources.append("ensemble")
            confidences.append(ensemble_confidence)
            continue

        branch_confidences = {
            "A": float(np.max(branch_a[index])),
            "B": float(np.max(branch_b[index])),
            "C": float(np.max(branch_c[index])),
        }
        selected_branch = max(branch_confidences, key=branch_confidences.get)
        branch_probabilities = {"A": branch_a, "B": branch_b, "C": branch_c}[selected_branch]
        predictions.append(int(np.argmax(branch_probabilities[index])))
        sources.append(f"branch_{selected_branch}")
        confidences.append(branch_confidences[selected_branch])
    return np.array(predictions), sources, np.array(confidences, dtype=np.float32)
