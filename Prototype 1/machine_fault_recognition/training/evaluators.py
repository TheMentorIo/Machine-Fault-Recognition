from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_predictions(labels: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> dict:
    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted")
    f1_macro = f1_score(labels, predictions, average="macro")
    report = classification_report(labels, predictions, target_names=class_names, zero_division=0)
    matrix = confusion_matrix(labels, predictions)
    return {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "report": report,
        "confusion_matrix": matrix.tolist(),
    }
