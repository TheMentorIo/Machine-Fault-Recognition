from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 3
    best_score: float | None = None
    bad_epochs: int = 0

    def update(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
