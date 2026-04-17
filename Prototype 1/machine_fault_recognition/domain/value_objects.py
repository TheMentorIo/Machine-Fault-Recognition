from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSplit:
    train: str = "train"
    validation: str = "validation"
    test: str = "test"
