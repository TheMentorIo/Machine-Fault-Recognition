from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)
