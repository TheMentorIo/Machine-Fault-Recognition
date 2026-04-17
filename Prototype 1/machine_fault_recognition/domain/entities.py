from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioSample:
    path: Path
    label: str
    waveform: np.ndarray | None = None


@dataclass(frozen=True)
class Prediction:
    sample_path: Path
    label_index: int
    confidence: float
    source: str
