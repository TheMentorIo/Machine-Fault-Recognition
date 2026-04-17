from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


FeatureExtractor = Callable[[np.ndarray], np.ndarray]


@dataclass
class FeatureRegistry:
    _extractors: dict[str, FeatureExtractor] = field(default_factory=dict)

    def register(self, name: str, extractor: FeatureExtractor) -> None:
        self._extractors[name] = extractor

    def get(self, name: str) -> FeatureExtractor:
        return self._extractors[name]
