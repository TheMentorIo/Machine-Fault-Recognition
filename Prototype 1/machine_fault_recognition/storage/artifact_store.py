from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ArtifactStore:
    root: Path

    @property
    def dataset_manifest_path(self) -> Path:
        return self.root / "dataset_manifest.csv"

    @property
    def feature_cache_path(self) -> Path:
        return self.root / "feature_cache.npz"

    @property
    def predictions_path(self) -> Path:
        return self.root / "predictions.csv"

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.json"

    @property
    def state_path(self) -> Path:
        return self.root / "pipeline_state.pkl"

    @property
    def results_path(self) -> Path:
        return self.root / "results.txt"

    @property
    def time_path(self) -> Path:
        return self.root / "time.txt"
