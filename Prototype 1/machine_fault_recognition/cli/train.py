from __future__ import annotations

from pathlib import Path

from ..config import load_config
from ..training.pipeline import TrainingPipeline


def scan(config_path: str | Path | None = None):
    config = load_config(config_path)
    pipeline = TrainingPipeline(config)
    return pipeline.scan_dataset()


def features(config_path: str | Path | None = None):
    config = load_config(config_path)
    pipeline = TrainingPipeline(config)
    frame, _ = pipeline.scan_dataset()
    return pipeline.extract_features(frame)


def train(config_path: str | Path | None = None):
    config = load_config(config_path)
    pipeline = TrainingPipeline(config)
    return pipeline.train_stage()
