from __future__ import annotations

from pathlib import Path

from ..config import load_config
from ..inference.predictor import Predictor
from ..storage.checkpoint import load_pickle


def infer(config_path: str | Path | None = None, state_path: str | Path | None = None, test_dir: str | Path | None = None):
    config = load_config(config_path)
    state_path = Path(state_path or config.paths.model_path)
    test_dir = Path(test_dir or config.paths.input_dir)
    predictor = Predictor(load_pickle(state_path))
    return predictor.predict_directory(test_dir, config.paths.output_dir / config.inference.results_filename, config.paths.output_dir / config.inference.time_filename)
