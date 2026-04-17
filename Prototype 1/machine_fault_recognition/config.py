from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    duration_seconds: float = 4.0
    rms_target_lufs: float = -23.0
    trim_threshold: float = 1e-3
    n_fft: int = 1024
    hop_length: int = 256
    mel_bins: int = 128
    spec_frames: int = 256


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    batch_size: int = 8
    micro_batch: int = 8
    num_workers: int = 0
    max_epochs: int = 1
    early_stopping_patience: int = 3


@dataclass(frozen=True)
class InferenceConfig:
    gate_threshold: float = 0.60
    results_filename: str = "results.txt"
    time_filename: str = "time.txt"


@dataclass(frozen=True)
class PathsConfig:
    input_dir: Path = Path("Student")
    work_dir: Path = Path("artifacts")
    model_path: Path = Path("artifacts/pipeline_state.pkl")
    output_dir: Path = Path("artifacts/output")


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _build_paths(raw: dict[str, Any] | None) -> PathsConfig:
    raw = raw or {}
    return PathsConfig(
        input_dir=Path(raw.get("input_dir", "Student")),
        work_dir=Path(raw.get("work_dir", "artifacts")),
        model_path=Path(raw.get("model_path", "artifacts/pipeline_state.pkl")),
        output_dir=Path(raw.get("output_dir", "artifacts/output")),
    )


def load_config(config_path: str | Path | None = None) -> AppConfig:
    if config_path is None:
        config_path = Path("configs/default.toml")
    else:
        config_path = Path(config_path)

    if tomllib is None or not config_path.exists():
        return AppConfig()

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    audio = data.get("audio", {})
    training = data.get("training", {})
    inference = data.get("inference", {})
    paths = data.get("paths", {})

    return AppConfig(
        audio=AudioConfig(**audio),
        training=TrainingConfig(**training),
        inference=InferenceConfig(**inference),
        paths=_build_paths(paths),
    )


def resolve_input_dir(config: AppConfig) -> Path:
    candidates = []
    dataset_dir = os.environ.get("DATASET_DIR")
    if dataset_dir:
        candidates.append(Path(dataset_dir))
    candidates.extend(
        [
            config.paths.input_dir,
            Path("Student"),
            Path("data"),
            Path("/kaggle/input"),
            Path("."),
        ]
    )
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return Path(".")


def resolve_work_dir(config: AppConfig) -> Path:
    work_dir = Path(os.environ.get("WORK_DIR", config.paths.work_dir))
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir
