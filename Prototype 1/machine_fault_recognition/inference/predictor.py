from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..config import AppConfig, resolve_work_dir
from ..ensemble.voting import confidence_gate, soft_vote
from ..features.handcrafted import HandcraftedFeatureExtractor
from ..features.spectrogram import SpectrogramExtractor
from ..preprocessing.audio import AudioPreprocessor
from ..storage.checkpoint import load_pickle
from ..storage.progress import ControlStore, ProgressStore
from ..training.pipeline import PipelineState
from ..utils.logging import ok


@dataclass
class Predictor:
    state: PipelineState

    def __post_init__(self) -> None:
        self.preprocessor = AudioPreprocessor(self.state.config.audio)
        self.spectrogram_extractor = SpectrogramExtractor(self.state.config.audio)
        self.handcrafted_extractor = HandcraftedFeatureExtractor(self.state.config.audio)
        work_dir = resolve_work_dir(self.state.config)
        self.progress = ProgressStore(work_dir)
        self.control = ControlStore(work_dir)

    @staticmethod
    def _num_batches(total_items: int, batch_size: int) -> int:
        if total_items == 0:
            return 0
        return (total_items + batch_size - 1) // batch_size

    def _iter_batch_ranges(self, total_items: int, batch_size: int, start_batch: int = 0):
        total_batches = self._num_batches(total_items, batch_size)
        for batch_index in range(start_batch, total_batches):
            start = batch_index * batch_size
            end = min(total_items, start + batch_size)
            yield batch_index, start, end

    @classmethod
    def from_path(cls, path: Path) -> "Predictor":
        return cls(load_pickle(path))

    def _extract_features(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        audio = self.preprocessor.preprocess(path, training=False)
        return self.spectrogram_extractor.features(audio), self.handcrafted_extractor.extract(audio)

    def predict_directory(self, test_dir: Path, results_path: Path, time_path: Path) -> pd.DataFrame:
        stage = "infer"
        wav_files = sorted(
            [path for path in test_dir.glob("*.wav") if path.is_file()],
            key=lambda path: int(path.stem) if path.stem.isdigit() else 0,
        )
        if not wav_files:
            raise FileNotFoundError(f"No WAV files found in {test_dir}")

        batch_size = max(1, int(self.state.config.training.micro_batch or self.state.config.training.batch_size))
        total_batches = self._num_batches(len(wav_files), batch_size)
        state = self.progress.load(stage)
        if state.get("total_batches") != total_batches or state.get("batch_size") != batch_size:
            state.update({"next_batch": 0, "retries": {}, "last_error": ""})

        partial_results = results_path.parent / f"{results_path.name}.partial"
        partial_time = time_path.parent / f"{time_path.name}.partial"
        next_batch = int(state.get("next_batch", 0))
        if next_batch == 0:
            if partial_results.exists():
                partial_results.unlink()
            if partial_time.exists():
                partial_time.unlink()
        if next_batch > 0 and (not partial_results.exists() or not partial_time.exists()):
            state.update({"next_batch": 0, "retries": {}, "last_error": ""})
            next_batch = 0

        state.update(
            {
                "status": "running",
                "batch_size": batch_size,
                "total_batches": total_batches,
                "next_batch": next_batch,
            }
        )
        self.progress.save(stage, state)

        max_retries = 3
        iterator = list(self._iter_batch_ranges(len(wav_files), batch_size, next_batch))
        for batch_index, start_idx, end_idx in tqdm(iterator, desc="Running inference batches", unit="batch"):
            self.control.check(stage, self.progress, state)
            attempt = 0
            while True:
                try:
                    batch_predictions: list[int] = []
                    batch_runtimes: list[float] = []
                    for wav_path in wav_files[start_idx:end_idx]:
                        start = perf_counter()
                        spec_features, hand_features = self._extract_features(wav_path)
                        pa = self.state.branch_a.predict_proba(spec_features[None, :])
                        pb = self.state.branch_b.predict_proba(hand_features[None, :])
                        pc = self.state.branch_c.predict_proba(hand_features[None, :])
                        pa = self.state.temp_a.transform(pa)
                        pb = self.state.temp_b.transform(pb)
                        pc = self.state.temp_c.transform(pc)
                        probabilities = soft_vote(pa, pb, pc, self.state.weights)
                        prediction, _, _ = confidence_gate(
                            probabilities,
                            pa,
                            pb,
                            pc,
                            self.state.gate_threshold,
                        )
                        batch_predictions.append(int(prediction[0]))
                        batch_runtimes.append(round(perf_counter() - start, 3))

                    partial_results.parent.mkdir(parents=True, exist_ok=True)
                    with partial_results.open("a", encoding="utf-8") as handle:
                        handle.write("\n".join(str(prediction) for prediction in batch_predictions) + "\n")
                    with partial_time.open("a", encoding="utf-8") as handle:
                        handle.write("\n".join(f"{runtime:.3f}" for runtime in batch_runtimes) + "\n")

                    state["next_batch"] = batch_index + 1
                    state["last_error"] = ""
                    self.progress.save(stage, state)
                    break
                except Exception as exc:
                    attempt += 1
                    retries = state.setdefault("retries", {})
                    retries[str(batch_index)] = attempt
                    state["last_error"] = str(exc)
                    self.progress.save(stage, state)
                    if attempt >= max_retries:
                        state["status"] = "failed"
                        self.progress.save(stage, state)
                        raise

        results_path.parent.mkdir(parents=True, exist_ok=True)
        if results_path.exists():
            results_path.unlink()
        if time_path.exists():
            time_path.unlink()
        partial_results.rename(results_path)
        partial_time.rename(time_path)

        state["status"] = "completed"
        state["next_batch"] = total_batches
        self.progress.save(stage, state)

        prediction_lines = [line.strip() for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        runtime_lines = [line.strip() for line in time_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        predictions = [int(value) for value in prediction_lines]
        runtimes = [float(value) for value in runtime_lines]

        ok(f"Saved {len(predictions)} predictions to {results_path}")
        return pd.DataFrame({"path": [str(path) for path in wav_files], "prediction": predictions, "runtime": runtimes})
