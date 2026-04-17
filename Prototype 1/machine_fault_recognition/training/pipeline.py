from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..calibration.temperature import TemperatureScaler
from ..config import AppConfig, resolve_input_dir, resolve_work_dir
from ..data.scanner import DatasetScanner, infer_audio_label
from ..data.splitter import StratifiedSplitter
from ..ensemble.optimizer import optimise_weights
from ..ensemble.voting import confidence_gate, soft_vote
from ..features.handcrafted import HandcraftedFeatureExtractor
from ..features.spectrogram import SpectrogramExtractor
from ..models.branch_a import BranchAModel
from ..models.branch_b import BranchBModel
from ..models.branch_c import BranchCModel
from ..preprocessing.audio import AudioPreprocessor
from ..storage.artifact_store import ArtifactStore
from ..storage.checkpoint import load_pickle, save_pickle
from ..storage.progress import ControlStore, ProgressStore
from ..training.evaluators import evaluate_predictions
from ..utils.logging import banner, info, ok, section
from ..utils.seeding import set_seed


@dataclass
class PipelineState:
    config: AppConfig
    label_names: list[str]
    branch_a: BranchAModel
    branch_b: BranchBModel
    branch_c: BranchCModel
    temp_a: TemperatureScaler
    temp_b: TemperatureScaler
    temp_c: TemperatureScaler
    weights: tuple[float, float, float]
    gate_threshold: float
    mode: str
    history: dict = field(default_factory=dict)


@dataclass
class TrainingPipeline:
    config: AppConfig

    def __post_init__(self) -> None:
        self.input_dir = resolve_input_dir(self.config)
        self.work_dir = resolve_work_dir(self.config)
        self.artifacts = ArtifactStore(self.work_dir)
        self.progress = ProgressStore(self.work_dir)
        self.control = ControlStore(self.work_dir)
        self._preprocessor = AudioPreprocessor(self.config.audio)
        self.spectrogram_extractor = SpectrogramExtractor(self.config.audio)
        self.handcrafted_extractor = HandcraftedFeatureExtractor(self.config.audio)

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

    def scan_dataset(self) -> tuple[pd.DataFrame, str]:
        stage = "scan"
        wav_files = sorted([path for path in self.input_dir.rglob("*.wav") if path.is_file()])
        csv_files = sorted([path for path in self.input_dir.rglob("*.csv") if path.is_file()])
        info(f"Found {len(wav_files)} WAV files | {len(csv_files)} CSV files")

        if csv_files and not wav_files:
            frame, mode = DatasetScanner(self.input_dir).scan()
            self.artifacts.root.mkdir(parents=True, exist_ok=True)
            frame.to_csv(self.artifacts.dataset_manifest_path, index=False)
            state = self.progress.load(stage)
            state.update({"status": "completed", "next_batch": 0, "total_batches": 0, "batch_size": 0})
            self.progress.save(stage, state)
            ok(f"Saved dataset manifest to {self.artifacts.dataset_manifest_path}")
            return frame, mode

        if not wav_files:
            raise RuntimeError(f"No audio or CSV files found under {self.input_dir}")

        batch_size = max(1, int(self.config.training.batch_size))
        total_batches = self._num_batches(len(wav_files), batch_size)
        partial_manifest = self.artifacts.root / "dataset_manifest.partial.csv"

        state = self.progress.load(stage)
        if state.get("total_batches") != total_batches or state.get("batch_size") != batch_size:
            state.update({"next_batch": 0, "retries": {}, "last_error": ""})
        next_batch = int(state.get("next_batch", 0))
        if next_batch == 0 and partial_manifest.exists():
            partial_manifest.unlink()
        if next_batch > 0 and not partial_manifest.exists():
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
        for batch_index, start, end in tqdm(iterator, desc="Scanning batches", unit="batch"):
            self.control.check(stage, self.progress, state)
            attempt = 0
            while True:
                try:
                    rows = [
                        {
                            "path": str(path),
                            "label": infer_audio_label(path, self.input_dir),
                            "sample_id": path.stem,
                        }
                        for path in wav_files[start:end]
                    ]
                    batch_frame = pd.DataFrame(rows)
                    write_header = not partial_manifest.exists()
                    batch_frame.to_csv(partial_manifest, mode="a", header=write_header, index=False)
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

        final_manifest = self.artifacts.dataset_manifest_path
        if final_manifest.exists():
            final_manifest.unlink()
        partial_manifest.rename(final_manifest)

        state["status"] = "completed"
        state["next_batch"] = total_batches
        self.progress.save(stage, state)

        frame = pd.read_csv(final_manifest)
        ok(f"Saved dataset manifest to {final_manifest}")
        return frame, "audio"

    def _build_feature_matrices(
        self,
        frame: pd.DataFrame,
        *,
        training: bool = False,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        spectrogram_features: list[np.ndarray] = []
        handcrafted_features: list[np.ndarray] = []
        iterator = frame["path"].tolist()
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features", unit="file")
        for path_text in iterator:
            audio = self._preprocessor.preprocess(Path(path_text), training=training)
            spectrogram_features.append(self.spectrogram_extractor.features(audio))
            handcrafted_features.append(self.handcrafted_extractor.extract(audio))
        return np.vstack(spectrogram_features).astype(np.float32), np.vstack(handcrafted_features).astype(np.float32)

    def _extract_feature_batch(self, paths: list[str], *, training: bool = False) -> tuple[np.ndarray, np.ndarray]:
        spectrogram_features: list[np.ndarray] = []
        handcrafted_features: list[np.ndarray] = []
        for path_text in paths:
            audio = self._preprocessor.preprocess(Path(path_text), training=training)
            spectrogram_features.append(self.spectrogram_extractor.features(audio))
            handcrafted_features.append(self.handcrafted_extractor.extract(audio))
        return np.vstack(spectrogram_features).astype(np.float32), np.vstack(handcrafted_features).astype(np.float32)

    def extract_features(self, frame: pd.DataFrame, *, training: bool = False) -> tuple[np.ndarray, np.ndarray]:
        stage = "features"
        batch_size = max(1, int(self.config.training.micro_batch or self.config.training.batch_size))
        paths = frame["path"].astype(str).tolist()
        labels = frame["label"].astype(str).tolist()
        total_batches = self._num_batches(len(paths), batch_size)

        batch_dir = self.artifacts.root / "feature_batches"
        batch_dir.mkdir(parents=True, exist_ok=True)

        state = self.progress.load(stage)
        if state.get("total_batches") != total_batches or state.get("batch_size") != batch_size:
            state.update({"next_batch": 0, "retries": {}, "last_error": ""})
        next_batch = int(state.get("next_batch", 0))
        if next_batch > 0:
            expected = batch_dir / f"spec_{next_batch - 1:06d}.npy"
            if not expected.exists():
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
        iterator = list(self._iter_batch_ranges(len(paths), batch_size, next_batch))
        for batch_index, start, end in tqdm(iterator, desc="Extracting feature batches", unit="batch"):
            self.control.check(stage, self.progress, state)
            attempt = 0
            while True:
                try:
                    X_spec_batch, X_hand_batch = self._extract_feature_batch(paths[start:end], training=training)
                    np.save(batch_dir / f"spec_{batch_index:06d}.npy", X_spec_batch)
                    np.save(batch_dir / f"hand_{batch_index:06d}.npy", X_hand_batch)
                    np.save(batch_dir / f"labels_{batch_index:06d}.npy", np.asarray(labels[start:end], dtype=np.str_))
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

        spec_batches: list[np.ndarray] = []
        hand_batches: list[np.ndarray] = []
        for batch_index in range(total_batches):
            spec_batches.append(np.load(batch_dir / f"spec_{batch_index:06d}.npy", allow_pickle=False))
            hand_batches.append(np.load(batch_dir / f"hand_{batch_index:06d}.npy", allow_pickle=False))

        X_spec = np.vstack(spec_batches).astype(np.float32)
        X_hand = np.vstack(hand_batches).astype(np.float32)

        np.savez_compressed(
            self.artifacts.feature_cache_path,
            spectrogram=X_spec,
            handcrafted=X_hand,
            labels=np.asarray(labels, dtype=np.str_),
        )

        state["status"] = "completed"
        state["next_batch"] = total_batches
        self.progress.save(stage, state)
        ok(f"Saved feature cache to {self.artifacts.feature_cache_path}")
        return X_spec, X_hand

    def _load_cached_features(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        cache_path = self.artifacts.feature_cache_path
        if not cache_path.exists():
            return None

        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                required = {"spectrogram", "handcrafted", "labels"}
                if not required.issubset(cache.files):
                    info(f"Ignoring feature cache {cache_path} (missing required keys)")
                    return None
                X_spec = cache["spectrogram"]
                X_hand = cache["handcrafted"]
                cached_labels = cache["labels"].astype(str)
        except Exception as exc:
            info(f"Ignoring feature cache {cache_path} ({exc})")
            return None

        labels = frame["label"].astype(str).to_numpy()
        if len(cached_labels) != len(labels):
            info(f"Ignoring feature cache {cache_path} (row count mismatch)")
            return None
        if not np.array_equal(cached_labels, labels):
            info(f"Ignoring feature cache {cache_path} (label ordering mismatch)")
            return None

        ok(f"Using cached features from {cache_path}")
        return X_spec.astype(np.float32), X_hand.astype(np.float32)

    def train_stage(self) -> PipelineState:
        stage = "train"
        train_state = self.progress.load(stage)
        train_state.update({"status": "running", "steps": train_state.get("steps", {}), "last_error": ""})
        self.progress.save(stage, train_state)

        set_seed(self.config.training.seed)

        banner("Stage 1 · Dataset Scan", f"Input: {self.input_dir}")
        self.control.check(stage, self.progress, train_state)
        frame, mode = self.scan_dataset()
        train_state["steps"]["scan"] = "completed"
        self.progress.save(stage, train_state)
        if mode != "audio":
            raise RuntimeError("This starter pipeline currently targets the audio dataset described in the docs.")

        class_names = sorted(frame["label"].unique().tolist())
        info(f"Classes: {class_names}")

        banner("Stage 2 · Feature Extraction")
        self.control.check(stage, self.progress, train_state)
        cached_features = self._load_cached_features(frame)
        if cached_features is None:
            X_spec, X_hand = self.extract_features(frame, training=False)
        else:
            X_spec, X_hand = cached_features
        train_state["steps"]["features"] = "completed"
        self.progress.save(stage, train_state)
        labels = frame["label"].astype(str).to_numpy()
        label_to_index = {name: index for index, name in enumerate(class_names)}
        y = np.array([label_to_index[label] for label in labels], dtype=np.int64)

        splitter = StratifiedSplitter(
            test_size=self.config.training.test_size,
            validation_size=self.config.training.val_size,
            random_state=self.config.training.seed,
        )
        split = splitter.split(X_spec, y)
        train_idx = split["train_idx"]
        validation_idx = split["validation_idx"]
        test_idx = split["test_idx"]

        x_spec_train = X_spec[train_idx]
        x_spec_validation = X_spec[validation_idx]
        x_spec_test = X_spec[test_idx]
        x_hand_train = X_hand[train_idx]
        x_hand_validation = X_hand[validation_idx]
        x_hand_test = X_hand[test_idx]
        y_train = y[train_idx]
        y_validation = y[validation_idx]
        y_test = y[test_idx]

        banner("Stage 3 · Branch Training")
        checkpoint_dir = self.artifacts.root / "train_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.control.check(stage, self.progress, train_state)
        section("Branch A")
        branch_a_path = checkpoint_dir / "branch_a.pkl"
        if branch_a_path.exists():
            branch_a = load_pickle(branch_a_path)
            info("Loaded Branch A from checkpoint")
        else:
            branch_a = BranchAModel(random_state=self.config.training.seed).fit(x_spec_train, y_train)
            save_pickle(branch_a, branch_a_path)
        train_state["steps"]["branch_a"] = "completed"
        self.progress.save(stage, train_state)

        self.control.check(stage, self.progress, train_state)
        section("Branch B")
        branch_b_path = checkpoint_dir / "branch_b.pkl"
        if branch_b_path.exists():
            branch_b = load_pickle(branch_b_path)
            info("Loaded Branch B from checkpoint")
        else:
            branch_b = BranchBModel(random_state=self.config.training.seed).fit(x_hand_train, y_train)
            save_pickle(branch_b, branch_b_path)
        train_state["steps"]["branch_b"] = "completed"
        self.progress.save(stage, train_state)

        self.control.check(stage, self.progress, train_state)
        section("Branch C")
        branch_c_path = checkpoint_dir / "branch_c.pkl"
        if branch_c_path.exists():
            branch_c = load_pickle(branch_c_path)
            info("Loaded Branch C from checkpoint")
        else:
            branch_c = BranchCModel(random_state=self.config.training.seed).fit(x_hand_train, y_train)
            save_pickle(branch_c, branch_c_path)
        train_state["steps"]["branch_c"] = "completed"
        self.progress.save(stage, train_state)

        section("Validation probabilities")
        pa_validation = branch_a.predict_proba(x_spec_validation)
        pb_validation = branch_b.predict_proba(x_hand_validation)
        pc_validation = branch_c.predict_proba(x_hand_validation)
        pa_test = branch_a.predict_proba(x_spec_test)
        pb_test = branch_b.predict_proba(x_hand_test)
        pc_test = branch_c.predict_proba(x_hand_test)

        section("Temperature scaling")
        calibration_path = checkpoint_dir / "calibration.pkl"
        if calibration_path.exists():
            calibration = load_pickle(calibration_path)
            temp_a = calibration["temp_a"]
            temp_b = calibration["temp_b"]
            temp_c = calibration["temp_c"]
            weights = tuple(calibration["weights"])
            info("Loaded calibration and weights from checkpoint")
        else:
            temp_a = TemperatureScaler().fit(pa_validation, y_validation)
            temp_b = TemperatureScaler().fit(pb_validation, y_validation)
            temp_c = TemperatureScaler().fit(pc_validation, y_validation)
            pa_validation = temp_a.transform(pa_validation)
            pb_validation = temp_b.transform(pb_validation)
            pc_validation = temp_c.transform(pc_validation)
            section("Weight optimisation")
            weights = optimise_weights(pa_validation, pb_validation, pc_validation, y_validation)
            save_pickle(
                {
                    "temp_a": temp_a,
                    "temp_b": temp_b,
                    "temp_c": temp_c,
                    "weights": list(weights),
                },
                calibration_path,
            )

        pa_validation = temp_a.transform(pa_validation)
        pb_validation = temp_b.transform(pb_validation)
        pc_validation = temp_c.transform(pc_validation)
        pa_test = temp_a.transform(pa_test)
        pb_test = temp_b.transform(pb_test)
        pc_test = temp_c.transform(pc_test)

        section("Weight optimisation")
        info(f"Optimal weights: {weights}")
        train_state["steps"]["calibration_and_weights"] = "completed"
        self.progress.save(stage, train_state)

        validation_probs = soft_vote(pa_validation, pb_validation, pc_validation, weights)
        test_probs = soft_vote(pa_test, pb_test, pc_test, weights)
        validation_predictions, validation_sources, validation_confidence = confidence_gate(
            validation_probs,
            pa_validation,
            pb_validation,
            pc_validation,
            self.config.inference.gate_threshold,
        )
        test_predictions, test_sources, test_confidence = confidence_gate(
            test_probs,
            pa_test,
            pb_test,
            pc_test,
            self.config.inference.gate_threshold,
        )

        validation_metrics = evaluate_predictions(y_validation, validation_predictions, class_names)
        test_metrics = evaluate_predictions(y_test, test_predictions, class_names)

        self.artifacts.root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "sample_index": test_idx,
                "pred": test_predictions,
                "label": [class_names[index] for index in test_predictions],
                "confidence": test_confidence,
                "source": test_sources,
            }
        ).to_csv(self.artifacts.predictions_path, index=False)

        with self.artifacts.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mode": mode,
                    "weights": list(weights),
                    "gate_threshold": self.config.inference.gate_threshold,
                    "validation": validation_metrics,
                    "test": test_metrics,
                    "classes": class_names,
                },
                handle,
                indent=2,
            )

        state = PipelineState(
            config=self.config,
            label_names=class_names,
            branch_a=branch_a,
            branch_b=branch_b,
            branch_c=branch_c,
            temp_a=temp_a,
            temp_b=temp_b,
            temp_c=temp_c,
            weights=weights,
            gate_threshold=self.config.inference.gate_threshold,
            mode=mode,
            history={"validation": validation_metrics, "test": test_metrics},
        )

        save_pickle(state, self.artifacts.state_path)
        ok(f"Saved artifacts to {self.artifacts.root}")
        train_state["steps"]["final_artifacts"] = "completed"
        train_state["status"] = "completed"
        self.progress.save(stage, train_state)
        return state

    def run(self) -> PipelineState:
        return self.train_stage()
