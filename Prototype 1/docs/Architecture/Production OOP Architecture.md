# Production OOP Architecture

This document defines a clean, production-quality object-oriented architecture for the Machine Fault Recognition project. The goal is to keep the current research pipeline intact while separating responsibilities into small, testable, replaceable components.

## Design Goals

- Keep each class focused on a single responsibility.
- Minimize coupling between preprocessing, feature extraction, modeling, calibration, and inference.
- Make configuration explicit and injectable.
- Keep training and inference code reusable outside the notebook.
- Preserve the current 3-branch ensemble design, but make it easier to maintain and extend.

## Recommended Package Layout

```text
machine_fault_recognition/
├── pyproject.toml
├── README.md
├── configs/
│   ├── default.yaml
│   └── kaggle.yaml
├── data/
│   ├── __init__.py
│   ├── contracts.py
│   ├── scanner.py
│   ├── splitter.py
│   └── repositories.py
├── domain/
│   ├── __init__.py
│   ├── entities.py
│   └── value_objects.py
├── preprocessing/
│   ├── __init__.py
│   ├── audio_loader.py
│   ├── denoising.py
│   ├── normalization.py
│   ├── trimming.py
│   ├── padding.py
│   ├── augmentation.py
│   └── pipeline.py
├── features/
│   ├── __init__.py
│   ├── spectrogram.py
│   ├── handcrafted.py
│   └── registry.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── branch_a.py
│   ├── branch_b.py
│   ├── branch_c.py
│   └── factory.py
├── calibration/
│   ├── __init__.py
│   └── temperature_scaler.py
├── ensemble/
│   ├── __init__.py
│   ├── voting.py
│   └── optimizer.py
├── inference/
│   ├── __init__.py
│   ├── predictor.py
│   └── batcher.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── evaluators.py
│   └── callbacks.py
├── storage/
│   ├── __init__.py
│   ├── checkpoint.py
│   └── artifact_store.py
├── utils/
│   ├── __init__.py
│   ├── logging.py
│   ├── metrics.py
│   ├── seeding.py
│   └── device.py
└── cli/
    ├── __init__.py
    ├── train.py
    └── infer.py
```

## Core OOP Building Blocks

### 1. Domain Layer

Keep project concepts here and avoid framework dependencies.

- `AudioSample`: immutable representation of a waveform plus metadata.
- `FaultLabel`: label contract for the six-class problem.
- `DatasetSplit`: train, validation, and test identities.

This layer should not know about PyTorch, scikit-learn, or notebooks.

### 2. Preprocessing Layer

Use a pipeline composed of small processors.

- `AudioLoader` reads WAV files and returns normalized arrays.
- `NoiseReducer` performs spectral subtraction or a future Cython-backed variant.
- `Normalizer` applies RMS/loudness normalization.
- `SilenceTrimmer` removes leading and trailing silence.
- `PadCropper` standardizes clip length.
- `AugmentationPipeline` applies training-only transformations.
- `PreprocessingPipeline` composes the above steps in order.

Suggested interface:

```python
class PreprocessingStep(Protocol):
    def process(self, audio: np.ndarray, *, training: bool = False) -> np.ndarray:
        ...
```

### 3. Feature Extraction Layer

Separate feature extraction from datasets and models.

- `SpectrogramExtractor` builds log-mel, CQT, and delta channels.
- `HandcraftedFeatureExtractor` computes statistical features.
- `FeatureRegistry` exposes a common lookup for extractors.

Each extractor should implement a stable interface such as `extract(audio) -> np.ndarray`.

### 4. Data Access Layer

This layer translates files into clean training inputs.

- `DatasetScanner` discovers audio or CSV sources.
- `LabelEncoderService` stores label mapping and inverse mapping.
- `TrainValTestSplitter` performs stratified or group-aware splitting.
- `AudioDataset` and `TabularDataset` adapt data to the training framework.

Keep file system logic here, not in model or training classes.

### 5. Model Layer

Each branch gets its own class with a clear contract.

- `BaseBranchModel` defines `fit`, `predict_proba`, `save`, and `load`.
- `BranchAModel` wraps EfficientNet-B2 or a fallback CNN.
- `BranchBModel` wraps the Conformer-style neural network.
- `BranchCModel` wraps the classical ensemble stack.
- `ModelFactory` creates branch objects from configuration.

Use interfaces to avoid hard dependencies between branches and orchestration code.

### 6. Calibration Layer

- `TemperatureScaler` calibrates one branch at a time.
- `CalibrationService` coordinates fitting and transforming branch outputs.

This layer should work only with probabilities or logits, not raw audio.

### 7. Ensemble Layer

- `SoftVotingEnsembler` combines calibrated probabilities.
- `WeightOptimizer` searches for the best ensemble weights.
- `ConfidenceGate` routes uncertain predictions to the most confident branch.

These classes should be pure and easy to unit test.

### 8. Training Layer

- `TrainingPipeline` orchestrates the full training flow.
- `Trainer` handles epochs, optimizer, scheduler, and early stopping.
- `Evaluator` computes metrics and confusion matrices.
- `CheckpointManager` persists model state and metadata.

The trainer should not own feature extraction or data discovery logic.

### 9. Inference Layer

- `Predictor` loads the trained artifacts and runs inference end to end.
- `BatchInferenceEngine` handles micro-batching and concurrency.
- `ResultWriter` writes `results.txt`, `time.txt`, and optional metadata.

Inference should be deterministic and should not depend on training-only state.

## Dependency Rules

Use the following direction of dependencies:

```text
cli -> application services -> training/inference -> models/features/preprocessing -> domain
```

Rules:

- `domain` depends on nothing else.
- `preprocessing`, `features`, and `data` can depend on `domain` and `utils`.
- `models` can depend on `features`, `domain`, and framework libraries.
- `training` and `inference` can depend on everything below them.
- `cli` should only call high-level application services.

## Suggested Class Responsibilities

| Class | Responsibility |
|---|---|
| `PreprocessingPipeline` | Apply loading, denoising, normalization, trimming, and padding in one predictable flow |
| `SpectrogramExtractor` | Produce tensor input for Branch A |
| `HandcraftedFeatureExtractor` | Produce tabular features for Branch B and C |
| `BranchAModel` | Deep spectrogram classifier |
| `BranchBModel` | Sequence-based classifier for handcrafted features |
| `BranchCModel` | Classical ensemble for CPU-friendly inference |
| `TemperatureScaler` | Post-hoc calibration of branch probabilities |
| `SoftVotingEnsembler` | Weighted probability fusion |
| `ConfidenceGate` | Hard-sample routing |
| `TrainingPipeline` | End-to-end orchestration of data, training, calibration, and evaluation |
| `Predictor` | End-to-end inference entrypoint |

## Recommended Application Flow

### Training

1. Load config.
2. Scan dataset.
3. Split data.
4. Build preprocessing and feature extractors.
5. Train Branch A, Branch B, and Branch C.
6. Calibrate all branches.
7. Optimize ensemble weights.
8. Evaluate and save artifacts.

### Inference

1. Load saved artifacts.
2. Preprocess test files in batches.
3. Run branch predictions.
4. Calibrate probabilities.
5. Apply weighted vote and confidence gate.
6. Write outputs in the required order.

## Clean Code Principles Applied

- **Single Responsibility**: one class, one job.
- **Open/Closed**: add a new feature extractor or branch without rewriting the pipeline.
- **Liskov Substitution**: all branch models expose the same prediction contract.
- **Interface Segregation**: separate training, inference, and persistence interfaces.
- **Dependency Inversion**: orchestration depends on abstractions, not concrete model details.

## Practical Refactoring Plan

1. Extract configuration into dataclasses or YAML-backed config objects.
2. Move preprocessing functions into dedicated classes.
3. Split feature extraction into spectrogram and handcrafted extractors.
4. Introduce branch model wrappers around the current implementations.
5. Add a predictor service for inference and artifact loading.
6. Move notebook logic into training and inference entrypoints.

## Notes For This Project

- The current prototype is already conceptually aligned with this architecture; the main missing step is separation into modules and interfaces.
- The notebook can remain as an experimentation surface, but production code should live in the package structure above.
- If the Cython preprocessing stage is added later, it can replace the internals of `NoiseReducer` or `PreprocessingPipeline` without changing the public API.

## Outcome

This structure keeps the research pipeline intact while making the project easier to extend, test, and deploy. It also creates a clean path from notebook experimentation to a production-quality training and inference stack.